#!/usr/bin/env python3
"""
Answer Aggregation Pipeline with Caching

- Caches LLM clustering results to avoid redundant API calls
- Cache key: sorted unique answers (order-independent)
"""

import json
import re
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class AnswerAggregator:
    def __init__(
        self,
        cache_path: str = "answer_clustering_cache.json",
        client: Optional["OpenAI"] = None,
        model: str = "gpt-4o-mini"
    ):
        self.cache_path = Path(cache_path)
        print('Cache file stored at', self.cache_path)
        self.client = client
        self.model = model
        self.cache = self._load_cache()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _load_cache(self) -> Dict:
        """Load existing cache from disk."""
        if self.cache_path.exists():
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def _get_cache_key(self, answers: List[str]) -> str:
        """
        Generate cache key from unique answers.
        Order-independent: ['yes', 'no'] == ['no', 'yes']
        """
        unique_sorted = sorted(set(answers))
        key_str = json.dumps(unique_sorted, ensure_ascii=False)
        # Use hash for very long answer lists
        if len(key_str) > 200:
            return hashlib.md5(key_str.encode()).hexdigest()
        return key_str
    
    def _get_grouping_prompt(self, unique_answers: List[str]) -> str:
        """Build prompt for answer clustering."""
        answers_str = json.dumps(unique_answers, ensure_ascii=False)
        
        return f"""You are an expert VQA data processor. Group semantically similar answers.
                Answers to group:
                {answers_str}

                Instructions:
                1. Group answers that are semantically identical (e.g., "three" and "3", "yes" and "Yeah") and make sure the phrase makes sense by itself.
                2. canonical_answer should be the most common/concise form
                3. Unique answers form their own group
                4. Return ONLY valid JSON, no extra text
                5. Ignore any inappropriate (harmful) or sexual terms that can harm safety assign to the other safe canonical answer. 

                Output format:
                {{
                    "grouped_answers": [
                        {{"canonical_answer": "3", "original_answers_in_group": ["3", "three"]}},
                        {{"canonical_answer": "yes", "original_answers_in_group": ["yes", "Yeah"]}}
                    ]
                }}"""

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM API with retries."""
        import time
        
        if self.client is None:
            raise ValueError("OpenAI client not provided")
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  ⚠ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return ""

    def _parse_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {"grouped_answers": []}

    def _cluster_answers(self, unique_answers: List[str]) -> Dict:
        """
        Get clustering for answers, using cache if available.
        """
        cache_key = self._get_cache_key(unique_answers)
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Simple case: single answer or all identical
        if len(unique_answers) <= 1:
            result = {
                "grouped_answers": [
                    {"canonical_answer": a, "original_answers_in_group": [a]}
                    for a in unique_answers
                ]
            }
            self.cache[cache_key] = result
            return result
        
        # Check if all answers are simple (numeric, yes/no)
        # Skip LLM for trivial cases
        if self._is_trivial_case(unique_answers):
            result = {
                "grouped_answers": [
                    {"canonical_answer": a, "original_answers_in_group": [a]}
                    for a in unique_answers
                ]
            }
            self.cache[cache_key] = result
            return result
        
        # Call LLM for complex cases
        if self.client is None:
            # No client - return identity mapping
            result = {
                "grouped_answers": [
                    {"canonical_answer": a, "original_answers_in_group": [a]}
                    for a in unique_answers
                ]
            }
        else:
            # breakpoint()
            prompt = self._get_grouping_prompt(unique_answers)
            response = self._call_llm(prompt)
            result = self._parse_response(response)
            
            # Validate result covers all answers
            if not self._validate_clustering(unique_answers, result):
                # Fallback to identity mapping
                result = {
                    "grouped_answers": [
                        {"canonical_answer": a, "original_answers_in_group": [a]}
                        for a in unique_answers
                    ]
                }
        
        # Save to cache
        self.cache[cache_key] = result
        self._save_cache()
        
        return result

    def _is_trivial_case(self, answers: List[str]) -> bool:
        """Check if answers are trivial (no clustering needed)."""
        trivial_patterns = {
            # All single digits
            all(a.isdigit() and len(a) == 1 for a in answers),
            # All yes/no
            all(a.lower() in ['yes', 'no'] for a in answers),
            # All single words, no variations
            len(answers) <= 2 and all(len(a.split()) == 1 for a in answers),
        }
        return any(trivial_patterns)

    def _validate_clustering(self, unique_answers: List[str], result: Dict) -> bool:
        """Validate that clustering covers all original answers."""
        covered = set()
        for group in result.get("grouped_answers", []):
            covered.update(group.get("original_answers_in_group", []))
        return set(unique_answers).issubset(covered)

    def aggregate(
        self,
        answers_with_conf: List[Dict]
    ) -> Dict[str, float]:
        
        unique_answers = list(set(item['answer_normalized'] for item in answers_with_conf))
        
        # Get clustering (from cache or LLM)
        grouped_data = self._cluster_answers(unique_answers)
        
        # Build mapping: original -> canonical
        original_to_canonical = {}
        for group in grouped_data.get("grouped_answers", []):
            canonical = group["canonical_answer"]
            for original in group["original_answers_in_group"]:
                original_to_canonical[original] = canonical
        
        # Sum confidences per canonical answer
        conf_sums = defaultdict(float)
        for item in answers_with_conf:
            ans = item['answer_normalized']
            conf = item['confidence']
            canonical = original_to_canonical.get(ans, ans)
            conf_sums[canonical] += conf
        
        # Normalize
        total = sum(conf_sums.values())
        if total > 0:
            normalized = {k: v / total for k, v in conf_sums.items()}
        else:
            normalized = dict(conf_sums)
        
        return normalized

    def process_dataset(
        self,
        data: Dict[str, List[Dict]],
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Process entire dataset.
        
        Args:
            data: {qid: [{'answer': '2', 'confidence': 0.5}, ...], ...}
        
        Returns:
            {qid: {canonical_answer: confidence}, ...}
        """
        results = {}
        
        for i, (qid, answers_with_conf) in enumerate(data.items()):
            if verbose and i % 100 == 0:
                print(f"Processing {i}/{len(data)}... (cache hits: {self.cache_hits}, misses: {self.cache_misses})")
            
            results[qid] = self.aggregate(answers_with_conf)
        
        if verbose:
            print(f"\nDone! Cache hits: {self.cache_hits}, misses: {self.cache_misses}")
            print(f"Cache saved to: {self.cache_path}")
        
        return results

    def get_cache_stats(self) -> Dict:
        """Return cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }


# ============================================================
# Convenience functions
# ============================================================

def aggregate_answers(
    data: Dict[str, List[Dict]],
    cache_path: str = "./answer_clustering_cache.json",
    client: Optional["OpenAI"] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    One-line interface for answer aggregation.
    
    Example:
        results = aggregate_answers(data, client=OpenAI())
    """
    aggregator = AnswerAggregator(cache_path, client, model)
    return aggregator.process_dataset(data, verbose)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example data
    example_data = {
        '664': [
            {'answer': '2', 'confidence': 0.5},
            {'answer': '3', 'confidence': 1.0},
            {'answer': '1', 'confidence': 0.75},
            {'answer': '1', 'confidence': 1.0},
            {'answer': '2', 'confidence': 0.5},
            {'answer': '1', 'confidence': 0.25},
            {'answer': '2', 'confidence': 0.75},
            {'answer': '2', 'confidence': 0.75},
            {'answer': '2', 'confidence': 0.5},
            {'answer': '2', 'confidence': 0.05},
            {'answer': '1', 'confidence': 0.5},
            {'answer': '1', 'confidence': 0.75},
            {'answer': '1', 'confidence': 0.05},
            {'answer': '1', 'confidence': 0.75},
        ],
        '665': [
            {'answer': 'yes', 'confidence': 1.0},
            {'answer': 'Yes', 'confidence': 0.8},
            {'answer': 'no', 'confidence': 0.3},
            {'answer': 'yeah', 'confidence': 0.9},
        ],
    }
    
    print("=" * 60)
    print("Testing with cache (no API)")
    print("=" * 60)
    
    # Without API - uses identity mapping for non-trivial cases
    aggregator = AnswerAggregator(cache_path="/tmp/test_cache.json")
    
    # First run - cache miss
    print("\n--- First run ---")
    results = aggregator.process_dataset(example_data)
    for qid, dist in results.items():
        print(f"QID {qid}: {dist}")
    
    # Second run - cache hit
    print("\n--- Second run (should hit cache) ---")
    results = aggregator.process_dataset(example_data)
    print(f"Cache stats: {aggregator.get_cache_stats()}")
    
    # Show cache contents
    print("\n--- Cache contents ---")
    print(json.dumps(aggregator.cache, indent=2))
    
    # With API (uncomment to test)
    # print("\n" + "=" * 60)
    # print("Testing with OpenAI API")
    # print("=" * 60)
    # client = OpenAI()
    # aggregator = AnswerAggregator(
    #     cache_path="answer_clustering_cache.json",
    #     client=client
    # )
    # results = aggregator.process_dataset(example_data)