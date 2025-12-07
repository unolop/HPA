#!/usr/bin/env python3
"""
Answer Aggregation with Global Answer Mapping

Instead of caching per unique-answer-set, we build a global mapping:
  original_answer → canonical_answer

New answers are batched and sent to LLM for clustering.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class AnswerAggregator:
    """
    Global answer mapping approach.
    
    Cache format:
    {
        "answer_map": {
            "yes": "yes",
            "Yeah": "yes",
            "yes 53": "yes",
            "3": "3",
            "three": "3"
        },
        "canonical_set": ["yes", "no", "3", ...]
    }
    """
    
    def __init__(
        self,
        cache_path: str = "answer_mapping_cache.json",
        client: Optional["OpenAI"] = None,
        model: str = "gpt-4o-mini",
        batch_size: int = 50  # Batch new answers for efficiency
    ):
        self.cache_path = Path(cache_path)
        self.client = client
        self.model = model
        self.batch_size = batch_size
        
        # Load or initialize cache
        self.cache = self._load_cache()
        self.answer_map: Dict[str, str] = self.cache.get("answer_map", {})
        self.canonical_set: Set[str] = set(self.cache.get("canonical_set", []))
        
        # Pending answers to be clustered
        self.pending_answers: Set[str] = set()
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _load_cache(self) -> Dict:
        if self.cache_path.exists():
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        return {"answer_map": {}, "canonical_set": []}
    
    def _save_cache(self):
        self.cache = {
            "answer_map": self.answer_map,
            "canonical_set": list(self.canonical_set)
        }
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def _get_canonical(self, answer: str) -> Optional[str]:
        """Look up canonical form, return None if unknown."""
        return self.answer_map.get(answer)
    
    def _cluster_new_answers(self, new_answers: List[str]) -> Dict[str, str]:
        """
        Cluster new answers with existing canonical set.
        Returns mapping: new_answer → canonical
        """
        if not new_answers:
            return {}
        
        # Include existing canonicals for context
        existing_canonicals = list(self.canonical_set)
        
        prompt = self._get_clustering_prompt(new_answers, existing_canonicals)
        
        if self.client is None:
            # No API - identity mapping
            return {a: a for a in new_answers}
        
        response = self._call_llm(prompt)
        result = self._parse_response(response)
        
        # Build mapping from response
        new_map = {}
        for group in result.get("grouped_answers", []):
            canonical = group["canonical_answer"]
            for orig in group["original_answers_in_group"]:
                if orig in new_answers:
                    new_map[orig] = canonical
        
        # Fallback for any missed answers
        for a in new_answers:
            if a not in new_map:
                new_map[a] = a
        
        return new_map
    
    def _get_clustering_prompt(
        self, 
        new_answers: List[str], 
        existing_canonicals: List[str]
    ) -> str:
        new_str = json.dumps(new_answers, ensure_ascii=False)
        existing_str = json.dumps(existing_canonicals, ensure_ascii=False) if existing_canonicals else "[]"
        
        return f"""You are a VQA answer normalizer. Map new answers to canonical forms.

New answers to classify:
{new_str}

Existing canonical answers (reuse if semantically equivalent):
{existing_str}

Instructions:
1. Group semantically identical answers (e.g., "yes"/"Yeah"/"yep", "3"/"three")
2. If a new answer matches an existing canonical, use that canonical
3. If a new answer is unique, it becomes its own canonical
4. Return ONLY valid JSON

Output format:
{{
    "grouped_answers": [
        {{"canonical_answer": "yes", "original_answers_in_group": ["Yeah", "yep"]}},
        {{"canonical_answer": "3", "original_answers_in_group": ["three", "Three"]}}
    ]
}}"""

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        import time
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1000,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  ⚠ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        return ""

    def _parse_response(self, response: str) -> Dict:
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"grouped_answers": []}

    def _process_pending(self):
        """Process all pending answers in batches."""
        if not self.pending_answers:
            return
        
        pending_list = list(self.pending_answers)
        print(f"Processing {len(pending_list)} new answers...")
        
        # Process in batches
        for i in range(0, len(pending_list), self.batch_size):
            batch = pending_list[i:i + self.batch_size]
            new_map = self._cluster_new_answers(batch)
            
            # Update global map
            for orig, canonical in new_map.items():
                self.answer_map[orig] = canonical
                self.canonical_set.add(canonical)
        
        self.pending_answers.clear()
        self._save_cache()

    def aggregate(self, answers_with_conf: List[Dict]) -> Dict[str, float]:
        """
        Aggregate confidences using global answer mapping.
        
        Args:
            answers_with_conf: [{'answer': '2', 'confidence': 0.5}, ...]
        
        Returns:
            {canonical_answer: normalized_confidence}
        """
        # Check which answers need clustering
        for item in answers_with_conf:
            ans = item['answer_normalized']
            if ans not in self.answer_map:
                self.cache_misses += 1
                self.pending_answers.add(ans)
            else:
                self.cache_hits += 1
        
        # Process any pending answers
        if self.pending_answers:
            self._process_pending()
        
        # Aggregate using the map
        conf_sums = defaultdict[Any, float](float)
        for item in answers_with_conf:
            ans = item['answer']
            conf = item['confidence']
            canonical = self.answer_map.get(ans, ans)
            conf_sums[canonical] += conf
        
        # Normalize
        total = sum(conf_sums.values())
        if total > 0:
            return {k: v / total for k, v in conf_sums.items()}
        return dict(conf_sums)

    def process_dataset(
        self,
        data: Dict[str, List[Dict]],
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Process entire dataset."""
        # First pass: collect all unique answers
        all_answers = set()
        for answers_with_conf in data.values():
            for item in answers_with_conf:
                all_answers.add(item['answer'])
        
        # Find new answers
        new_answers = [a for a in all_answers if a not in self.answer_map]
        if new_answers and verbose:
            print(f"Found {len(new_answers)} new answers to cluster")
        
        # Batch process all new answers at once (more efficient)
        self.pending_answers = set(new_answers)
        self._process_pending()
        
        # Now aggregate each question
        results = {}
        for i, (qid, answers_with_conf) in enumerate(data.items()):
            results[qid] = self.aggregate(answers_with_conf)
            
            if verbose and i % 500 == 0:
                print(f"Aggregated {i}/{len(data)}...")
        
        if verbose:
            print(f"\nDone! Total answers in map: {len(self.answer_map)}")
            print(f"Canonical answers: {len(self.canonical_set)}")
        
        return results

    def get_stats(self) -> Dict:
        return {
            "total_answers_mapped": len(self.answer_map),
            "canonical_answers": len(self.canonical_set),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
    
    def lookup(self, answer: str) -> str:
        """Direct lookup - returns canonical or original if unknown."""
        return self.answer_map.get(answer, answer)

    def show_mapping(self, limit: int = 20):
        """Print sample of the mapping."""
        print(f"\n--- Answer Mapping (showing {limit}/{len(self.answer_map)}) ---")
        for i, (orig, canon) in enumerate(self.answer_map.items()):
            if i >= limit:
                break
            if orig != canon:
                print(f"  '{orig}' → '{canon}'")
            else:
                print(f"  '{orig}' (canonical)")


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    example_data = {
        '100': [
            {'answer': 'yes', 'confidence': 1.0},
            {'answer': 'Yeah', 'confidence': 0.8},
            {'answer': 'no', 'confidence': 0.3},
        ],
        '101': [
            {'answer': 'yes', 'confidence': 0.9},  # Already seen
            {'answer': 'yep', 'confidence': 0.7},  # New but similar
            {'answer': 'no', 'confidence': 0.2},   # Already seen
        ],
        '102': [
            {'answer': '3', 'confidence': 1.0},
            {'answer': 'three', 'confidence': 0.8},
            {'answer': 'Three', 'confidence': 0.6},
        ],
    }
    
    print("=" * 60)
    print("Testing Global Answer Mapping (no API)")
    print("=" * 60)
    
    agg = AnswerAggregator(cache_path="/tmp/answer_map_v2.json")
    results = agg.process_dataset(example_data)
    
    for qid, dist in results.items():
        print(f"QID {qid}: {dist}")
    
    agg.show_mapping()
    print(f"\nStats: {agg.get_stats()}")
    
    # Show cache file
    print("\n--- Cache file ---")
    with open("/tmp/answer_map_v2.json") as f:
        print(f.read())