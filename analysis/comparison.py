#!/usr/bin/env python3
"""
Human-Model Comparison Analysis for Blind VQA Study

This script provides comprehensive analysis comparing human and model performance:
1. Per-question correlation (do humans and models get the same questions right/wrong?)
2. Category-level analysis (MMStar categories, VQA question types, SPUBench spurious types)
3. Embedding-based similarity (sentence transformer for open-ended VQA)
4. Confidence calibration analysis
5. Agreement metrics (Cohen's Kappa, Fleiss' Kappa)

Key Metrics:
- Accuracy correlation: Are humans and models accurate on the same questions?
- Answer similarity: Do they give similar answers (even if both wrong)?
- Difficulty correlation: Questions hard for humans also hard for models?
- Confidence correlation: High human confidence → high model confidence?

Usage:
    python analyze_human_model_comparison.py \
        --human_data ./human_data/vqav2/*.csv \
        --model_predictions ./predictions/model_vqav2.json \
        --questions_path ./data/vqav2/questions.json \
        --annotations_path ./data/vqav2/annotations.json \
        --output_dir ./analysis_results \
        --benchmark vqav2
"""

import os
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# For embedding similarity
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Embedding analysis disabled.")

# For statistical tests
try:
    from sklearn.metrics import cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HumanResponse:
    """Single human response to a question."""
    qid: str
    answer: str
    confidence: int
    time_spent: float
    participant_id: str = ""


@dataclass
class QuestionData:
    """All data for a single question."""
    qid: str
    question_text: str
    ground_truth: str
    all_gt_answers: List[str] = field(default_factory=list)
    category: str = ""  # MMStar category / VQA question type / SPUBench type
    subcategory: str = ""
    
    # Human responses
    human_responses: List[HumanResponse] = field(default_factory=list)
    human_majority_answer: str = ""
    human_accuracy: float = 0.0  # % of humans correct
    human_mean_confidence: float = 0.0
    human_agreement: float = 0.0  # Inter-annotator agreement
    
    # Model predictions (can have multiple models)
    model_predictions: Dict[str, str] = field(default_factory=dict)  # model_name -> prediction
    model_accuracies: Dict[str, bool] = field(default_factory=dict)  # model_name -> is_correct
    model_confidences: Dict[str, float] = field(default_factory=dict)  # model_name -> confidence


@dataclass
class AnalysisResults:
    """Results from human-model comparison analysis."""
    # Overall metrics
    human_accuracy: float = 0.0
    model_accuracies: Dict[str, float] = field(default_factory=dict)
    
    # Correlation metrics
    accuracy_correlation: Dict[str, float] = field(default_factory=dict)  # Spearman r
    accuracy_correlation_pvalue: Dict[str, float] = field(default_factory=dict)
    
    # Agreement metrics
    cohen_kappa: Dict[str, float] = field(default_factory=dict)
    
    # Per-category results
    category_results: Dict[str, Dict] = field(default_factory=dict)
    
    # Embedding similarity (for VQA)
    embedding_similarity: Dict[str, float] = field(default_factory=dict)
    
    # Difficulty analysis
    difficulty_correlation: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Data Loading
# =============================================================================

def load_human_data(
    csv_files: List[str],
    questions_mapping: Dict[str, str] = None,
) -> Dict[str, List[HumanResponse]]:
    """
    Load human responses from CSV files.
    
    Returns:
        Dictionary mapping qid -> list of HumanResponse
    """
    responses = defaultdict(list)
    
    for csv_file in csv_files:
        participant_id = Path(csv_file).stem
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row['qid'])
                
                if questions_mapping and qid not in questions_mapping:
                    continue
                
                responses[qid].append(HumanResponse(
                    qid=qid,
                    answer=str(row['answer']),
                    confidence=int(row['confidence']),
                    time_spent=float(row['time_spent_seconds']),
                    participant_id=participant_id,
                ))
    
    print(f"Loaded {sum(len(v) for v in responses.values())} human responses for {len(responses)} questions")
    return dict(responses)


def load_model_predictions(
    predictions_path: str,
) -> Dict[str, Dict[str, str]]:
    """
    Load model predictions.
    
    Expected format:
    {
        "model_name": "InternVL3_5-2B",
        "predictions": {
            "qid1": {"answer": "yes", "confidence": 0.95},
            "qid2": {"answer": "no", "confidence": 0.80},
            ...
        }
    }
    
    Or simpler format:
    {
        "qid1": "yes",
        "qid2": "no",
        ...
    }
    
    Returns:
        Dictionary mapping qid -> {model_name: prediction}
    """
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    
    predictions = defaultdict(dict)
    
    if 'predictions' in data:
        # Full format
        model_name = data.get('model_name', 'model')
        for qid, pred in data['predictions'].items():
            if isinstance(pred, dict):
                predictions[qid][model_name] = pred.get('answer', pred.get('prediction', ''))
            else:
                predictions[qid][model_name] = str(pred)
    else:
        # Simple format
        model_name = Path(predictions_path).stem
        for qid, pred in data.items():
            if isinstance(pred, dict):
                predictions[qid][model_name] = pred.get('answer', pred.get('prediction', ''))
            else:
                predictions[qid][model_name] = str(pred)
    
    print(f"Loaded predictions for {len(predictions)} questions")
    return dict(predictions)


def load_questions_and_annotations(
    questions_path: str,
    annotations_path: str,
    benchmark: str = "vqav2",
) -> Dict[str, QuestionData]:
    """
    Load questions and ground truth annotations.
    
    Returns:
        Dictionary mapping qid -> QuestionData
    """
    with open(questions_path, 'r') as f:
        questions_raw = json.load(f)
    
    with open(annotations_path, 'r') as f:
        annotations_raw = json.load(f)
    
    questions = {}
    
    if benchmark == "vqav2":
        q_list = questions_raw.get('questions', questions_raw)
        a_list = annotations_raw.get('annotations', annotations_raw)
        
        q_map = {str(q['question_id']): q for q in q_list}
        a_map = {str(a['question_id']): a for a in a_list}
        
        for qid, q in q_map.items():
            if qid not in a_map:
                continue
            
            ann = a_map[qid]
            all_answers = [a['answer'] for a in ann.get('answers', [])]
            
            # Most common answer as GT
            from collections import Counter
            gt = Counter(all_answers).most_common(1)[0][0] if all_answers else ""
            
            questions[qid] = QuestionData(
                qid=qid,
                question_text=q['question'],
                ground_truth=gt,
                all_gt_answers=list(set(all_answers)),
                category=ann.get('question_type', ''),
                subcategory=ann.get('answer_type', ''),
            )
    
    elif benchmark == "mmstar":
        for item in questions_raw:
            qid = str(item.get('id', item.get('question_id')))
            questions[qid] = QuestionData(
                qid=qid,
                question_text=item['question'],
                ground_truth=item.get('answer', item.get('gt_answer', '')),
                category=item.get('category', item.get('l2_category', '')),
                subcategory=item.get('l3_category', ''),
            )
    
    elif benchmark == "mmspubench":
        for item in questions_raw:
            qid = str(item.get('id'))
            questions[qid] = QuestionData(
                qid=qid,
                question_text=item['question'],
                ground_truth=item.get('answer', ''),
                category=item.get('spurious_type', item.get('bias_type', '')),
                subcategory=item.get('subcategory', ''),
            )
    
    print(f"Loaded {len(questions)} questions from {benchmark}")
    return questions


# =============================================================================
# Answer Normalization & Matching
# =============================================================================

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = str(answer).lower().strip()
    
    # Remove articles
    for article in ['a', 'an', 'the']:
        if answer.startswith(article + ' '):
            answer = answer[len(article)+1:]
    
    # Remove punctuation
    import string
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize common variations
    answer = answer.replace('  ', ' ')
    
    return answer.strip()


def is_correct(prediction: str, ground_truth: str, all_gt_answers: List[str] = None) -> bool:
    """Check if prediction is correct (VQA-style matching)."""
    pred_norm = normalize_answer(prediction)
    
    if all_gt_answers:
        gt_answers = [normalize_answer(a) for a in all_gt_answers]
        return pred_norm in gt_answers
    else:
        return pred_norm == normalize_answer(ground_truth)


def compute_vqa_accuracy(prediction: str, all_gt_answers: List[str]) -> float:
    """
    Compute VQA accuracy score (soft accuracy).
    
    VQA accuracy = min(1, #humans_that_gave_this_answer / 3)
    """
    pred_norm = normalize_answer(prediction)
    gt_norms = [normalize_answer(a) for a in all_gt_answers]
    
    count = sum(1 for a in gt_norms if a == pred_norm)
    return min(1.0, count / 3.0)


# =============================================================================
# Embedding-based Similarity
# =============================================================================

class EmbeddingSimilarityCalculator:
    """Calculate semantic similarity using sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers required for embedding similarity")
        
        self.model = SentenceTransformer(model_name)
        self._cache = {}
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text (with caching)."""
        if text not in self._cache:
            self._cache[text] = self.model.encode(text, convert_to_numpy=True)
        return self._cache[text]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def compute_answer_similarity(
        self,
        prediction: str,
        ground_truths: List[str],
    ) -> float:
        """Compute max similarity between prediction and any ground truth."""
        if not ground_truths:
            return 0.0
        
        similarities = [self.compute_similarity(prediction, gt) for gt in ground_truths]
        return max(similarities)


# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_human_statistics(responses: List[HumanResponse], ground_truth: str, all_gt: List[str]) -> Dict:
    """Compute statistics for human responses to a question."""
    if not responses:
        return {}
    
    # Answer distribution
    answer_counts = defaultdict(int)
    for r in responses:
        answer_counts[normalize_answer(r.answer)] += 1
    
    # Majority answer
    majority_answer = max(answer_counts.keys(), key=lambda x: answer_counts[x])
    
    # Accuracy (% of humans correct)
    correct_count = sum(1 for r in responses if is_correct(r.answer, ground_truth, all_gt))
    accuracy = correct_count / len(responses)
    
    # Mean confidence
    mean_confidence = np.mean([r.confidence for r in responses])
    
    # Agreement (entropy-based or simple majority %)
    total = sum(answer_counts.values())
    agreement = max(answer_counts.values()) / total  # % giving majority answer
    
    # Confidence by correctness
    correct_confidences = [r.confidence for r in responses if is_correct(r.answer, ground_truth, all_gt)]
    incorrect_confidences = [r.confidence for r in responses if not is_correct(r.answer, ground_truth, all_gt)]
    
    return {
        'majority_answer': majority_answer,
        'accuracy': accuracy,
        'mean_confidence': mean_confidence,
        'agreement': agreement,
        'num_responses': len(responses),
        'num_unique_answers': len(answer_counts),
        'mean_confidence_correct': np.mean(correct_confidences) if correct_confidences else 0,
        'mean_confidence_incorrect': np.mean(incorrect_confidences) if incorrect_confidences else 0,
        'mean_time': np.mean([r.time_spent for r in responses]),
    }


def compute_correlation(
    human_values: List[float],
    model_values: List[float],
    method: str = "spearman",
) -> Tuple[float, float]:
    """
    Compute correlation between human and model values.
    
    Args:
        human_values: List of human metrics (e.g., accuracy per question)
        model_values: List of model metrics
        method: "spearman", "pearson", or "kendall"
        
    Returns:
        (correlation, p-value)
    """
    if len(human_values) != len(model_values):
        raise ValueError("Lists must have same length")
    
    if len(human_values) < 3:
        return 0.0, 1.0
    
    # Remove NaN values
    mask = ~(np.isnan(human_values) | np.isnan(model_values))
    h = np.array(human_values)[mask]
    m = np.array(model_values)[mask]
    
    if len(h) < 3:
        return 0.0, 1.0
    
    if method == "spearman":
        r, p = spearmanr(h, m)
    elif method == "pearson":
        r, p = pearsonr(h, m)
    elif method == "kendall":
        r, p = kendalltau(h, m)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(r), float(p)


def compute_cohen_kappa(
    human_correct: List[bool],
    model_correct: List[bool],
) -> float:
    """Compute Cohen's Kappa for agreement between human and model correctness."""
    if not HAS_SKLEARN:
        return 0.0
    
    return cohen_kappa_score(human_correct, model_correct)


# =============================================================================
# Main Analysis Functions
# =============================================================================

def analyze_per_question(
    questions: Dict[str, QuestionData],
    human_responses: Dict[str, List[HumanResponse]],
    model_predictions: Dict[str, Dict[str, str]],
    embedding_calculator: EmbeddingSimilarityCalculator = None,
) -> Dict[str, QuestionData]:
    """
    Analyze each question with human and model data.
    
    Returns:
        Updated questions dictionary with analysis results
    """
    for qid, q in questions.items():
        # Add human responses
        if qid in human_responses:
            q.human_responses = human_responses[qid]
            
            # Compute human statistics
            human_stats = compute_human_statistics(
                q.human_responses,
                q.ground_truth,
                q.all_gt_answers,
            )
            
            q.human_majority_answer = human_stats.get('majority_answer', '')
            q.human_accuracy = human_stats.get('accuracy', 0.0)
            q.human_mean_confidence = human_stats.get('mean_confidence', 0.0)
            q.human_agreement = human_stats.get('agreement', 0.0)
        
        # Add model predictions
        if qid in model_predictions:
            for model_name, pred in model_predictions[qid].items():
                q.model_predictions[model_name] = pred
                q.model_accuracies[model_name] = is_correct(
                    pred, q.ground_truth, q.all_gt_answers
                )
    
    return questions


def analyze_correlations(
    questions: Dict[str, QuestionData],
    model_names: List[str],
) -> Dict[str, Any]:
    """
    Compute correlation metrics between humans and models.
    
    Returns:
        Dictionary with correlation results
    """
    results = {
        'accuracy_correlation': {},
        'difficulty_correlation': {},
        'cohen_kappa': {},
        'per_question': [],
    }
    
    # Prepare data for correlation
    qids = [qid for qid, q in questions.items() if q.human_responses]
    
    human_acc = [questions[qid].human_accuracy for qid in qids]
    human_conf = [questions[qid].human_mean_confidence for qid in qids]
    
    for model_name in model_names:
        # Model accuracy per question (binary)
        model_acc = [
            1.0 if questions[qid].model_accuracies.get(model_name, False) else 0.0
            for qid in qids
        ]
        
        # Human accuracy (continuous 0-1) correlation with model accuracy
        r, p = compute_correlation(human_acc, model_acc, method="spearman")
        results['accuracy_correlation'][model_name] = {
            'spearman_r': r,
            'p_value': p,
            'significant': p < 0.05,
        }
        
        # Difficulty correlation (inverse of accuracy)
        human_difficulty = [1 - a for a in human_acc]
        model_difficulty = [1 - a for a in model_acc]
        r_diff, p_diff = compute_correlation(human_difficulty, model_difficulty)
        results['difficulty_correlation'][model_name] = {
            'spearman_r': r_diff,
            'p_value': p_diff,
        }
        
        # Cohen's Kappa (binary agreement)
        human_correct = [questions[qid].human_accuracy > 0.5 for qid in qids]  # Majority correct
        model_correct = [questions[qid].model_accuracies.get(model_name, False) for qid in qids]
        kappa = compute_cohen_kappa(human_correct, model_correct)
        results['cohen_kappa'][model_name] = kappa
    
    # Per-question data for visualization
    for qid in qids:
        q = questions[qid]
        entry = {
            'qid': qid,
            'question': q.question_text[:100],
            'category': q.category,
            'human_accuracy': q.human_accuracy,
            'human_confidence': q.human_mean_confidence,
            'human_agreement': q.human_agreement,
            'ground_truth': q.ground_truth,
            'human_majority': q.human_majority_answer,
        }
        for model_name in model_names:
            entry[f'{model_name}_prediction'] = q.model_predictions.get(model_name, '')
            entry[f'{model_name}_correct'] = q.model_accuracies.get(model_name, False)
        
        results['per_question'].append(entry)
    
    return results


def analyze_by_category(
    questions: Dict[str, QuestionData],
    model_names: List[str],
) -> Dict[str, Dict]:
    """
    Analyze human-model comparison grouped by category.
    
    Returns:
        Dictionary with per-category results
    """
    # Group questions by category
    by_category = defaultdict(list)
    for qid, q in questions.items():
        if q.human_responses:
            by_category[q.category].append(q)
    
    results = {}
    
    for category, cat_questions in by_category.items():
        if len(cat_questions) < 3:
            continue
        
        cat_result = {
            'num_questions': len(cat_questions),
            'human_mean_accuracy': np.mean([q.human_accuracy for q in cat_questions]),
            'human_std_accuracy': np.std([q.human_accuracy for q in cat_questions]),
            'human_mean_confidence': np.mean([q.human_mean_confidence for q in cat_questions]),
            'models': {},
        }
        
        for model_name in model_names:
            model_acc = [
                1.0 if q.model_accuracies.get(model_name, False) else 0.0
                for q in cat_questions
            ]
            human_acc = [q.human_accuracy for q in cat_questions]
            
            r, p = compute_correlation(human_acc, model_acc)
            
            cat_result['models'][model_name] = {
                'accuracy': np.mean(model_acc),
                'correlation_with_human': r,
                'correlation_pvalue': p,
            }
        
        results[category] = cat_result
    
    return results


def analyze_embedding_similarity(
    questions: Dict[str, QuestionData],
    model_names: List[str],
    embedding_calculator: EmbeddingSimilarityCalculator,
) -> Dict[str, Any]:
    """
    Analyze answer similarity using sentence embeddings.
    
    This is useful for open-ended VQA where exact match is too strict.
    """
    results = {
        'human_gt_similarity': [],  # Human answers vs ground truth
        'model_gt_similarity': {m: [] for m in model_names},
        'human_model_similarity': {m: [] for m in model_names},
    }
    
    for qid, q in questions.items():
        if not q.human_responses or not q.ground_truth:
            continue
        
        gt_list = q.all_gt_answers if q.all_gt_answers else [q.ground_truth]
        
        # Human majority answer similarity to GT
        if q.human_majority_answer:
            sim = embedding_calculator.compute_answer_similarity(
                q.human_majority_answer, gt_list
            )
            results['human_gt_similarity'].append(sim)
        
        # Model predictions similarity
        for model_name in model_names:
            pred = q.model_predictions.get(model_name, '')
            if pred:
                # Model vs GT
                sim_gt = embedding_calculator.compute_answer_similarity(pred, gt_list)
                results['model_gt_similarity'][model_name].append(sim_gt)
                
                # Model vs Human
                if q.human_majority_answer:
                    sim_human = embedding_calculator.compute_similarity(
                        pred, q.human_majority_answer
                    )
                    results['human_model_similarity'][model_name].append(sim_human)
    
    # Compute summary statistics
    summary = {
        'human_gt': {
            'mean': np.mean(results['human_gt_similarity']),
            'std': np.std(results['human_gt_similarity']),
        }
    }
    
    for model_name in model_names:
        summary[model_name] = {
            'model_gt_mean': np.mean(results['model_gt_similarity'][model_name]),
            'model_gt_std': np.std(results['model_gt_similarity'][model_name]),
            'human_model_mean': np.mean(results['human_model_similarity'][model_name]),
            'human_model_std': np.std(results['human_model_similarity'][model_name]),
        }
    
    return {
        'detailed': results,
        'summary': summary,
    }


def analyze_confidence_calibration(
    questions: Dict[str, QuestionData],
) -> Dict[str, Any]:
    """
    Analyze relationship between human confidence and accuracy.
    
    Tests: Are humans well-calibrated? (high confidence → high accuracy)
    """
    # Group by confidence level
    by_confidence = defaultdict(list)
    
    for qid, q in questions.items():
        for resp in q.human_responses:
            is_correct_resp = is_correct(resp.answer, q.ground_truth, q.all_gt_answers)
            by_confidence[resp.confidence].append(is_correct_resp)
    
    # Compute accuracy per confidence level
    calibration = {}
    for conf in sorted(by_confidence.keys()):
        correct_list = by_confidence[conf]
        calibration[conf] = {
            'num_responses': len(correct_list),
            'accuracy': np.mean(correct_list),
            'expected': conf / 5.0,  # If well-calibrated, conf 5 → 100%, conf 1 → 20%
        }
    
    # Compute calibration error
    conf_levels = []
    acc_levels = []
    for conf, data in calibration.items():
        conf_levels.append(conf / 5.0)  # Normalize to 0-1
        acc_levels.append(data['accuracy'])
    
    calibration_error = np.mean(np.abs(np.array(conf_levels) - np.array(acc_levels)))
    
    return {
        'by_confidence_level': calibration,
        'calibration_error': calibration_error,
        'confidence_accuracy_correlation': compute_correlation(conf_levels, acc_levels)[0],
    }


# =============================================================================
# Report Generation
# =============================================================================

def generate_summary_report(
    questions: Dict[str, QuestionData],
    correlation_results: Dict,
    category_results: Dict,
    embedding_results: Dict = None,
    calibration_results: Dict = None,
    model_names: List[str] = None,
) -> str:
    """Generate a text summary report of the analysis."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("HUMAN-MODEL COMPARISON ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall statistics
    human_accs = [q.human_accuracy for q in questions.values() if q.human_responses]
    lines.append("📊 OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total questions analyzed: {len(human_accs)}")
    lines.append(f"Human mean accuracy: {np.mean(human_accs):.3f} (±{np.std(human_accs):.3f})")
    
    for model_name in (model_names or []):
        model_accs = [
            1.0 if q.model_accuracies.get(model_name, False) else 0.0
            for q in questions.values() if q.human_responses
        ]
        lines.append(f"{model_name} accuracy: {np.mean(model_accs):.3f}")
    
    lines.append("")
    
    # Correlation results
    lines.append("🔗 HUMAN-MODEL CORRELATION")
    lines.append("-" * 40)
    for model_name, corr in correlation_results.get('accuracy_correlation', {}).items():
        sig = "**" if corr.get('significant', False) else ""
        lines.append(f"{model_name}:")
        lines.append(f"  Spearman r = {corr['spearman_r']:.3f} (p={corr['p_value']:.4f}){sig}")
    
    lines.append("")
    lines.append("📐 COHEN'S KAPPA (Agreement)")
    for model_name, kappa in correlation_results.get('cohen_kappa', {}).items():
        interpretation = (
            "Poor" if kappa < 0.2 else
            "Fair" if kappa < 0.4 else
            "Moderate" if kappa < 0.6 else
            "Good" if kappa < 0.8 else
            "Excellent"
        )
        lines.append(f"  {model_name}: κ = {kappa:.3f} ({interpretation})")
    
    lines.append("")
    
    # Category results
    lines.append("📁 PER-CATEGORY ANALYSIS")
    lines.append("-" * 40)
    for category, data in sorted(category_results.items(), key=lambda x: -x[1]['num_questions']):
        lines.append(f"\n{category} (n={data['num_questions']}):")
        lines.append(f"  Human accuracy: {data['human_mean_accuracy']:.3f}")
        for model_name, model_data in data.get('models', {}).items():
            lines.append(f"  {model_name}: acc={model_data['accuracy']:.3f}, r={model_data['correlation_with_human']:.3f}")
    
    # Embedding similarity
    if embedding_results:
        lines.append("")
        lines.append("🔤 EMBEDDING SIMILARITY")
        lines.append("-" * 40)
        summary = embedding_results.get('summary', {})
        lines.append(f"Human-GT similarity: {summary.get('human_gt', {}).get('mean', 0):.3f}")
        for model_name in (model_names or []):
            if model_name in summary:
                lines.append(f"{model_name}-GT: {summary[model_name].get('model_gt_mean', 0):.3f}")
                lines.append(f"{model_name}-Human: {summary[model_name].get('human_model_mean', 0):.3f}")
    
    # Calibration
    if calibration_results:
        lines.append("")
        lines.append("🎯 HUMAN CONFIDENCE CALIBRATION")
        lines.append("-" * 40)
        lines.append(f"Calibration error: {calibration_results['calibration_error']:.3f}")
        lines.append("Accuracy by confidence level:")
        for conf, data in calibration_results.get('by_confidence_level', {}).items():
            lines.append(f"  Conf {conf}: {data['accuracy']:.3f} (n={data['num_responses']})")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_full_analysis(
    human_csv_files: List[str],
    model_predictions_path: str,
    questions_path: str,
    annotations_path: str,
    output_dir: str,
    benchmark: str = "vqav2",
    use_embeddings: bool = True,
) -> Dict[str, Any]:
    """
    Run full human-model comparison analysis.
    
    Returns:
        Dictionary with all analysis results
    """
    print("=" * 80)
    print("HUMAN-MODEL COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    questions = load_questions_and_annotations(questions_path, annotations_path, benchmark)
    human_responses = load_human_data(human_csv_files, {qid: q.question_text for qid, q in questions.items()})
    model_predictions = load_model_predictions(model_predictions_path)
    
    model_names = list(set(
        model_name 
        for preds in model_predictions.values() 
        for model_name in preds.keys()
    ))
    print(f"Models found: {model_names}")
    
    # Initialize embedding calculator
    embedding_calculator = None
    if use_embeddings and HAS_SENTENCE_TRANSFORMERS:
        print("\n🔤 Initializing embedding model...")
        embedding_calculator = EmbeddingSimilarityCalculator()
    
    # Per-question analysis
    print("\n📊 Analyzing per-question metrics...")
    questions = analyze_per_question(
        questions, human_responses, model_predictions, embedding_calculator
    )
    
    # Correlation analysis
    print("\n🔗 Computing correlations...")
    correlation_results = analyze_correlations(questions, model_names)
    
    # Category analysis
    print("\n📁 Analyzing by category...")
    category_results = analyze_by_category(questions, model_names)
    
    # Embedding similarity
    embedding_results = None
    if embedding_calculator:
        print("\n🔤 Computing embedding similarities...")
        embedding_results = analyze_embedding_similarity(
            questions, model_names, embedding_calculator
        )
    
    # Confidence calibration
    print("\n🎯 Analyzing confidence calibration...")
    calibration_results = analyze_confidence_calibration(questions)
    
    # Generate report
    report = generate_summary_report(
        questions,
        correlation_results,
        category_results,
        embedding_results,
        calibration_results,
        model_names,
    )
    
    print("\n" + report)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'correlation': correlation_results,
        'by_category': category_results,
        'embedding_similarity': embedding_results,
        'calibration': calibration_results,
        'summary': {
            'num_questions': len([q for q in questions.values() if q.human_responses]),
            'model_names': model_names,
            'benchmark': benchmark,
        }
    }
    
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write(report)
    
    # Save per-question data for further analysis
    with open(os.path.join(output_dir, 'per_question_data.json'), 'w') as f:
        json.dump(correlation_results['per_question'], f, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Human-Model Comparison Analysis for Blind VQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--human_data", type=str, nargs='+', required=True,
                        help="Human response CSV files")
    parser.add_argument("--model_predictions", type=str, required=True,
                        help="Model predictions JSON file")
    parser.add_argument("--questions_path", type=str, required=True,
                        help="Questions JSON file")
    parser.add_argument("--annotations_path", type=str, required=True,
                        help="Annotations JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--benchmark", type=str, default="vqav2",
                        choices=["vqav2", "mmstar", "mmspubench"],
                        help="Benchmark type")
    parser.add_argument("--no_embeddings", action="store_true",
                        help="Disable embedding similarity analysis")
    
    args = parser.parse_args()
    
    run_full_analysis(
        human_csv_files=args.human_data,
        model_predictions_path=args.model_predictions,
        questions_path=args.questions_path,
        annotations_path=args.annotations_path,
        output_dir=args.output_dir,
        benchmark=args.benchmark,
        use_embeddings=not args.no_embeddings,
    )


if __name__ == "__main__":
    main()