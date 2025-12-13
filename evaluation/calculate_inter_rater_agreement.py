#!/usr/bin/env python3
"""
Calculate inter-rater agreement between human annotators.

Computes various agreement metrics including:
- Percent agreement
- Fleiss' Kappa (for multiple raters)
- Cohen's Kappa (average pairwise)
- Krippendorff's Alpha (optional)

Supports both multiple choice (MC) and open-ended (VQA) questions.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from itertools import combinations
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))


# =============================================================================
# Inter-Rater Agreement Metrics
# =============================================================================

def percent_agreement(ratings: List) -> float:
    """
    Calculate simple percent agreement.

    Args:
        ratings: List of ratings from different raters

    Returns:
        Proportion of raters who agree with the most common rating
    """
    if not ratings or len(ratings) < 2:
        return 1.0

    # Most common rating
    mode_count = Counter(ratings).most_common(1)[0][1]
    return mode_count / len(ratings)


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Calculate Fleiss' Kappa for multiple raters.

    Fleiss' Kappa measures agreement among multiple raters on categorical ratings.

    Args:
        ratings_matrix: numpy array of shape (n_items, n_categories)
                       where element [i,j] is the number of raters who assigned
                       item i to category j

    Returns:
        Fleiss' Kappa value (-1 to 1, higher is better)
        1.0 = perfect agreement
        0.0 = agreement by chance
        < 0 = less than chance agreement
    """
    n_items, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]  # Assume same number of raters per item

    # Proportion of all assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)

    # Calculate P_i for each item (proportion of agreement)
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    # Mean of P_i
    P_bar = P_i.mean()

    # Expected proportion of agreement by chance
    P_e = np.sum(p_j ** 2)

    # Fleiss' Kappa
    if P_e == 1.0:
        return 1.0  # Perfect agreement

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def cohen_kappa(rater1: List, rater2: List) -> float:
    """
    Calculate Cohen's Kappa for two raters.

    Args:
        rater1, rater2: Lists of ratings from two raters

    Returns:
        Cohen's Kappa value
    """
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(rater1, rater2)


def average_pairwise_cohen_kappa(ratings_by_rater: List[List]) -> float:
    """
    Calculate average Cohen's Kappa across all pairs of raters.

    Args:
        ratings_by_rater: List of rating lists, one per rater

    Returns:
        Average Cohen's Kappa
    """
    if len(ratings_by_rater) < 2:
        return 1.0

    kappas = []
    for r1, r2 in combinations(range(len(ratings_by_rater)), 2):
        try:
            k = cohen_kappa(ratings_by_rater[r1], ratings_by_rater[r2])
            if not np.isnan(k):
                kappas.append(k)
        except:
            pass

    return np.mean(kappas) if kappas else 0.0


def krippendorff_alpha(ratings_matrix: np.ndarray, level_of_measurement='nominal') -> float:
    """
    Calculate Krippendorff's Alpha.

    More general than Kappa, handles missing data and different measurement levels.

    Args:
        ratings_matrix: numpy array of shape (n_raters, n_items)
        level_of_measurement: 'nominal', 'ordinal', 'interval', or 'ratio'

    Returns:
        Krippendorff's Alpha value
    """
    try:
        import krippendorff
        return krippendorff.alpha(reliability_data=ratings_matrix, level_of_measurement=level_of_measurement)
    except ImportError:
        print("   Warning: krippendorff package not installed, skipping Alpha calculation")
        return None


# =============================================================================
# Per-Question Agreement
# =============================================================================

def calculate_agreement_per_question_mc(data: List[Dict]) -> List[Dict]:
    """
    Calculate inter-rater agreement for each MC question.

    Args:
        data: List of question dicts with 'extracted_choices' field

    Returns:
        List of results with agreement metrics per question
    """
    results = []

    for item in data:
        qid = item.get('qid')
        choices = item.get('extracted_choices', [])

        if not choices or len(choices) < 2:
            continue

        # Percent agreement
        pct_agree = percent_agreement(choices)

        # Fleiss' Kappa
        # Create ratings matrix: (1 item, n_categories)
        categories = ['A', 'B', 'C', 'D']
        ratings_matrix = np.array([[choices.count(cat) for cat in categories]])
        fleiss_k = fleiss_kappa(ratings_matrix)

        result = {
            'qid': qid,
            'num_raters': len(choices),
            'percent_agreement': pct_agree,
            'fleiss_kappa': fleiss_k,
            'choices': choices,
            'choice_distribution': dict(Counter(choices)),
            'majority_choice': Counter(choices).most_common(1)[0][0] if choices else None,
        }

        # Add ground truth info if available
        if 'answer' in item:
            result['ground_truth'] = item['answer']
            result['majority_correct'] = result['majority_choice'] == item['answer'].strip().upper()[0]

        results.append(result)

    return results


def calculate_agreement_per_question_vqa(data: List[Dict]) -> List[Dict]:
    """
    Calculate inter-rater agreement for each VQA question.

    For open-ended text, we normalize answers before computing agreement.

    Args:
        data: List of question dicts with 'answers' field

    Returns:
        List of results with agreement metrics per question
    """
    from evaluation.utils import normalize_answer

    results = []

    for item in data:
        qid = item.get('qid')
        answers = item.get('answers', [])

        if not answers or len(answers) < 2:
            continue

        # Normalize answers
        normalized = [normalize_answer(ans) for ans in answers]

        # Percent agreement
        pct_agree = percent_agreement(normalized)

        # For Fleiss' Kappa, we need to treat unique normalized answers as categories
        unique_answers = sorted(set(normalized))

        if len(unique_answers) > 1:
            ratings_matrix = np.array([[normalized.count(ans) for ans in unique_answers]])
            fleiss_k = fleiss_kappa(ratings_matrix)
        else:
            fleiss_k = 1.0  # Perfect agreement

        result = {
            'qid': qid,
            'num_raters': len(answers),
            'percent_agreement': pct_agree,
            'fleiss_kappa': fleiss_k,
            'answers': answers,
            'normalized_answers': normalized,
            'answer_distribution': dict(Counter(normalized)),
            'majority_answer': Counter(normalized).most_common(1)[0][0] if normalized else None,
        }

        # Add VQA accuracy if available
        if 'mean_accuracy' in item:
            result['mean_accuracy'] = item['mean_accuracy']

        results.append(result)

    return results


# =============================================================================
# Overall Agreement Across All Questions
# =============================================================================

def calculate_overall_agreement_mc(data: List[Dict]) -> Dict:
    """
    Calculate overall inter-rater agreement across all MC questions.

    Args:
        data: List of question dicts

    Returns:
        Dict with overall agreement metrics
    """
    all_choices = []

    # Collect all ratings for Fleiss' Kappa
    for item in data:
        choices = item.get('extracted_choices', [])
        if choices and len(choices) >= 2:
            all_choices.append(choices)

    if not all_choices:
        return {}

    # Overall Fleiss' Kappa
    categories = ['A', 'B', 'C', 'D']
    ratings_matrix = np.array([[choices.count(cat) for cat in categories] for choices in all_choices])
    overall_fleiss = fleiss_kappa(ratings_matrix)

    # Average percent agreement
    avg_pct_agree = np.mean([percent_agreement(choices) for choices in all_choices])

    # Distribution of rater counts
    rater_counts = [len(choices) for choices in all_choices]

    results = {
        'num_questions': len(all_choices),
        'total_ratings': sum(rater_counts),
        'avg_raters_per_question': np.mean(rater_counts),
        'min_raters': min(rater_counts),
        'max_raters': max(rater_counts),
        'overall_fleiss_kappa': overall_fleiss,
        'avg_percent_agreement': avg_pct_agree,
    }

    return results


def calculate_overall_agreement_vqa(data: List[Dict]) -> Dict:
    """
    Calculate overall inter-rater agreement across all VQA questions.

    Args:
        data: List of question dicts

    Returns:
        Dict with overall agreement metrics
    """
    from evaluation.utils import normalize_answer

    all_answers = []

    for item in data:
        answers = item.get('answers', [])
        if answers and len(answers) >= 2:
            normalized = [normalize_answer(ans) for ans in answers]
            all_answers.append(normalized)

    if not all_answers:
        return {}

    # Average percent agreement
    avg_pct_agree = np.mean([percent_agreement(answers) for answers in all_answers])

    # Distribution of rater counts
    rater_counts = [len(answers) for answers in all_answers]

    # For overall Fleiss', we need consistent categories across questions
    # This is challenging for open-ended text, so we report per-question average instead
    per_q_fleiss = []
    for answers in all_answers:
        unique_answers = sorted(set(answers))
        if len(unique_answers) > 1:
            ratings_matrix = np.array([[answers.count(ans) for ans in unique_answers]])
            fleiss_k = fleiss_kappa(ratings_matrix)
            per_q_fleiss.append(fleiss_k)
        else:
            per_q_fleiss.append(1.0)

    results = {
        'num_questions': len(all_answers),
        'total_ratings': sum(rater_counts),
        'avg_raters_per_question': np.mean(rater_counts),
        'min_raters': min(rater_counts),
        'max_raters': max(rater_counts),
        'avg_fleiss_kappa': np.mean(per_q_fleiss),
        'avg_percent_agreement': avg_pct_agree,
    }

    return results


# =============================================================================
# Main Processing
# =============================================================================

def calculate_inter_rater_agreement(
    mc_file: str = None,
    vqa_file: str = None,
    output_dir: str = None
):
    """
    Calculate inter-rater agreement for human annotations.

    Args:
        mc_file: Path to MC human responses CSV
        vqa_file: Path to VQA human responses CSV
        output_dir: Directory to save results
    """
    print("\n" + "="*80)
    print("INTER-RATER AGREEMENT ANALYSIS")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Process MC data
    if mc_file and os.path.exists(mc_file):
        print(f"\n📊 Processing MC (Multiple Choice) Data")
        print(f"   File: {mc_file}")

        mc_data = pd.read_csv(mc_file)
        mc_data['extracted_choices'] = mc_data['extracted_choices'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        mc_results_per_q = calculate_agreement_per_question_mc(mc_data.to_dict('records'))
        mc_overall = calculate_overall_agreement_mc(mc_data.to_dict('records'))

        # Save per-question results
        mc_per_q_file = os.path.join(output_dir, 'mc_inter_rater_per_question.json')
        with open(mc_per_q_file, 'w', encoding='utf-8') as f:
            json.dump(mc_results_per_q, f, indent=2)
        print(f"\n   ✓ Saved per-question results: {mc_per_q_file}")

        # Save overall results
        mc_overall_file = os.path.join(output_dir, 'mc_inter_rater_overall.json')
        with open(mc_overall_file, 'w', encoding='utf-8') as f:
            json.dump(mc_overall, f, indent=2)
        print(f"   ✓ Saved overall results: {mc_overall_file}")

        # Print summary
        print(f"\n   MC Summary:")
        print(f"      Number of questions: {mc_overall['num_questions']}")
        print(f"      Average raters per question: {mc_overall['avg_raters_per_question']:.1f}")
        print(f"      Overall Fleiss' Kappa: {mc_overall['overall_fleiss_kappa']:.4f}")
        print(f"      Average percent agreement: {mc_overall['avg_percent_agreement']:.4f}")

    # Process VQA data
    if vqa_file and os.path.exists(vqa_file):
        print(f"\n📊 Processing VQA (Open-Ended) Data")
        print(f"   File: {vqa_file}")

        vqa_data = pd.read_csv(vqa_file)
        vqa_data['answers'] = vqa_data['answers'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        vqa_results_per_q = calculate_agreement_per_question_vqa(vqa_data.to_dict('records'))
        vqa_overall = calculate_overall_agreement_vqa(vqa_data.to_dict('records'))

        # Save per-question results
        vqa_per_q_file = os.path.join(output_dir, 'vqa_inter_rater_per_question.json')
        with open(vqa_per_q_file, 'w', encoding='utf-8') as f:
            json.dump(vqa_results_per_q, f, indent=2)
        print(f"\n   ✓ Saved per-question results: {vqa_per_q_file}")

        # Save overall results
        vqa_overall_file = os.path.join(output_dir, 'vqa_inter_rater_overall.json')
        with open(vqa_overall_file, 'w', encoding='utf-8') as f:
            json.dump(vqa_overall, f, indent=2)
        print(f"   ✓ Saved overall results: {vqa_overall_file}")

        # Print summary
        print(f"\n   VQA Summary:")
        print(f"      Number of questions: {vqa_overall['num_questions']}")
        print(f"      Average raters per question: {vqa_overall['avg_raters_per_question']:.1f}")
        print(f"      Average Fleiss' Kappa: {vqa_overall['avg_fleiss_kappa']:.4f}")
        print(f"      Average percent agreement: {vqa_overall['avg_percent_agreement']:.4f}")

    # Create combined summary
    summary = {}
    if mc_file and os.path.exists(mc_file):
        summary['mc'] = mc_overall
    if vqa_file and os.path.exists(vqa_file):
        summary['vqa'] = vqa_overall

    summary_file = os.path.join(output_dir, 'inter_rater_agreement_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*80}")

    # Interpretation guide
    print("\n📚 Interpretation Guide:")
    print("   Fleiss' Kappa / Cohen's Kappa:")
    print("      < 0.00: Poor agreement (less than chance)")
    print("      0.00-0.20: Slight agreement")
    print("      0.21-0.40: Fair agreement")
    print("      0.41-0.60: Moderate agreement")
    print("      0.61-0.80: Substantial agreement")
    print("      0.81-1.00: Almost perfect agreement")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate inter-rater agreement for human annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze both MC and VQA
  python calculate_inter_rater_agreement.py \\
      --mc_file evaluation/scored/humans/human_mc_per_question.csv \\
      --vqa_file evaluation/scored/humans/human_vqa_per_question.csv \\
      --output_dir evaluation/scored/humans/agreement

  # Analyze only MC
  python calculate_inter_rater_agreement.py \\
      --mc_file evaluation/scored/humans/human_mc_per_question.csv \\
      --output_dir evaluation/scored/humans/agreement
        """
    )

    parser.add_argument("--mc_file", type=str,
                       default="evaluation/scored/humans/human_mc_per_question.csv",
                       help="Path to MC human responses CSV")
    parser.add_argument("--vqa_file", type=str,
                       default="evaluation/scored/humans/human_vqa_per_question.csv",
                       help="Path to VQA human responses CSV")
    parser.add_argument("--output_dir", type=str,
                       default="evaluation/scored/humans/agreement",
                       help="Output directory for agreement metrics")

    args = parser.parse_args()

    calculate_inter_rater_agreement(
        mc_file=args.mc_file,
        vqa_file=args.vqa_file,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
