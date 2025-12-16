#!/usr/bin/env python3
"""
Calculate cross-subject similarity for human responses.

For VQA questions: Calculate pairwise similarity between all subjects' answers
For MMStar (MC): Create confusion matrix showing agreement patterns

Outputs CSV files with per-subject per-question similarity metrics.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple


def compute_similarity(gt: str, pred: str, encoder) -> float:
    """Compute embedding similarity."""
    if encoder is None or not gt or not pred:
        return 0.0
    try:
        pred = pred.strip().lower()
        gt = str(gt).strip().lower()
        emb = encoder.encode([pred, gt])
        similarities = encoder.similarity(emb, emb)
        return float(similarities[1, 0])
    except:
        return 0.0


def load_human_responses(mc_path: str, vqa_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load human MC and VQA responses."""
    with open(mc_path, 'r') as f:
        mc_data = json.load(f)
    with open(vqa_path, 'r') as f:
        vqa_data = json.load(f)
    return mc_data, vqa_data


def group_by_question(responses: List[Dict]) -> Dict[str, List[Dict]]:
    """Group responses by question ID."""
    grouped = defaultdict(list)
    for resp in responses:
        qid = str(resp.get('qid', ''))
        if qid:
            grouped[qid].append(resp)
    return grouped


def calculate_mc_confusion_matrix(responses: List[Dict]) -> Dict:
    """
    For MC questions, calculate confusion matrix showing agreement patterns.
    Returns metrics like Cohen's Kappa, percent agreement, and confusion stats.
    """
    answers = [r.get('answer_normalized', '') for r in responses]
    n_raters = len(answers)

    if n_raters < 2:
        return {
            'n_raters': n_raters,
            'percent_agreement': 0.0,
            'majority_answer': answers[0] if answers else '',
            'answer_distribution': {}
        }

    # Count answer distribution
    from collections import Counter
    answer_counts = Counter(answers)
    majority_answer = answer_counts.most_common(1)[0][0] if answer_counts else ''

    # Calculate percent agreement (proportion agreeing with majority)
    majority_count = answer_counts[majority_answer]
    percent_agreement = majority_count / n_raters

    # Calculate all pairwise agreements
    agreements = []
    for i, j in combinations(range(n_raters), 2):
        agreements.append(1 if answers[i] == answers[j] else 0)

    pairwise_agreement = np.mean(agreements) if agreements else 0.0

    return {
        'n_raters': n_raters,
        'percent_agreement': percent_agreement,
        'pairwise_agreement': pairwise_agreement,
        'majority_answer': majority_answer,
        'answer_distribution': dict(answer_counts)
    }


def calculate_vqa_similarity_matrix(responses: List[Dict], encoder) -> Dict:
    """
    For VQA questions, calculate pairwise similarity between all subjects.
    Returns average similarity, min, max, and per-pair similarities.
    """
    answers = [r.get('answer_normalized', '') for r in responses]
    participant_ids = [r.get('participant_id', '') for r in responses]
    n_raters = len(answers)

    if n_raters < 2:
        return {
            'n_raters': n_raters,
            'mean_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
            'pairwise_similarities': []
        }

    # Calculate all pairwise similarities
    similarities = []
    pairwise_details = []

    for i, j in combinations(range(n_raters), 2):
        sim = compute_similarity(answers[i], answers[j], encoder)
        similarities.append(sim)
        pairwise_details.append({
            'subject_1': participant_ids[i],
            'subject_2': participant_ids[j],
            'answer_1': answers[i],
            'answer_2': answers[j],
            'similarity': sim
        })

    return {
        'n_raters': n_raters,
        'mean_similarity': np.mean(similarities) if similarities else 0.0,
        'min_similarity': np.min(similarities) if similarities else 0.0,
        'max_similarity': np.max(similarities) if similarities else 0.0,
        'std_similarity': np.std(similarities) if similarities else 0.0,
        'pairwise_similarities': pairwise_details
    }


def create_per_subject_dataframe_mc(grouped_responses: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Create dataframe with one row per (subject, question) for MC data.
    Includes their answer and how it compares to others.
    """
    rows = []

    for qid, responses in grouped_responses.items():
        confusion = calculate_mc_confusion_matrix(responses)
        majority_answer = confusion['majority_answer']

        for resp in responses:
            subject_id = resp.get('participant_id', '')
            subject_answer = resp.get('answer_normalized', '')

            # Calculate if this subject agrees with majority
            agrees_with_majority = (subject_answer == majority_answer)

            # Calculate how many others chose the same answer
            same_answer_count = sum(1 for r in responses
                                   if r.get('answer_normalized', '') == subject_answer)

            rows.append({
                'qid': qid,
                'subject_id': subject_id,
                'answer': subject_answer,
                'majority_answer': majority_answer,
                'agrees_with_majority': agrees_with_majority,
                'same_answer_count': same_answer_count,
                'total_raters': len(responses),
                'percent_agreement': confusion['percent_agreement'],
                'pairwise_agreement': confusion['pairwise_agreement']
            })

    return pd.DataFrame(rows)


def create_per_subject_dataframe_vqa(grouped_responses: Dict[str, List[Dict]], encoder) -> pd.DataFrame:
    """
    Create dataframe with one row per (subject, question) for VQA data.
    Includes their answer and similarity to all other answers.
    """
    rows = []

    for qid, responses in grouped_responses.items():
        similarity_matrix = calculate_vqa_similarity_matrix(responses, encoder)

        # For each subject, calculate their average similarity to others
        for i, resp in enumerate(responses):
            subject_id = resp.get('participant_id', '')
            subject_answer = resp.get('answer_normalized', '')

            # Calculate this subject's average similarity to all others
            subject_similarities = []
            for j, other_resp in enumerate(responses):
                if i != j:
                    sim = compute_similarity(
                        subject_answer,
                        other_resp.get('answer_normalized', ''),
                        encoder
                    )
                    subject_similarities.append(sim)

            avg_sim_to_others = np.mean(subject_similarities) if subject_similarities else 0.0

            rows.append({
                'qid': qid,
                'subject_id': subject_id,
                'answer': subject_answer,
                'avg_similarity_to_others': avg_sim_to_others,
                'min_similarity_to_others': np.min(subject_similarities) if subject_similarities else 0.0,
                'max_similarity_to_others': np.max(subject_similarities) if subject_similarities else 0.0,
                'total_raters': len(responses),
                'mean_pairwise_similarity': similarity_matrix['mean_similarity']
            })

    return pd.DataFrame(rows)


def main():
    # Paths
    base_dir = Path("/home/work/yuna/HPA/evaluation/scored/humans") 
    mc_path = base_dir / "cleaned_n15_choice.json"
    vqa_path = base_dir / "cleaned_n15_text.json"
    output_dir = base_dir / "cross_subject_analysis"
    output_dir.mkdir(exist_ok=True)

    print("Loading human responses...")
    mc_data, vqa_data = load_human_responses(str(mc_path), str(vqa_path))
    print(f"Loaded {len(mc_data)} MC responses and {len(vqa_data)} VQA responses")

    # Group by question
    print("\nGrouping responses by question...")
    mc_grouped = group_by_question(mc_data)
    vqa_grouped = group_by_question(vqa_data)
    print(f"MC: {len(mc_grouped)} unique questions")
    print(f"VQA: {len(vqa_grouped)} unique questions")

    # Load encoder for VQA similarity
    print("\nLoading sentence transformer for VQA similarity...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Process MC data - create per-subject dataframe
    print("\nProcessing MC data (creating confusion matrices)...")
    mc_per_subject_df = create_per_subject_dataframe_mc(mc_grouped)
    mc_output_path = output_dir / "mc_per_subject_agreement.csv"
    mc_per_subject_df.to_csv(mc_output_path, index=False)
    print(f"✓ Saved MC per-subject data to {mc_output_path}")
    print(f"  {len(mc_per_subject_df)} rows (subject × question pairs)")

    # Process VQA data - calculate similarities
    print("\nProcessing VQA data (calculating cross-subject similarities)...")
    vqa_per_subject_df = create_per_subject_dataframe_vqa(vqa_grouped, encoder)
    vqa_output_path = output_dir / "vqa_per_subject_similarity.csv"
    vqa_per_subject_df.to_csv(vqa_output_path, index=False)
    print(f"✓ Saved VQA per-subject data to {vqa_output_path}")
    print(f"  {len(vqa_per_subject_df)} rows (subject × question pairs)")

    # Generate summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print("\nMC (Multiple Choice) Data:")
    print(f"  Mean pairwise agreement: {mc_per_subject_df['pairwise_agreement'].mean():.3f}")
    print(f"  Mean percent agreement: {mc_per_subject_df['percent_agreement'].mean():.3f}")
    print(f"  Questions with perfect agreement: {(mc_per_subject_df.groupby('qid')['pairwise_agreement'].first() == 1.0).sum()}")
    print(f"  Questions with <50% agreement: {(mc_per_subject_df.groupby('qid')['percent_agreement'].first() < 0.5).sum()}")

    print("\nVQA (Open-ended) Data:")
    print(f"  Mean similarity to others: {vqa_per_subject_df['avg_similarity_to_others'].mean():.3f}")
    print(f"  Mean pairwise similarity: {vqa_per_subject_df['mean_pairwise_similarity'].mean():.3f}")
    print(f"  Questions with high similarity (>0.7): {(vqa_per_subject_df.groupby('qid')['mean_pairwise_similarity'].first() > 0.7).sum()}")
    print(f"  Questions with low similarity (<0.3): {(vqa_per_subject_df.groupby('qid')['mean_pairwise_similarity'].first() < 0.3).sum()}")

    print("\n✓ Cross-subject analysis complete!")
    print(f"✓ Results saved to {output_dir}")


if __name__ == "__main__":
    main()
