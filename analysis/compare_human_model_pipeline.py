#!/usr/bin/env python3
"""
Complete Pipeline for Human vs Model Analysis on Blind VQA

Compares human and model performance on blind VQA datasets, matched by qid.
Supports filtering by answer_type (choice vs text).

Usage:
    python compare_human_model_pipeline.py \
        --human_results data/training/s1_choice/cleaned_n14_choice.json \
        --model_results outputs/results/model_predictions.jsonl \
        --questions dataset/questions/s1.csv \
        --output_dir analysis/results \
        --answer_type choice
"""

import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


class HumanModelAnalyzer:
    """Analyzes differences between human and model performance on blind VQA."""

    def __init__(self, output_dir: str = "analysis/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.human_data = None
        self.model_data = None
        self.questions = None
        self.merged_data = None

    def load_human_results(self, path: str) -> pd.DataFrame:
        """Load human results from JSON file."""
        print(f"Loading human results from {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        # Flatten if nested by answer_type
        if isinstance(data, dict) and 'choice' in data or 'text' in data:
            # Assuming structure: {answer_type: [responses]}
            all_responses = []
            for answer_type, responses in data.items():
                for resp in responses:
                    resp['answer_type'] = answer_type
                    all_responses.append(resp)
            data = all_responses

        df = pd.DataFrame(data)

        # Normalize answer field
        if 'answer_normalized' not in df.columns and 'answer' in df.columns:
            df['answer_normalized'] = df['answer'].str.lower().str.strip()

        print(f"  Loaded {len(df)} human responses")
        print(f"  Unique questions: {df['qid'].nunique()}")

        return df

    def load_model_results(self, path: str) -> pd.DataFrame:
        """Load model predictions from JSONL file."""
        print(f"Loading model results from {path}")

        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        df = pd.DataFrame(data)

        # Handle different output formats
        if 'output' in df.columns:
            df['model_answer'] = df['output']
        elif 'prediction' in df.columns:
            df['model_answer'] = df['prediction']
        elif 'answer' in df.columns:
            df['model_answer'] = df['answer']

        # Normalize
        if 'model_answer' in df.columns:
            df['model_answer_normalized'] = df['model_answer'].astype(str).str.lower().str.strip()

        print(f"  Loaded {len(df)} model predictions")
        print(f"  Unique questions: {df['qid'].nunique() if 'qid' in df.columns else 'N/A'}")

        return df

    def load_questions(self, path: str) -> pd.DataFrame:
        """Load question metadata."""
        print(f"Loading questions from {path}")

        df = pd.read_csv(path)

        # Standardize qid column
        if 'qid' not in df.columns:
            if 'question_id' in df.columns:
                df['qid'] = df['question_id']
            elif 'index' in df.columns:
                df['qid'] = df['index']

        df['qid'] = df['qid'].astype(str)

        print(f"  Loaded {len(df)} questions")

        return df

    def aggregate_human_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate multiple human responses per question."""
        print("Aggregating human responses by qid...")

        aggregated = []

        for qid, group in df.groupby('qid'):
            # Get most common answer (consensus)
            answer_counts = group['answer_normalized'].value_counts()
            consensus_answer = answer_counts.index[0]
            consensus_count = answer_counts.iloc[0]
            total_responses = len(group)

            # Average confidence
            avg_confidence = group['confidence'].mean() if 'confidence' in group.columns else None

            aggregated.append({
                'qid': qid,
                'human_answer': consensus_answer,
                'human_consensus_rate': consensus_count / total_responses,
                'human_num_responses': total_responses,
                'human_confidence': avg_confidence,
                'answer_type': group['answer_type'].iloc[0] if 'answer_type' in group.columns else None,
                'all_human_answers': group['answer_normalized'].tolist(),
            })

        agg_df = pd.DataFrame(aggregated)
        print(f"  Aggregated to {len(agg_df)} unique questions")

        return agg_df

    def merge_human_model(
        self,
        human_df: pd.DataFrame,
        model_df: pd.DataFrame,
        questions_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Merge human and model results by qid."""
        print("Merging human and model data...")

        # Ensure qid is string
        human_df['qid'] = human_df['qid'].astype(str)
        model_df['qid'] = model_df['qid'].astype(str)

        # Merge
        merged = pd.merge(
            human_df,
            model_df[['qid', 'model_answer_normalized']],
            on='qid',
            how='inner'
        )

        # Add question metadata if available
        if questions_df is not None:
            questions_df['qid'] = questions_df['qid'].astype(str)
            merged = pd.merge(
                merged,
                questions_df[['qid', 'question', 'category', 'answer_type']],
                on='qid',
                how='left',
                suffixes=('', '_q')
            )

            # Use question metadata answer_type if available
            if 'answer_type_q' in merged.columns:
                merged['answer_type'] = merged['answer_type_q'].fillna(merged['answer_type'])
                merged = merged.drop(columns=['answer_type_q'])

        print(f"  Merged dataset: {len(merged)} questions")
        print(f"  Answer types: {merged['answer_type'].value_counts().to_dict() if 'answer_type' in merged.columns else 'N/A'}")

        return merged

    def compute_agreement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute human-model agreement."""
        df = df.copy()

        # Exact match
        df['exact_match'] = (df['human_answer'] == df['model_answer_normalized']).astype(int)

        return df

    def analyze_overall_performance(self, df: pd.DataFrame) -> Dict:
        """Compute overall performance metrics."""
        print("\n=== Overall Performance ===")

        metrics = {
            'total_questions': len(df),
            'agreement_rate': df['exact_match'].mean(),
            'human_avg_consensus': df['human_consensus_rate'].mean(),
            'human_avg_confidence': df['human_confidence'].mean() if 'human_confidence' in df.columns else None,
        }

        print(f"Total questions: {metrics['total_questions']}")
        print(f"Human-Model agreement: {metrics['agreement_rate']:.2%}")
        print(f"Human consensus rate: {metrics['human_avg_consensus']:.2%}")
        if metrics['human_avg_confidence']:
            print(f"Human avg confidence: {metrics['human_avg_confidence']:.2f}")

        return metrics

    def analyze_by_answer_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by answer type."""
        print("\n=== Performance by Answer Type ===")

        if 'answer_type' not in df.columns:
            print("  No answer_type column found")
            return None

        results = []

        for answer_type, group in df.groupby('answer_type'):
            metrics = {
                'answer_type': answer_type,
                'num_questions': len(group),
                'agreement_rate': group['exact_match'].mean(),
                'human_consensus': group['human_consensus_rate'].mean(),
            }

            if 'human_confidence' in group.columns:
                metrics['human_confidence'] = group['human_confidence'].mean()

            results.append(metrics)

        results_df = pd.DataFrame(results)
        print(results_df.to_string())

        return results_df

    def analyze_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by question category."""
        print("\n=== Performance by Category ===")

        if 'category' not in df.columns:
            print("  No category column found")
            return None

        results = []

        for category, group in df.groupby('category'):
            metrics = {
                'category': category,
                'num_questions': len(group),
                'agreement_rate': group['exact_match'].mean(),
                'human_consensus': group['human_consensus_rate'].mean(),
            }

            results.append(metrics)

        results_df = pd.DataFrame(results).sort_values('agreement_rate', ascending=False)
        print(results_df.head(10).to_string())

        return results_df

    def analyze_disagreements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze cases where human and model disagree."""
        print("\n=== Disagreement Analysis ===")

        disagreements = df[df['exact_match'] == 0].copy()

        print(f"Total disagreements: {len(disagreements)} ({len(disagreements)/len(df):.1%})")

        if 'category' in disagreements.columns:
            print("\nTop categories with disagreements:")
            cat_disagree = disagreements['category'].value_counts().head(5)
            print(cat_disagree)

        return disagreements

    def plot_agreement_by_confidence(self, df: pd.DataFrame):
        """Plot agreement rate vs human confidence."""
        if 'human_confidence' not in df.columns:
            print("No confidence data available for plotting")
            return

        print("Plotting agreement by confidence...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Bin by confidence
        df['conf_bin'] = pd.cut(df['human_confidence'], bins=[0, 1, 2, 3, 4, 5], labels=['1', '2', '3', '4', '5'])

        grouped = df.groupby('conf_bin')['exact_match'].agg(['mean', 'count'])

        grouped['mean'].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel('Human Confidence Level')
        ax.set_ylabel('Agreement Rate')
        ax.set_title('Human-Model Agreement vs Human Confidence')
        ax.set_ylim([0, 1])

        # Add counts on top
        for i, (idx, row) in enumerate(grouped.iterrows()):
            ax.text(i, row['mean'] + 0.02, f"n={row['count']}", ha='center', fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / 'agreement_by_confidence.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    def plot_confusion_matrix(self, df: pd.DataFrame, answer_type: str = None):
        """Plot confusion matrix for choice questions."""
        if answer_type:
            df = df[df['answer_type'] == answer_type]

        print(f"Plotting confusion matrix for {answer_type or 'all'} questions...")

        # Get unique answers
        all_answers = sorted(set(df['human_answer'].unique()) | set(df['model_answer_normalized'].unique()))

        # Only plot if reasonable number of categories
        if len(all_answers) > 20:
            print(f"  Too many unique answers ({len(all_answers)}), skipping confusion matrix")
            return

        cm = confusion_matrix(df['human_answer'], df['model_answer_normalized'], labels=all_answers)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_answers, yticklabels=all_answers, ax=ax)
        ax.set_xlabel('Model Answer')
        ax.set_ylabel('Human Answer (Consensus)')
        ax.set_title(f'Confusion Matrix - {answer_type or "All"}')

        plt.tight_layout()
        suffix = f'_{answer_type}' if answer_type else ''
        save_path = self.output_dir / f'confusion_matrix{suffix}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    def plot_answer_distribution(self, df: pd.DataFrame):
        """Plot distribution of answers for choice questions."""
        print("Plotting answer distribution...")

        # For choice questions (A/B/C/D)
        choice_df = df[df['answer_type'] == 'choice'] if 'answer_type' in df.columns else df

        if len(choice_df) == 0:
            print("  No choice questions found")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Human answers
        human_counts = choice_df['human_answer'].value_counts().sort_index()
        human_counts.plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title('Human Consensus Answers')
        axes[0].set_xlabel('Answer')
        axes[0].set_ylabel('Count')

        # Model answers
        model_counts = choice_df['model_answer_normalized'].value_counts().sort_index()
        model_counts.plot(kind='bar', ax=axes[1], color='coral')
        axes[1].set_title('Model Answers')
        axes[1].set_xlabel('Answer')
        axes[1].set_ylabel('Count')

        plt.tight_layout()
        save_path = self.output_dir / 'answer_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    def save_results(self, df: pd.DataFrame, metrics: Dict):
        """Save analysis results to files."""
        print("\nSaving results...")

        # Save merged data
        csv_path = self.output_dir / 'human_model_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # Save disagreements
        disagreements = df[df['exact_match'] == 0]
        disagree_path = self.output_dir / 'disagreements.csv'
        disagreements.to_csv(disagree_path, index=False)
        print(f"  Saved: {disagree_path}")

        # Save metrics
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved: {metrics_path}")

    def run_full_analysis(
        self,
        human_results_path: str,
        model_results_path: str,
        questions_path: str,
        answer_type: str = None
    ):
        """Run complete analysis pipeline."""
        print("="*60)
        print("HUMAN vs MODEL ANALYSIS PIPELINE")
        print("="*60)

        # Load data
        self.human_data = self.load_human_results(human_results_path)
        self.model_data = self.load_model_results(model_results_path)
        self.questions = self.load_questions(questions_path)

        # Aggregate human responses
        human_agg = self.aggregate_human_responses(self.human_data)

        # Merge
        self.merged_data = self.merge_human_model(human_agg, self.model_data, self.questions)

        # Filter by answer type if specified
        if answer_type:
            print(f"\nFiltering for answer_type: {answer_type}")
            if 'answer_type' in self.merged_data.columns:
                self.merged_data = self.merged_data[self.merged_data['answer_type'] == answer_type]
                print(f"  Filtered to {len(self.merged_data)} questions")
            else:
                print(f"  Warning: No answer_type column found!")

        # Compute agreement
        self.merged_data = self.compute_agreement(self.merged_data)

        # Analyze
        overall_metrics = self.analyze_overall_performance(self.merged_data)
        by_type_metrics = self.analyze_by_answer_type(self.merged_data)
        by_category_metrics = self.analyze_by_category(self.merged_data)
        disagreements = self.analyze_disagreements(self.merged_data)

        # Plot
        self.plot_agreement_by_confidence(self.merged_data)
        self.plot_confusion_matrix(self.merged_data, answer_type='choice')
        self.plot_answer_distribution(self.merged_data)

        # Save
        all_metrics = {
            'overall': overall_metrics,
            'by_answer_type': by_type_metrics.to_dict('records') if by_type_metrics is not None else None,
            'by_category': by_category_metrics.to_dict('records') if by_category_metrics is not None else None,
        }
        self.save_results(self.merged_data, all_metrics)

        print("\n" + "="*60)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)

        return self.merged_data, all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Analyze human vs model performance on blind VQA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--human_results", type=str, required=True,
                        help="Path to human results JSON")
    parser.add_argument("--model_results", type=str, required=True,
                        help="Path to model predictions JSONL")
    parser.add_argument("--questions", type=str, required=True,
                        help="Path to questions CSV")
    parser.add_argument("--output_dir", type=str, default="analysis/results",
                        help="Output directory")
    parser.add_argument("--answer_type", type=str, default=None,
                        choices=['choice', 'text'],
                        help="Filter by answer type")

    args = parser.parse_args()

    analyzer = HumanModelAnalyzer(output_dir=args.output_dir)
    analyzer.run_full_analysis(
        human_results_path=args.human_results,
        model_results_path=args.model_results,
        questions_path=args.questions,
        answer_type=args.answer_type
    )


if __name__ == "__main__":
    main()
