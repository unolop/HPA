import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp
import numpy as np 
import matplotlib.pyplot as plt
import re

def parse_size(name):
    # Extracts numbers followed by 'B' or 'b' (e.g., InternVL3_5-1B -> 1.0)
    match = re.search(r'(\d+(?:\.\d+)?)B', name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 7.0  # Default size if not found


def get_family(model_name):
    name = model_name.lower()
    if 'llama' in name: return 'Llama'
    if 'internvl' in name: return 'InternVL'
    if 'gpt' in name: return 'OpenAI'
    if 'claude' in name: return 'Anthropic'
    if 'qwen' in name: return 'Qwen'
    return 'Other'


def plot_vqa_distributions_with_metrics(H, M, model_name): 

    plt.figure(figsize=(6,4))
    sns.histplot(H, bins=20, kde=True, stat="density",
                 alpha=0.5, label="Human avg")
    sns.histplot(M, bins=20, kde=True, stat="density",
                 alpha=0.5, label="Model")

    plt.xlim(0,1)
    plt.xlabel("VQA score")
    plt.title(
        f"{model_name}\n"
        f"Wasserstein={wd:.3f}, KS={ks:.3f} (p={ks_p:.1e})"
    )
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./figures/distribution_{model_name}", dpi=300)  

    wd = wasserstein_distance(H, M)
    ks, ks_p = ks_2samp(H, M)

    return {
        "wasserstein": wd,
        "ks_stat": ks,
        "ks_p": ks_p,
        "human_mean": H.mean(),
        "model_mean": M.mean(),
    }



def plot_scatter_correlation(df, dataset_name, output_path,
                             x_col='human_accuracy',
                             x_label='Human Accuracy',
                             title_suffix=''):
    """
    Create scatter plot showing correlation between human and model performance.

    Args:
        df: DataFrame with aggregated data
        dataset_name: Name of dataset for title
        output_path: Path to save figure
        x_col: Column name for x-axis
        x_label: Label for x-axis
        title_suffix: Additional text for title
    """
    if len(df) == 0:
        print(f"  ⚠️  No data for {dataset_name}")
        return

    # Calculate correlation
    x = df[x_col]
    y = df['model_accuracy']

    # Filter out NaN values
    valid = ~(x.isna() | y.isna())
    x_clean = x[valid]
    y_clean = y[valid]

    if len(x_clean) < 3:
        print(f"  ⚠️  Insufficient data for correlation ({len(x_clean)} points)")
        return

    r, p = stats.pearsonr(x_clean, y_clean)
    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Scatter plot
    ax.scatter(x, y, alpha=0.5, s=40,
              color=COLORS.get(dataset_name, '#666666'),
              edgecolors='white', linewidths=0.5)

    # Add regression line
    z = np.polyfit(x_clean, y_clean, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p_fit(x_line), '--', color='#2C3E50', alpha=0.8, linewidth=1.5)

    # Add diagonal reference line (perfect correlation)
    ax.plot([0, 1], [0, 1], ':', color='gray', alpha=0.5, linewidth=1, label='y=x')

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel('Model Accuracy')
    title = f'{dataset_name.upper()}: Human vs Model Performance'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title)

    # Add correlation statistics
    stats_text = f'Pearson r = {r:.3f} (p = {p:.4f})\nSpearman ρ = {spearman_r:.3f} (p = {spearman_p:.4f})\nn = {len(x_clean)} questions'
    ax.text(0.05, 0.95, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
           fontsize=8)

    # Set limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

    # Grid
    ax.grid(True, alpha=0.2, linestyle='--')

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    print(f"    Pearson r={r:.3f} (p={p:.4f}), n={len(x_clean)}")


def plot_distribution_histogram(stats_by_dataset, output_path):
    """
    Create histogram comparing human and model accuracy distributions.

    Args:
        stats_by_dataset: Dict of {dataset: DataFrame with aggregated data}
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, len(stats_by_dataset), figsize=(15, 4))

    if len(stats_by_dataset) == 1:
        axes = [axes]

    for idx, (dataset, df) in enumerate(stats_by_dataset.items()):
        ax = axes[idx]

        # Plot histograms
        ax.hist(df['human_accuracy'], bins=20, alpha=0.6, label='Human',
               color='#3498DB', edgecolor='white', linewidth=0.5)
        ax.hist(df['model_accuracy'], bins=20, alpha=0.6, label='Model',
               color='#E74C3C', edgecolor='white', linewidth=0.5)

        # Add mean lines
        human_mean = df['human_accuracy'].mean()
        model_mean = df['model_accuracy'].mean()
        ax.axvline(human_mean, color='#3498DB', linestyle='--', linewidth=2,
                  label=f'Human μ={human_mean:.3f}')
        ax.axvline(model_mean, color='#E74C3C', linestyle='--', linewidth=2,
                  label=f'Model μ={model_mean:.3f}')

        # Labels
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{dataset.upper()}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")