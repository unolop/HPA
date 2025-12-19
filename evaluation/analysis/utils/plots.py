import seaborn as sns
from scipy.stats import linregress, spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp 
from adjustText import adjust_text
import re

def parse_size(name):
    # Extracts numbers followed by 'B' or 'b' (e.g., InternVL3_5-1B -> 1.0)
    match = re.search(r'(\d+(?:\.\d+)?)B', name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 7.0  # Default size if not found


def get_family(model_name):
    name = model_name.lower()
    if 'llava' in name: return 'LLaVA'
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

def scatterplot(subset, x_col='pearson_r', y_col='model_avg', title='accuracy'): 
    x = subset[x_col].values 
    y = subset[y_col].values 

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    spearman_r, _ = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # 1. Regression line + CI (68% for cleaner look)
    sns.regplot(
        data=subset, x=x_col, y=y_col,
        scatter=False, ci=68, ax=ax,
        line_kws=dict(color="gray", linestyle="--", linewidth=1.2, alpha=0.4),
    )

    # 2. Scatter points
    for fam in subset["family"].unique():
        fam_data = subset[subset["family"] == fam]
        ax.scatter(
            fam_data[x_col], fam_data[y_col],
            s=fam_data["model_size"] * 65,
            alpha=0.8, label=fam, edgecolors="none", zorder=3
        )

    # 3. Labeling
    texts = []
    for _, row in subset.iterrows():
        text = ax.text(
            row[x_col], row[y_col] + ((y.max() - y.min()) * 0.04),
            row["model"],
            fontsize=10, color="#333333"
        )
        texts.append(text)

    adjust_text(
        texts,
        only_move={'points': 'y', 'text': 'xy'},
        arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.5),
        expand_text=(1.5, 2.8),
        expand_points=(1.5, 2.8),
        force_text=0.15,
        force_points=0.15,
        lim=300,
    )

    # 4. DISTINCT AXIS FORMATTING
    # Show the left and bottom spines (the "L" frame)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.2)
    
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make ticks visible and pointing outward
    ax.tick_params(left=True, bottom=True, direction='out', color='black', width=1.2)

    # Tighten limits based on data
    x_margin = (x.max() - x.min()) * 0.1
    y_margin = (y.max() - y.min()) * 0.15
    ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
    ax.set_ylim(y.min() - y_margin, y.max() + y_margin)
    
    ax.set_xlabel("Human–Model Alignment (Spearman $rho$)", labelpad=10)
    ax.set_ylabel(y_col, labelpad=10) #  "Mean Model Score" 
    
    ax.set_title(
        f"{title}\n"
        f"Pearson $r$ = {r_value:.3f} ($R^2$ = {r_value**2:.3f}) | Spearman $\\rho$ = {spearman_r:.3f}",
        pad=20,
    )

    print(f"Pearson $r$ = {r_value:.3f} ($R^2$ = {r_value**2:.3f}) | Spearman $\\rho$ = {spearman_r:.3f}") 
    plt.tight_layout()
    plt.savefig(f"./figures/scatter-{title}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
def get_correlation_by_model( 
    mmg,
    model, 
    score_col='answer_similarity',
    grouping=['question_type'], 
):

    model_type = mmg.meta_model.unique()[0] 
    df_model = mmg[mmg['model'] == model] 

    counts = df_model.groupby(grouping).size()  
    count_stats = {
        "num_question_types": int(len(counts)),
        "min_q_per_type": int(counts.min()),
        "median_q_per_type": float(counts.median()),
        "max_q_per_type": int(counts.max()),
    }

    grouped = (
        df_model
        .groupby(grouping)
        .agg({
            f'{score_col}_human': 'mean',
            f'{score_col}_model': 'mean'
        })
        .reset_index()
    )
    x = grouped[f'{score_col}_human']
    y = grouped[f'{score_col}_model']
    
    plot_scatter(grouped, model, score_col) 
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    score_human = float(x.mean())
    score_model = float(y.mean())

    return {
        "model": model.split('/')[-1],
        "model_type": model_type,
        "eval_metric": score_col,
        "pearson_r": r_value,
        "pearson_p": p_value,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "slope": slope,
        "intercept": intercept,
        "std_err": std_err,
        "model_avg": score_model,
        "human_avg": score_human,
        **count_stats,
    }, df_model

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

def draw_histograms_vqa(x, y, x2, y2, save=None): 
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    # --- Accuracy plot ---
    sns.ecdfplot(x, label="Human", ax=axes[0])
    sns.ecdfplot(y, label="Model", ax=axes[0])
    axes[0].axvline(x.mean(), color='blue', ls='--', lw=1,
                    label=f'Human μ={x.mean():.3f}')
    axes[0].axvline(y.mean(), color='orange', ls='--', lw=1,
                    label=f'Model μ={y.mean():.3f}')
    axes[0].set_xlabel("Per-question accuracy")
    axes[0].set_ylabel("ECDF")
    axes[0].set_title("Accuracy")
    axes[0].legend(frameon=False)

    # --- Embedding plot ---
    sns.ecdfplot(x2, label="Human", ax=axes[1])
    sns.ecdfplot(y2, label="Model", ax=axes[1])
    axes[1].axvline(x2.mean(), color='blue', ls='--', lw=1,
                    label=f'Human μ={x2.mean():.3f}')
    axes[1].axvline(y2.mean(), color='orange', ls='--', lw=1,
                    label=f'Model μ={y2.mean():.3f}')
    axes[1].set_xlabel("Embedding similarity (cosine)")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title("Embedding")
    axes[1].legend(frameon=False)

    sns.despine()
    plt.tight_layout()

    if save is None:
        plt.show()
    else:
        plt.savefig(f"./figures/align/{save}.png", bbox_inches='tight')
    plt.close()


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
