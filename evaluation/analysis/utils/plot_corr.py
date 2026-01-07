import matplotlib.pyplot as plt
from scipy import stats
import numpy as np 
from scipy.stats import t
from matplotlib.lines import Line2D

family_markers = {
    'InternVL': 'o', 
    'LLaVA': 's', 
    'Qwen': '^',
}
family_colors = {
    'InternVL': '#4C72B0',
    'LLaVA': '#DD8452',
    'Qwen': '#55A868'}  

regime_colors  = {
    "Ground Truth":  "#4C72B0",  # darker muted blue
    # "SFT-VQA":      "#8DA0CB",  # lighter blue
    "Blind": "#D9A066",  
    "Blind (n=10)": "#D9A066",  
    "Blind (n=15)": "#C07A2D", 
    "Pretrained": "#B0B0B0"
}

regime_markers = {
    'Pretrained': 'o', 
    'JS': '^', 
    'SFT': 's',
}

def legend_handler(ax): 

    regime_handles = [
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markerfacecolor=regime_colors[reg],
            markeredgecolor='white',
            markeredgewidth=0.6,
            markersize=9,
            label=reg,
        )
        for reg in regime_colors # ['JS VQA (GT)', 'SFT-VQA', 'JS-Blind VQA (n=15)', 'SFT-Blind VQA']
    ]

    leg1 = ax.legend(
        handles=regime_handles,
        # title="Training regime",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        fontsize=11,
        frameon=False,
    )

    ax.add_artist(leg1)  # ✅ CRITICAL

    marker_handles = [
        Line2D(
            [0], [0],
            marker=regime_markers[fam],
            linestyle='None',
            color='black',
            markersize=9,
            label=fam,
        )
        for fam in regime_markers
    ]

    ax.legend(
        handles=marker_handles,
        # title="Model family",
        loc="lower center",
        bbox_to_anchor=(1, 1),
        ncol=4,
        fontsize=11,
        frameon=False,
    )
    return ax
 
def plot_families_corr(model_corr, human, llm): 
    nrows, ncols = 1, 3
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(3.2 * ncols, 2 * nrows),
        constrained_layout=True,
        sharey=True,  # Share y axis
        # sharex=True, # if xshared else False,
        squeeze=False
    )
    ylabel, xlabel = "Accuracy", r"${\rho}$" 
    for ax in axes.flatten():
        ax.set_xlim(0.33, 0.65)
    axes_flat = axes.flatten() 

    for idx, fam in enumerate(model_corr.family.unique()) : 
        ax = axes_flat[idx] 
        ax.set_title(fam, fontsize=10)
        if idx == 0 : 
            ax.set_ylabel(ylabel, fontsize=10) 

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(10)
        if idx == 0 : 
            ax.set_ylabel(ylabel) 
        df = model_corr[model_corr['family'] == fam]
        
        for _, row in df.iterrows():
            color = regime_colors[row["blind"]]
            ax.errorbar(
                row["mean_r"],
                row["correct"],
                # xerr=row["std_corr"],
                fmt=regime_markers.get(row["strategy"], "o"),
                markerfacecolor=color,
                markeredgewidth=0.0,  # Remove edge on marker
                markeredgecolor=color,
                markersize=np.sqrt(row['model_size'] * 15),  # scale marker size
                capsize=2,
                zorder=3,
                ecolor=color,
                elinewidth=0,  # Slightly thinner error line for softness
                alpha=0.8,       # Make the error bars softer and more transparent
                errorevery=1,
            )

        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.axvspan(llm['ci_2p5'], llm['ci_97p5'] , alpha=0.05, color='gray', zorder=1)
        ax.axvline(llm['mean_r'], color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1, label='Qwen') 
    
        ax.axvspan(human['ci_2p5'], human['ci_97p5'] , alpha=0.05, color='red', zorder=1)
        ax.axvline(human['mean_r'], color='red', linestyle='--', linewidth=1, alpha=0.7, zorder=1, label='Humans') 

    handles = []
    labels = []

    # Collect unique regimes from both the 'strategy' and 'blind' columns for the current fam
    # Separate the regimes by row in the legend
    # Get unique regimes for each row label
    regimes_strategy = list(df["strategy"].unique())
    regimes_blind = list(df["blind"].unique()) if "blind" in df.columns else []

    # Prepare separate handles/labels for the two groups
    handles_strategy = []
    labels_strategy = []
    for regime in regimes_strategy:
        marker = regime_markers.get(regime, "o")
        color = regime_colors.get(regime, "gray")
        handle = plt.Line2D([0], [0], marker=marker, color='w',
                            markerfacecolor=color, markeredgecolor=color,
                            markersize=8, linestyle='', label=regime)
        handles_strategy.append(handle)
        labels_strategy.append(regime)

    handles_blind = []
    labels_blind = []
    for regime in regimes_blind:
        # Avoid duplicate if regime is already in strategy
        if regime in regimes_strategy:
            continue
        marker = regime_markers.get(regime, "o")
        color = regime_colors.get(regime, "gray")
        handle = plt.Line2D([0], [0], marker=marker, color='w',
                            markerfacecolor=color, markeredgecolor=color,
                            markersize=8, linestyle='', label=regime)
        handles_blind.append(handle)
        labels_blind.append(regime)

    # Create 2-row legend (each row = group of regimes)
    handles = handles_strategy + handles_blind
    labels = labels_strategy + labels_blind

    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=5,
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
        fontsize=10,
        columnspacing=1,
        handletextpad=0.7,
        labelspacing=1.1,
        borderaxespad=0.2,
        title=None,
        fancybox=False
    )

    plt.show() 