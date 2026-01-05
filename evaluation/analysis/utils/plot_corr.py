import matplotlib.pyplot as plt
from scipy import stats
import numpy as np 
from scipy.stats import t
from matplotlib.lines import Line2D

JS_REGIMES  = ["JS VQA (GT)", "JS-Blind VQA (n=15)"]
SFT_REGIMES = ["SFT-Blind VQA", "SFT-VQA"] 
REGIME_GROUPS = {
    "JS":  JS_REGIMES,
    "SFT": SFT_REGIMES,
}
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
    # Ground‑truth supervised (blue family)
    "JS VQA (GT)":  "#4C72B0",  # darker muted blue
    "SFT-VQA":      "#8DA0CB",  # lighter blue

    # Blind training (gray family)
    "JS-Blind VQA (n=15)": "#D9A066",  
    "JS-Blind VQA (n=10)": "#D9A066",  
    "SFT-Blind VQA":       "#B65F2E",  

    'SFT-Blind MMStar' : "#D9A066", 
    'JS-Blind MMStar': "#D9A066",   

    "Pretrained": "#B0B0B0"
}

regime_markers = {
    'Pretrained': 'o', 
    'JS VQA (GT)': '^', 
    'SFT-VQA': 's',
    'JS-Blind VQA (n=10)': '^',  
    'JS-Blind VQA (n=15)': '^',  
    'SFT-Blind VQA': 's', 

    'SFT-Blind MMStar' : "s", 
    'JS-Blind MMStar': "^",   

}

def plot_families_corr(llm_mean, llm_sd, mean_corr, std_corr): 
    nrows, ncols = 1, 3
    max_panels = nrows * ncols

    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(3.2 * ncols, 2.3 * nrows),
        constrained_layout=True,
        sharey=True,  # Share y axis
        # sharex=True, # if xshared else False,
        squeeze=False
    )
    ylabel, xlabel = "Accuracy", r"${\rho}$" 
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
        ax = plot_family_scatter(
            pcorr[pcorr['family'] == fam], ax, value_col='correct', mu=mean_corr, sd=std_corr 
        ) 
        ci_low, ci_high = get_ci(llm_mean, llm_sd, len(test_subjects)) 
        ax.axvspan(ci_low, ci_high, alpha=0.05, color='gray', zorder=1)
        ax.axvline(qwen, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1, label='Qwen') 
    plt.show() 
    
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

    family_handles = [
        Line2D(
            [0], [0],
            marker=family_markers[fam],
            linestyle='None',
            color='black',
            markersize=9,
            label=fam,
        )
        for fam in family_markers
    ]

    ax.legend(
        handles=family_handles,
        # title="Model family",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
        fontsize=11,
        frameon=False,
    )
    return ax
 
def get_ci(mean_corr, std_corr, n): 
    alpha = 0.05
    tcrit = t.ppf(1 - alpha/2, df=n-1)

    ci_low  = mean_corr - tcrit * std_corr / np.sqrt(n)
    ci_high = mean_corr + tcrit * std_corr / np.sqrt(n) 

    return(ci_low, ci_high)

def plot_family_scatter(
    df, ax,
    value_col,
    mu=None,
    sd=None,
):

    if mu is not None and sd is not None:
        ci_low, ci_high = get_ci(mu, sd, 6) 
        ax.axvspan(ci_low, ci_high, alpha=0.01, color='red', zorder=1)
        ax.axvline(mu, color='red', linestyle='--', linewidth=1, alpha=0.5, 
                    zorder=2, label='Human mean')
    
    for _, row in df.iterrows():
        color = regime_colors[row["finetuned"]]
        ax.errorbar(
            row["mean_corr"],
            row[value_col],
            xerr=row["std_corr"],
            fmt=regime_markers.get(row["finetuned"], "o"),
            markerfacecolor=color,
            markeredgewidth=0.0,  # Remove edge on marker
            markeredgecolor=color,
            markersize=np.sqrt(row['model_size'] * 10),  # scale marker size
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

    return ax  
  

def get_stats(delta_df): 
    regression_stats = {}
    for regime, labels in REGIME_GROUPS.items():
        data = delta_df[delta_df["finetuned"].isin(labels)]
        if len(data) < 3:
            continue

        x = data["delta_theta"].to_numpy()
        y = data["delta_mg"].to_numpy()

        slope, intercept, r_val, _, _ = stats.linregress(x, y)
        rho, _ = stats.spearmanr(x, y)

        regression_stats[regime] = {
            "slope": slope,
            "intercept": intercept,
            "r": r_val,
            "rho": rho,
        } 
    return regression_stats 
