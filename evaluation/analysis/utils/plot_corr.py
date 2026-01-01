import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd 
from matplotlib.lines import Line2D 

JS_REGIMES  = ["JS VQA (GT)", "JS-Blind VQA (n=15)"]
SFT_REGIMES = ["SFT-Blind VQA", "SFT-VQA"] 
REGIME_GROUPS = {
    "JS":  JS_REGIMES,
    "SFT": SFT_REGIMES,
}
regime_colors  = {
    # Ground‑truth supervised (blue family)
    "JS VQA (GT)":  "#4C72B0",  # darker muted blue
    "SFT-VQA":      "#8DA0CB",  # lighter blue

    # Blind training (gray family)
    "JS-Blind VQA (n=15)": "#7F7F7F",  # darker gray
    "SFT-Blind VQA":       "#B0B0B0",  # lighter gray
}
REGIME_LINESTYLE = {'JS': {'color': '#4C72B0', 'linestyle': '--'},
                    'SFT': {'color': "gray", 'linestyle': '--'}}  # '#DD8452' 
family_markers = {
    'InternVL': 'o', 
    'LLaVA': 's', 
    'Qwen': '^',
    'BLIP': '^',  
}
family_colors = {
    "InternVL": "#4C72B0",        # muted blue
    "LLaVA":    "#DD8452",        # muted orange
    "Qwen":     "#55A868",        # muted green
    "BLIP":     "#C44E52",        # muted red
}

regime_markers = {
    'Pretrained': 'o', 
    'JS VQA (GT)': 's', 
    'SFT-VQA': 's',
    'JS-Blind VQA (n=15)': '^',  
    'SFT-Blind VQA': '^', 
}


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

def get_delta_df(df): 
    delta_rows = []

    for fam in df["family"].unique():
        fam_data = df[df["family"] == fam]

        for _, fin in fam_data[fam_data["finetuned"] != "Pretrained"].iterrows():
            pre = fam_data[
                (fam_data["model"] == fin["model"]) &
                (fam_data["finetuned"] == "Pretrained")
            ]
            if pre.empty:
                continue

            pre = pre.iloc[0]

            delta_rows.append({
                "family": fam,
                "model": fin["model"],
                "finetuned": fin["finetuned"],
                "delta_theta": fin["theta_hat"] - pre["theta_hat"],
                "delta_mg": fin["MG_acc"] - pre["MG_acc"],
                "model_size": fin["model_size"],
            })

    delta_df = pd.DataFrame(delta_rows)
    size_vals = delta_df["model_size"].astype(float)
    size_norm = (size_vals - size_vals.min()) / (size_vals.max() - size_vals.min() + 1e-6)
    delta_df["_size_norm"] = 0.5 + size_norm
    return delta_df 

def scatterplot_delta_alignment_vs_mg(
    delta_df,
    title,
    colors,          # keyed by finetuned regime
    markers,         # keyed by family (or regime if you prefer)
    regression_stats,
    EQ_POS, 
    size_scale=500,
    figsize=(12, 7),
):
    fig, ax = plt.subplots(figsize=figsize)
    # --------------------------------------------------
    # Regression per regime group
    # --------------------------------------------------
    for regime, labels in REGIME_GROUPS.items():
        data = delta_df[delta_df["finetuned"].isin(labels)]
        if len(data) < 3:
            continue

        sns.regplot(
            data=data,
            x="delta_theta",
            y="delta_mg",
            scatter=False,
            ci=68,
            # n_boot=200,          # ✅ bootstrap
            truncate=True,        # ✅ KEY FIX
            ax=ax,
            line_kws=dict(
                linewidth=2.0,
                alpha=0.9,
                zorder=2,
                **REGIME_LINESTYLE[regime],
            ),
        )
    for regime, stats_ in regression_stats.items():
        slope = stats_["slope"]
        intercept = stats_["intercept"]
        r_val = stats_["r"]

        eq_text = (
            rf"$\Delta$MG $= {slope:.2f}\,\Delta\hat{{\theta}} {intercept:+.2f}$" "\n"
            rf"$r = {r_val:.2f}$"
        )

        ax.text(
            EQ_POS[regime][0],
            EQ_POS[regime][1],
            eq_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            color=REGIME_LINESTYLE[regime]["color"],
            # alpha=0.9,
        )
    # --------------------------------------------------
    # Scatter points
    # --------------------------------------------------
    used_regime_labels = set()

    for _, row in delta_df.iterrows():
        reg = row["finetuned"]
        fam = row["family"]

        show_regime = reg not in used_regime_labels

        ax.scatter(
            row["delta_theta"],
            row["delta_mg"],
            s=row["_size_norm"] * size_scale,
            color=colors[reg],
            marker=markers[fam],
            alpha=0.75,
            edgecolors="white",
            linewidth=0.4,
            zorder=3,
            label=reg if show_regime else None,
        )

        if show_regime:
            used_regime_labels.add(reg)
    ax.axhline(0, color="black", lw=1.1, alpha=0.7)
    ax.axvline(0, color="black", lw=1.1, alpha=0.7)

    # --------------------------------------------------
    # Labels, title, legend
    # --------------------------------------------------
    ax.set_xlabel(r"$\Delta\ \hat{\theta}$ (alignment change)", fontsize=11)
    ax.set_ylabel(r"$\Delta$ MG accuracy", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(
        True,
        which="major",
        linewidth=0.5,
        alpha=0.2,
    )
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    desired_order = ['JS VQA (GT)', 'SFT-VQA', 'JS-Blind VQA (n=15)', 'SFT-Blind VQA']
    order = [labels.index(reg) for reg in desired_order if reg in labels]
    ordered_handles = [handles[idx] for idx in order]
    ordered_labels = [labels[idx] for idx in order]

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
        for reg in ['JS VQA (GT)', 'SFT-VQA',
                    'JS-Blind VQA (n=15)', 'SFT-Blind VQA']
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
    plt.subplots_adjust(bottom=0.35)
    plt.tight_layout()
    plt.show()