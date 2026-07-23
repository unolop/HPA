"""
SBERT truncation scatter: x = raw HM SBERT per model, y = 5-word-truncated HM SBERT.
Color = model group, marker shape = model family, size = parameter count.
Outputs two files:
  sbert_truncation_scatter_vc_filtered.png   — abstention-filtered (primary)
  sbert_truncation_scatter_vc_with_abst.png  — abstentions included
"""
import re
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

from figures.helpers import filter_abstained_pairs
from analysis.utils.constants import GROUP_COLORS

EXPORTS = ROOT / "analysis/session2/exports"
OUT = ROOT / "latex/AAAI2026/LaTeX/figures/appQ_sbert_robustness"
OUT.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ["VLM", "VLM backbone decoder", "standalone LLM", "standalone LLM (think)"]
GROUP_LABELS = {
    "VLM": "VLM",
    "VLM backbone decoder": "Backbone",
    "standalone LLM": "SA-LLM",
    "standalone LLM (think)": "Think",
}

FAMILY_MARKERS = {
    "InternVL":       "o",
    "Qwen":           "s",
    "LLaVA-1.5":      "^",
    "LLaVA-Mistral":  "D",
    "LLaVA-Vicuna":   "v",
    "Phi":            "P",
}

VARIANT = "C"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 300, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def model_family(name: str) -> str:
    n = name.replace(" (LM)", "").replace(" (think)", "")
    if "InternVL" in n:       return "InternVL"
    if "Qwen" in n:           return "Qwen"
    if "LLaVA-Mistral" in n:  return "LLaVA-Mistral"
    if "LLaVA-Vicuna" in n:   return "LLaVA-Vicuna"
    if "LLaVA-1.5" in n or "LLaVA 1.5" in n: return "LLaVA-1.5"
    if "Vicuna" in n:         return "LLaVA-Vicuna"
    if "Mistral" in n:        return "LLaVA-Mistral"
    if "Phi" in n:            return "Phi"
    return "Qwen"


def model_size_b(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", name)
    return float(m.group(1)) if m else 7.0


def marker_size(b: float) -> float:
    return 18 + b * 1.8


def per_model_scores(base_df, w5_df, filter_abst: bool) -> pd.DataFrame:
    hm    = base_df[(base_df["pair_type"] == "HM") & (base_df["variant"] == VARIANT)]
    hm_w5 = w5_df[(w5_df["pair_type"] == "HM") & (w5_df["variant"] == VARIANT)]
    if filter_abst:
        hm    = filter_abstained_pairs(hm)
        hm_w5 = filter_abstained_pairs(hm_w5)
    raw_s   = hm.groupby(["subject_group_2", "subject_2"])["sbert_score"].mean()
    trunc_s = hm_w5.groupby(["subject_group_2", "subject_2"])["sbert_score"].mean()
    df = pd.DataFrame({"raw": raw_s, "trunc": trunc_s}).reset_index()
    df.columns = ["group", "model", "raw", "trunc"]
    return df[df["group"].isin(GROUP_ORDER)].copy()


def make_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.8))

    for grp in GROUP_ORDER:
        sub = df[df["group"] == grp].copy()
        for _, row in sub.iterrows():
            fam = model_family(row["model"])
            ax.scatter(
                row["raw"], row["trunc"],
                color=GROUP_COLORS[grp],
                marker=FAMILY_MARKERS.get(fam, "o"),
                s=marker_size(model_size_b(row["model"])),
                zorder=3, alpha=0.85,
                edgecolors="white", linewidths=0.4,
            )

    lims = [
        min(df["raw"].min(), df["trunc"].min()) - 0.01,
        max(df["raw"].max(), df["trunc"].max()) + 0.01,
    ]
    ax.plot(lims, lims, color="#999999", lw=0.9, ls="--", zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("HM SBERT (raw)")
    ax.set_ylabel("HM SBERT (5-word truncated)")

    # Legend: group colors
    group_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS[g],
               markersize=7, label=GROUP_LABELS[g])
        for g in GROUP_ORDER
    ]
    # Legend: family markers
    family_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="#555555",
               markersize=7, label=fam)
        for fam, m in FAMILY_MARKERS.items()
    ]

    leg1 = ax.legend(handles=group_handles, title=None, frameon=True,
                     fontsize=6, loc="upper left",
                     framealpha=0.9, edgecolor="#ccc")
    ax.add_artist(leg1)
    ax.legend(handles=family_handles, title=None, frameon=True,
              fontsize=6, loc="lower right",
              framealpha=0.9, edgecolor="#ccc")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


print("Loading caches…")
cleaned    = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
cleaned_w5 = pd.read_parquet(EXPORTS / "pair_cache_cleaned_w5.parquet")
raw        = pd.read_parquet(EXPORTS / "pair_cache_raw.parquet")

print("\nComputing filtered (abstentions removed)…")
df_filtered = per_model_scores(cleaned, cleaned_w5, filter_abst=True)
make_scatter(df_filtered, OUT / "sbert_truncation_scatter_vc_filtered.png")

print("\nComputing with abstentions included…")
df_with_abst = per_model_scores(raw, cleaned_w5, filter_abst=False)
make_scatter(df_with_abst, OUT / "sbert_truncation_scatter_vc_with_abst.png")
