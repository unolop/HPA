from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))

from helpers import sample_composition_tables  # noqa: E402

OUT_DIR = ROOT / "figures" / "sample_composition"
LATEX_DIR = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "sample_composition"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 9,
    "font.size": 8,
})


def autopct_counts(values):
    total = sum(values)

    def _fmt(pct):
        count = int(round(pct * total / 100.0))
        return f"{count}\n({pct:.0f}%)" if pct >= 6 else ""

    return _fmt


def draw():
    overall, by_op, by_ent = sample_composition_tables()

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

    # Overall answer format
    vals = [int(overall.iloc[0]["Yes/No"]), int(overall.iloc[0]["Count"]), int(overall.iloc[0]["Free-text"])]
    labels = ["Yes/No", "Count", "Free-text"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    axes[0].pie(
        vals,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=autopct_counts(vals),
        pctdistance=0.70,
        labeldistance=1.08,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        textprops={"fontsize": 7.5},
    )
    axes[0].set_title("Answer format")

    # Operation
    op_vals = by_op["Total"].astype(int).tolist()
    op_labels = by_op["Group"].tolist()
    axes[1].pie(
        op_vals,
        labels=op_labels,
        startangle=90,
        counterclock=False,
        autopct=autopct_counts(op_vals),
        pctdistance=0.72,
        labeldistance=1.10,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        textprops={"fontsize": 6.5},
    )
    axes[1].set_title("Operation type")

    # Entity
    ent_vals = by_ent["Total"].astype(int).tolist()
    ent_labels = by_ent["Group"].tolist()
    axes[2].pie(
        ent_vals,
        labels=ent_labels,
        startangle=90,
        counterclock=False,
        autopct=autopct_counts(ent_vals),
        pctdistance=0.72,
        labeldistance=1.10,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        textprops={"fontsize": 6.5},
    )
    axes[2].set_title("Entity type")

    fig.tight_layout(w_pad=1.0)
    out = OUT_DIR / "sample_q113_composition_pies.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    shutil.copy2(out, LATEX_DIR / out.name)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    draw()
