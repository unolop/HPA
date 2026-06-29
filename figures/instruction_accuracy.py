from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/david/Desktop/yuna/HPA")
OUT = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "inst_accuracy_shift_models.png"


ROWS = [
    ("InternVL3.5-1B", 80.5, 43.6, 43.8),
    ("InternVL3.5-2B", 82.1, 45.5, 45.3),
    ("InternVL3.5-4B", 83.0, 45.0, 47.2),
    ("InternVL3.5-8B", 86.9, 47.1, 45.4),
    ("Qwen3-VL-2B", 83.0, 45.0, 45.7),
    ("Qwen3-VL-4B", 85.9, 45.5, 47.4),
    ("Qwen3-VL-8B", 89.4, 44.3, 47.4),
]


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.1, 4.9), dpi=180)

    models = [r[0] for r in ROWS]
    gt = np.array([r[1] for r in ROWS])
    blind = np.array([r[2] for r in ROWS])
    inst = np.array([r[3] for r in ROWS])
    y = np.arange(len(models))

    for yi, b, i in zip(y, blind, inst):
        ax.plot([b, i], [yi, yi], color="#9aa0a6", lw=1.8, alpha=0.9, zorder=1)

    ax.scatter(gt, y, marker="x", s=46, lw=1.6, color="#8c8c8c", label="GT", zorder=3)
    ax.scatter(blind, y, marker="o", s=42, color="#4c78a8", edgecolor="white", linewidth=0.6,
               label="Blind", zorder=4)
    ax.scatter(inst, y, marker="s", s=42, color="#f58518", edgecolor="white", linewidth=0.6,
               label="+ instruction", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(40, 92)
    ax.set_xlabel("VQA accuracy (%)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.8)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.axhline(3.5, color="#d0d0d0", lw=0.8)
    ax.text(91.5, 1.5, "InternVL3.5", ha="right", va="center", fontsize=9, color="#666666")
    ax.text(91.5, 5.0, "Qwen3-VL", ha="right", va="center", fontsize=9, color="#666666")

    leg = ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    leg.get_frame().set_alpha(0.95)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
