"""
ECDF confidence figure: rows = model groups, columns = individual models.
7B/8B models only. No background coloring. Compact panels.

Outputs
-------
ecdf_confidence/grouped_7b.png   — supplementary figure
  (also copied to LaTeX figures/confidence/)
"""

from pathlib import Path
import json, re, sys, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from utils.constants import VARIANT_COLORS
from utils.model_registry import default_all_models, MODEL_TYPE

OUT_DIR   = Path(__file__).parent / "ecdf_confidence"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_OUT = ROOT / "latex" / "AAAI2026" / "LaTeX" / "figures" / "confidence"
LATEX_OUT.mkdir(parents=True, exist_ok=True)

HUMAN_DIR        = ROOT / "evaluation" / "humans" / "by_participant"
VARIANT_FILE_KEY = {"C": "question", "B": "weaker_object", "A": "pronominalized"}
VARIANT_ORDER    = ["C", "B", "A"]
VARIANT_LABELS   = {"C": "Original", "B": "Weaker-object", "A": "Pronominalized"}

# 7B/8B model selection per row (display order = column order).
# Think model placed immediately after its nothink counterpart.
# Rows: VLM | Backbone Decoder | Standalone LLM (think merged in)
MODELS_7B = {
    "VLM": [
        "Qwen3-VL-8B", "LLaVA-1.5-7B", "LLaVA-Mistral", "LLaVA-Vicuna", "InternVL-8B",
    ],
    "VLM backbone decoder": [
        "Qwen3-VL-8B (LM)", "LLaVA-1.5 (LM)", "LLaVA-Mistral (LM)",
        "LLaVA-Vicuna (LM)", "InternVL-8B (LM)",
    ],
    # Think model placed right after Qwen3-8B
    "standalone LLM": [
        "Qwen3-8B", "Qwen3-8B (think)", "Mistral-7B", "Vicuna-7B", "Qwen2.5-7B-Instruct",
    ],
}

ROW_ORDER = ["VLM", "VLM backbone decoder", "standalone LLM"]
ROW_LABEL = {
    "VLM":                  "VLM",
    "VLM backbone decoder": "Backbone\nDecoder",
    "standalone LLM":       "Standalone\nLLM",
}

# Short single-line display names for column headers
SHORT_LABEL = {
    "Qwen3-VL-8B":          "Qwen3-VL-8B",
    "LLaVA-1.5-7B":         "LLaVA-1.5-7B",
    "LLaVA-Mistral":        "LLaVA-Mistral",
    "LLaVA-Vicuna":         "LLaVA-Vicuna",
    "InternVL-8B":          "InternVL-8B",
    "Qwen3-VL-8B (LM)":     "Q3-VL-8B",
    "LLaVA-1.5 (LM)":       "LLaVA-1.5",
    "LLaVA-Mistral (LM)":   "Mistral",
    "LLaVA-Vicuna (LM)":    "Vicuna",
    "InternVL-8B (LM)":     "InternVL",
    "Qwen3-8B":             "Qwen3-8B",
    "Qwen3-8B (think)":     "Qwen3-8B (T)",
    "Mistral-7B":           "Mistral-7B",
    "Vicuna-7B":            "Vicuna-7B",
    "Qwen2.5-7B-Instruct":  "Qwen2.5-7B",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "normal",
    "axes.labelsize":    9,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
    "axes.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

EOS_TOKENS = {"<|im_end|>", "</s>", "<|endoftext|>", "<eos>"}


def plot_ecdf(ax, values: np.ndarray, color: str, lw: float = 2.0):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=color, lw=lw)


def load_human_data() -> dict:
    records = []
    for fpath in sorted(HUMAN_DIR.glob("*.json")):
        d = json.load(open(fpath))
        for ans in d.get("answers", []):
            records.append({
                "variant":   ans["variant"],
                "conf_norm": (ans["confidence"] - 1) / 4,
            })
    return {
        v: np.array([r["conf_norm"] for r in records if r["variant"] == v])
        for v in VARIANT_ORDER
    }


def answer_logprobs(tokens: list) -> list:
    think_end = -1
    for i, t in enumerate(tokens):
        if "</think>" in t.get("token", ""):
            think_end = i
            break
    answer_tokens = tokens[think_end + 1:] if think_end >= 0 else tokens
    return [
        t["logprob"] for t in answer_tokens
        if t.get("token", "").strip() and t.get("token") not in EOS_TOKENS
    ]


def load_model_data(jsonl_path: Path) -> dict:
    out = {v: [] for v in VARIANT_ORDER}
    if not jsonl_path.exists():
        return {v: np.array([]) for v in VARIANT_ORDER}
    for line in open(jsonl_path):
        rec = json.loads(line)
        gl  = rec.get("generated_logits", {})
        for v in VARIANT_ORDER:
            tokens = gl.get(VARIANT_FILE_KEY[v], {}).get("content", [])
            lps = answer_logprobs(tokens)
            if lps:
                out[v].append(np.mean(lps))
    return {v: np.array(vals) for v, vals in out.items()}


def _load_model_cache() -> dict[str, dict]:
    """Load logprob data for all 7B models (cached for reuse)."""
    all_models = default_all_models(ROOT)
    model_data: dict[str, dict] = {}
    all_7b = [m for g in MODELS_7B.values() for m in g]
    for label in all_7b:
        if label not in all_models:
            continue
        results_dir, model_subdir = all_models[label]
        jsonl = results_dir / model_subdir / "vqa_1k_control_inst_blind.jsonl"
        d = load_model_data(jsonl)
        if any(len(v) > 0 for v in d.values()):
            model_data[label] = d
        else:
            print(f"  Skipping {label} — no inst_blind data")
    return model_data


def make_grouped_rows_7b(model_data: dict):
    """3-row figure (VLM / Backbone / SA-LLM). Think model next to Qwen3-8B. No human."""
    NCOLS = 5
    NROWS = len(ROW_ORDER)   # 3

    panel_w, panel_h = 1.35, 1.20
    left_margin   = 0.80
    right_margin  = 0.05
    top_margin    = 0.15
    bottom_margin = 0.55

    fig_w = left_margin + NCOLS * panel_w + right_margin
    fig_h = top_margin + NROWS * panel_h + bottom_margin

    fig, axes = plt.subplots(
        NROWS, NCOLS, figsize=(fig_w, fig_h), squeeze=False,
        sharex="col", sharey=True,
        gridspec_kw={
            "hspace": 0.40, "wspace": 0.15,
            "left":   left_margin / fig_w,
            "right":  1.0 - right_margin / fig_w,
            "top":    1.0 - top_margin / fig_h,
            "bottom": bottom_margin / fig_h,
        },
    )

    for row_idx, group_key in enumerate(ROW_ORDER):
        models_in_row = [m for m in MODELS_7B[group_key] if m in model_data]

        for col_idx in range(NCOLS):
            ax = axes[row_idx][col_idx]

            if col_idx >= len(models_in_row):
                ax.set_visible(False)
                continue

            label = models_in_row[col_idx]
            data  = model_data[label]

            for v in VARIANT_ORDER:
                vals = data[v]
                if len(vals):
                    plot_ecdf(ax, vals, color=VARIANT_COLORS[v])

            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])

            # x-label only on bottom row (sharex handles tick labels automatically)
            if row_idx == NROWS - 1:
                ax.set_xlabel("Mean logprob", fontsize=9)

            if col_idx == 0:
                ax.set_ylabel("")

            ax.set_title(SHORT_LABEL.get(label, label), fontsize=10, pad=3,
                         fontweight="normal")

        # Row label (rotated, left of first panel)
        axes[row_idx][0].annotate(
            ROW_LABEL[group_key],
            xy=(-0.56, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="normal", ha="center", va="center",
            rotation=90, annotation_clip=False,
        )

    handles = [mpatches.Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
               for v in VARIANT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.035))

    out = OUT_DIR / "grouped_7b.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_OUT / out.name)
    print(f"Saved:  {out}")
    print(f"Copied: {LATEX_OUT / out.name}")
    plt.close(fig)


def make_human_ecdf():
    """Single-panel human confidence ECDF for separate supplementary section."""
    human_data = load_human_data()

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    for v in VARIANT_ORDER:
        vals = human_data[v]
        if len(vals):
            plot_ecdf(ax, vals, color=VARIANT_COLORS[v], lw=2.1)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (normalised, 0–1)", fontsize=9)
    ax.set_ylabel("ECDF", fontsize=9)

    handles = [mpatches.Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
               for v in VARIANT_ORDER]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()

    out = OUT_DIR / "human_ecdf.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy(out, LATEX_OUT / out.name)
    print(f"Saved:  {out}")
    print(f"Copied: {LATEX_OUT / out.name}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading 7B model data…")
    model_data = _load_model_cache()

    print("Building grouped 7B ECDF figure…")
    make_grouped_rows_7b(model_data)

    print("Building human ECDF figure…")
    make_human_ecdf()

    print("Done.")
