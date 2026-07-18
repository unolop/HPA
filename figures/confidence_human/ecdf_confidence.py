"""
Per-entity ECDF of confidence by variant (C, B, A).

For models : confidence = mean logprob per token (vqa_1k_control_inst_blind.jsonl)
             variant mapping: C=question, B=weaker_object, A=pronominalized
For humans : confidence = self-reported Likert (1-5), normalised to [0,1]

Outputs
-------
ecdf_confidence/human.png          — human ECDF
ecdf_confidence/{model_name}.png   — per-model ECDF
ecdf_confidence/_all.png           — unified grid (all models + human)
"""

from pathlib import Path
import json, re, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from utils.constants import VARIANT_COLORS
from utils.model_registry import default_all_models, MODEL_TYPE

OUT_DIR = Path(__file__).parent / "ecdf_confidence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HUMAN_DIR = ROOT / "evaluation" / "humans" / "by_participant"
VARIANT_FILE_KEY = {"C": "question", "B": "weaker_object", "A": "pronominalized"}
VARIANT_ORDER    = ["C", "B", "A"]
VARIANT_LABELS   = {"C": "C (Original)", "B": "B (Weaker)", "A": "A (Pronominalized)"}

# Group display order for the unified plot
GROUP_ORDER = ["human", "VLM", "VLM backbone decoder", "standalone LLM", "standalone LLM (think)"]
GROUP_LABEL = {
    "human":                   "Human",
    "VLM":                     "VLM",
    "VLM backbone decoder":    "Backbone decoder",
    "standalone LLM":          "Standalone LLM",
    "standalone LLM (think)":  "Standalone LLM (think)",
}
GROUP_BG = {
    "human":                   "#f0f4ff",
    "VLM":                     "#fff8f0",
    "VLM backbone decoder":    "#f0fff4",
    "standalone LLM":          "#fff0f8",
    "standalone LLM (think)":  "#f8f0ff",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         8,
    "axes.titlesize":    7,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6.5,
    "figure.dpi":        150,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def safe_name(label: str) -> str:
    return re.sub(r"[^\w]+", "_", label).strip("_").lower()


def plot_ecdf(ax, values: np.ndarray, color: str, label: str, lw: float = 1.3):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=color, lw=lw, label=label)


def save_fig(fig, name: str):
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_human_data() -> dict:
    """Return {variant: np.ndarray of normalised confidence [0,1]}."""
    records = []
    for fpath in sorted(HUMAN_DIR.glob("*.json")):
        d = json.load(open(fpath))
        for ans in d.get("answers", []):
            records.append({
                "variant":    ans["variant"],
                "conf_norm":  (ans["confidence"] - 1) / 4,
            })
    return {
        v: np.array([r["conf_norm"] for r in records if r["variant"] == v])
        for v in VARIANT_ORDER
    }


EOS_TOKENS = {"<|im_end|>", "</s>", "<|endoftext|>", "<eos>"}


def answer_logprobs(tokens: list) -> list:
    """Extract mean-logprob source tokens from a token list.

    For think models: tokens after </think> only (thinking chain excluded).
    For all models:   strip leading whitespace-only tokens and EOS tokens.
    Returns list of logprob floats; empty list if no answer tokens found.
    """
    # Split at </think> if present
    think_end = -1
    for i, t in enumerate(tokens):
        if "</think>" in t.get("token", ""):
            think_end = i
            break
    answer_tokens = tokens[think_end + 1:] if think_end >= 0 else tokens

    # Filter out EOS and pure-whitespace tokens
    lps = [
        t["logprob"] for t in answer_tokens
        if t.get("token", "").strip() and t.get("token") not in EOS_TOKENS
    ]
    return lps


def load_model_data(jsonl_path: Path) -> dict:
    """Return {variant: np.ndarray of mean logprob per answer token}."""
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


# ── Single-entity plot ────────────────────────────────────────────────────────

def make_single(ax, data: dict, xlabel: str, title: str, group: str,
                show_legend: bool = False):
    """Draw ECDF lines onto `ax`. Returns True if any data was plotted."""
    has_data = False
    ax.set_facecolor(GROUP_BG.get(group, "white"))
    for v in VARIANT_ORDER:
        vals = data[v]
        if len(vals):
            plot_ecdf(ax, vals, color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
            has_data = True
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel("ECDF",  fontsize=6)
    ax.set_title(title,    fontsize=7, pad=3)
    if show_legend:
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    return has_data


# ── Individual file outputs ───────────────────────────────────────────────────

def save_individual(label: str, data: dict, xlabel: str, group: str):
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    make_single(ax, data, xlabel, label, group, show_legend=True)
    fig.tight_layout()
    save_fig(fig, safe_name(label))


# ── Unified grid ──────────────────────────────────────────────────────────────

def make_unified_grid(entries: list):
    """
    entries: list of (label, group, data_dict, xlabel)
    Ordered: human first, then VLM, backbone, SA-LLM, think.
    """
    N     = len(entries)
    NCOLS = 6
    NROWS = (N + NCOLS - 1) // NCOLS

    fig, axes = plt.subplots(NROWS, NCOLS,
                              figsize=(NCOLS * 2.3, NROWS * 2.0),
                              squeeze=False)

    for idx, (label, group, data, xlabel) in enumerate(entries):
        r, c = divmod(idx, NCOLS)
        ax   = axes[r][c]
        make_single(ax, data, xlabel, label, group, show_legend=False)

    # hide unused axes
    for idx in range(N, NROWS * NCOLS):
        r, c = divmod(idx, NCOLS)
        axes[r][c].set_visible(False)

    # legend (shared) — draw on figure
    handles = [
        mpatches.Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
        for v in VARIANT_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    # group header annotations above first subplot of each group
    group_starts = {}
    for idx, (_, group, _, _) in enumerate(entries):
        if group not in group_starts:
            group_starts[group] = idx

    for group, idx in group_starts.items():
        r, c = divmod(idx, NCOLS)
        ax = axes[r][c]
        ax.annotate(
            f"[{GROUP_LABEL[group]}]",
            xy=(0, 1.22), xycoords="axes fraction",
            fontsize=6.5, color="#555555", style="italic",
            annotation_clip=False,
        )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, "_all")


# ── Main ──────────────────────────────────────────────────────────────────────

# Human
print("Human…")
human_data = load_human_data()
save_individual("human", human_data, "Confidence (normalised, 0–1)", "human")

# Models — collect all for unified grid
all_models   = default_all_models(ROOT)
model_entries = []   # (label, group, data, xlabel)

for label, (results_dir, model_subdir) in all_models.items():
    print(f"{label}…")
    jsonl = results_dir / model_subdir / "vqa_1k_control_inst_blind.jsonl"
    data  = load_model_data(jsonl)
    group = MODEL_TYPE.get(label, "standalone LLM")
    if all(len(v) == 0 for v in data.values()):
        print(f"  Skipping {label} — no inst_blind data")
        continue
    save_individual(label, data, "Mean logprob / token", group)
    model_entries.append((label, group, data, "Mean logprob / token"))

# Build unified grid: human first, then by group order
unified = [("human", "human", human_data, "Confidence (norm.)")]
for grp in GROUP_ORDER[1:]:   # skip "human", already added
    unified += [(l, g, d, x) for l, g, d, x in model_entries if g == grp]

print("Building unified grid…")
make_unified_grid(unified)

print("Done.")
