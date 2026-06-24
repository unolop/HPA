"""
Export image-ablation figures and audit tables (Notebook 13 replacement).

This exporter standardizes image-condition ablation outputs for VLMs and writes
results into a dedicated folder:

  figures/image_ablation/plots/
  figures/image_ablation/tables/

What it does:
  1) Audits availability/completeness of blind ablation files per VLM
     (blank / gray / noise / white).
  2) Selects "complete" models (blank+gray+noise with >= --min_rows examples).
  3) Exports:
       - availability table (all VLMs)
       - summary table (selected models × condition)
       - accuracy-by-condition grouped bar chart
       - answer-change-vs-blank grouped bar chart

Run from repo root:
  conda run -n aide python figures/image_ablation/export_figures.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from utils.vqa import preprocess_answer, VQAAnswerMapper, vqa_accuracy
from helpers import clear_output_plots, save_fig


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete existing plot files in the output folder before exporting.",
)
parser.add_argument(
    "--min_rows",
    type=int,
    default=1000,
    help="Minimum rows per condition for a model to be treated as complete (default: 1000).",
)
parser.add_argument(
    "--models",
    type=str,
    default="",
    help="Optional comma-separated display names to include (e.g., 'Qwen3-VL-4B,Qwen3-VL-8B').",
)
parser.add_argument(
    "--human_subset",
    action="store_true",
    help=(
        "Restrict analysis to the human-study shared question subset "
        "(question_ids from analysis/session2/exports/responses_human.csv)."
    ),
)
parser.add_argument(
    "--human_subset_csv",
    type=str,
    default="",
    help=(
        "Optional path to responses_human.csv used to derive question_ids. "
        "Defaults to analysis/session2/exports/responses_human.csv."
    ),
)
parser.add_argument(
    "--annotations",
    type=str,
    default="",
    help=(
        "Optional explicit path to v2_mscoco_val2014_annotations.json. "
        "If omitted, the script prefers dataset/vqa inside this repo."
    ),
)
args = parser.parse_args()


# ── Paths / styles ────────────────────────────────────────────────────────────
OUT_ROOT = ROOT / "figures/image_ablation"
PLOTS_DIR = OUT_ROOT / "plots"
TABLES_DIR = OUT_ROOT / "tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
clear_output_plots(PLOTS_DIR, overwrite=args.overwrite)

LOGITS_VLM = ROOT / "evaluation/logits/vlm/pretrained"
LOGITS_ABL = ROOT / "evaluation/logits/ablation/pretrained"

COND_ORDER = ["blank", "gray", "noise", "white"]
COND_FILE_CONTROL = {
    "blank": "vqa_1k_control_blind.jsonl",
    "gray": "vqa_1k_control_blind_gray.jsonl",
    "noise": "vqa_1k_control_blind_noise.jsonl",
    "white": "vqa_1k_control_blind_white.jsonl",
}
COND_FILE_ABLATION = {
    "blank": "vqa_1k_blind.jsonl",
    "gray": "vqa_1k_blind_gray.jsonl",
    "noise": "vqa_1k_blind_noise.jsonl",
    "white": "vqa_1k_blind_white.jsonl",
}

COND_COLORS = {
    "blank": "#455A64",
    "gray": "#78909C",
    "noise": "#E53935",
    "white": "#FB8C00",
}
COND_LABEL = {
    "blank": "Blank",
    "gray": "Gray",
    "noise": "Noise",
    "white": "White",
}

VLM_DIR_TO_MODEL = {
    "InternVL3_5-1B": "InternVL-1B",
    "InternVL3_5-2B": "InternVL-2B",
    "InternVL3_5-8B": "InternVL-8B",
    "llava-1.5-7b-hf": "LLaVA-1.5-7B",
    "llava-v1.6-mistral-7b-hf": "LLaVA-Mistral",
    "llava-v1.6-vicuna-13b-hf": "LLaVA-Vicuna-13B",
    "llava-v1.6-vicuna-7b-hf": "LLaVA-Vicuna",
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B",
    "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
    "Qwen3-VL-32B-Instruct": "Qwen3-VL-32B",
}

# Prefer this order when present; append any remaining models alphabetically.
PREFERRED_MODEL_ORDER = [
    "Qwen3-VL-4B",
    "Qwen3-VL-8B",
    "LLaVA-Mistral",
    "Qwen3-VL-2B",
    "Qwen3-VL-32B",
    "InternVL-1B",
    "InternVL-2B",
    "InternVL-8B",
    "LLaVA-1.5-7B",
    "LLaVA-Vicuna",
    "LLaVA-Vicuna-13B",
]

SOFT_ABSTAIN = {
    "nothing", "none", "nowhere", "unanswerable", "unknown",
    "no answer", "no image", "no picture", "i don't know",
    "i'm sorry", "i cannot", "i can not", "i am unable",
    "i am not able", "unable to",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


def model_name_from_dir(dirname: str) -> str:
    return VLM_DIR_TO_MODEL.get(dirname, dirname)


def is_soft_abstain(ans: str) -> bool:
    a = ans.lower().strip()
    return any(a.startswith(tok) for tok in SOFT_ABSTAIN)


def n_rows_jsonl(path: Path) -> int:
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def preferred_source_for_model(model_dir: str) -> str:
    """
    Choose one primary source per model to avoid mixing runs across conditions.

    Priority:
      1) ablation, if blank/gray/noise all exist there
      2) control, if blank/gray/noise all exist there
      3) mixed fallback
    """
    need = ("blank", "gray", "noise")
    abl_ok = all((LOGITS_ABL / model_dir / COND_FILE_ABLATION[c]).exists() for c in need)
    if abl_ok:
        return "ablation"
    ctl_ok = all((LOGITS_VLM / model_dir / COND_FILE_CONTROL[c]).exists() for c in need)
    if ctl_ok:
        return "control"
    return "mixed"


def resolve_condition_path(model_dir: str, cond: str, preferred_source: str) -> tuple[Path | None, str]:
    """Resolve per-model condition path honoring preferred source first."""
    control_path = LOGITS_VLM / model_dir / COND_FILE_CONTROL[cond]
    ablation_path = LOGITS_ABL / model_dir / COND_FILE_ABLATION[cond]

    if preferred_source == "ablation":
        if ablation_path.exists():
            return ablation_path, "ablation"
        if control_path.exists():
            return control_path, "control"
    elif preferred_source == "control":
        if control_path.exists():
            return control_path, "control"
        if ablation_path.exists():
            return ablation_path, "ablation"
    else:  # mixed fallback
        if ablation_path.exists():
            return ablation_path, "ablation"
        if control_path.exists():
            return control_path, "control"
    return None, "missing"


def load_condition(path: Path, mapper: VQAAnswerMapper, subset_qids: set[int] | None = None) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            ex = json.loads(line.strip())
            qid = int(ex["question_id"])
            if subset_qids is not None and qid not in subset_qids:
                continue
            raw = ex.get("generated_answers", {}).get("question", "")
            ans = preprocess_answer(str(raw), strip_think=True)
            gt = mapper.get_answers(qid)
            acc = vqa_accuracy(ans, gt) if gt else float("nan")
            atype = ex.get("answer_type")
            if atype is None:
                atype = mapper.annotations.get(qid, {}).get("answer_type", "other")
            rows.append(
                {
                    "qid": qid,
                    "answer": ans,
                    "raw": str(raw),
                    "acc": acc,
                    "atype": atype,
                    "is_abstain": is_soft_abstain(ans),
                    "is_loop": "<|im_start|>" in str(raw),
                }
            )
    return pd.DataFrame(rows)


def order_models(models: list[str]) -> list[str]:
    pref = [m for m in PREFERRED_MODEL_ORDER if m in models]
    rest = sorted([m for m in models if m not in pref])
    return pref + rest


def normalize_model_filter(raw: str) -> set[str] | None:
    raw = raw.strip()
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def resolve_human_subset_qids(csv_path_raw: str) -> tuple[set[int], int]:
    if csv_path_raw.strip():
        csv_path = Path(csv_path_raw.strip())
    else:
        csv_path = ROOT / "analysis" / "session2" / "exports" / "responses_human.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Human subset CSV not found: {csv_path}")
    human_df = pd.read_csv(csv_path, usecols=["question_id", "participant"])
    qids = set(human_df["question_id"].dropna().astype(int).unique().tolist())
    n_participants = int(human_df["participant"].nunique())
    if not qids:
        raise ValueError(f"No question_id values found in {csv_path}")
    print(f"Using human subset from {csv_path} (q={len(qids)}, h={n_participants})")
    return qids, n_participants


def resolve_annotations_path(cli_path: str) -> Path:
    """Pick an annotation file that exists, preferring repo-local dataset/vqa."""
    candidates: list[Path] = []
    if cli_path.strip():
        candidates.append(Path(cli_path.strip()))
    env_path = os.environ.get("VQA_ANNOT")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(ROOT / "dataset" / "vqa" / "v2_mscoco_val2014_annotations.json")

    for p in candidates:
        if p.exists():
            return p
    # Keep first candidate for diagnostics when none exists.
    return candidates[0] if candidates else ROOT / "dataset" / "vqa" / "v2_mscoco_val2014_annotations.json"


print("Loading VQA annotations…")
annot_path = resolve_annotations_path(args.annotations)
print(f"Using annotations: {annot_path}")
mapper = VQAAnswerMapper(str(annot_path))
subset_qids: set[int] | None = None
subset_n_h: int | None = None
if args.human_subset:
    subset_qids, subset_n_h = resolve_human_subset_qids(args.human_subset_csv)

model_filter = normalize_model_filter(args.models)
model_dirs = sorted([p.name for p in LOGITS_VLM.iterdir() if p.is_dir()])

availability_rows: list[dict] = []
data: dict[str, dict[str, pd.DataFrame]] = {}
n_by_model_cond: dict[str, dict[str, int]] = {}

for model_dir in model_dirs:
    model = model_name_from_dir(model_dir)
    if model_filter is not None and model not in model_filter:
        continue

    preferred_source = preferred_source_for_model(model_dir)
    cond_dfs: dict[str, pd.DataFrame] = {}
    cond_counts: dict[str, int] = {}
    for cond in COND_ORDER:
        path, source = resolve_condition_path(model_dir, cond, preferred_source)
        if path is None:
            availability_rows.append(
                {
                    "model": model,
                    "model_dir": model_dir,
                    "condition": cond,
                    "preferred_source": preferred_source,
                    "source": source,
                    "rows_raw": 0,
                    "rows_used": 0,
                    "path": "",
                    "exists": False,
                }
            )
            cond_counts[cond] = 0
            continue

        n_raw = n_rows_jsonl(path)
        df_cond = load_condition(path, mapper, subset_qids=subset_qids)
        n_used = len(df_cond)
        availability_rows.append(
            {
                "model": model,
                "model_dir": model_dir,
                "condition": cond,
                "preferred_source": preferred_source,
                "source": source,
                "rows_raw": n_raw,
                "rows_used": n_used,
                "path": str(path),
                "exists": True,
            }
        )
        cond_counts[cond] = n_used
        if n_used > 0:
            cond_dfs[cond] = df_cond

    if cond_dfs:
        data[model] = cond_dfs
        n_by_model_cond[model] = cond_counts

availability_df = pd.DataFrame(availability_rows)
availability_out = TABLES_DIR / "vlm_ablation_availability.csv"
availability_df.to_csv(availability_out, index=False)
print(f"Saved: {availability_out}")


min_rows_required = args.min_rows
if subset_qids is not None:
    min_rows_required = min(args.min_rows, len(subset_qids))


def is_complete_model(model: str) -> bool:
    counts = n_by_model_cond.get(model, {})
    return (
        counts.get("blank", 0) >= min_rows_required
        and counts.get("gray", 0) >= min_rows_required
        and counts.get("noise", 0) >= min_rows_required
    )


selected_models = order_models([m for m in data if is_complete_model(m)])
if not selected_models:
    print(
        f"\nNo complete models found for required conditions (blank+gray+noise, min_rows={min_rows_required}). "
        "Check availability table."
    )
    sys.exit(0)

print("\nSelected complete models:")
for m in selected_models:
    c = n_by_model_cond[m]
    print(f"  {m}: blank={c.get('blank',0)} gray={c.get('gray',0)} noise={c.get('noise',0)} white={c.get('white',0)}")

# Common qid universe across selected models/conditions used in comparison plots.
common_qids: set[int] | None = None
for model in selected_models:
    for cond in ("blank", "gray", "noise"):
        qids = set(data[model][cond]["qid"].unique())
        common_qids = qids if common_qids is None else (common_qids & qids)
n_q_common = len(common_qids or set())
if subset_qids is not None:
    h_tag = subset_n_h if subset_n_h is not None else "na"
    suffix = f"_q{n_q_common}_h{h_tag}"
else:
    suffix = f"_q{n_q_common}" if n_q_common else ""

# ── Summary table (selected models × available conditions) ───────────────────
summary_rows = []
for model in selected_models:
    blank_ans = data[model]["blank"].set_index("qid")["answer"]
    for cond in COND_ORDER:
        if cond not in data[model]:
            continue
        df = data[model][cond]
        other = df.set_index("qid")["answer"]
        common = blank_ans.index.intersection(other.index)
        changed = float("nan") if cond == "blank" else (blank_ans[common] != other[common]).mean()
        summary_rows.append(
            {
                "model": model,
                "condition": cond,
                "mean_acc": float(df["acc"].mean()),
                "abstain_rate": float(df["is_abstain"].mean()),
                "loop_rate": float(df["is_loop"].mean()),
                "changed_vs_blank": float(changed) if not np.isnan(changed) else np.nan,
                "n_rows": int(len(df)),
            }
        )

summary_df = pd.DataFrame(summary_rows)
summary_out = TABLES_DIR / f"vlm_ablation_summary{suffix}.csv"
summary_df.to_csv(summary_out, index=False)
print(f"Saved: {summary_out}")

# ── Plot 1: accuracy by condition ─────────────────────────────────────────────
print("\nExporting accuracy-by-condition plot…")
plot_conds = [c for c in COND_ORDER if any(c in data[m] for m in selected_models)]
x = np.arange(len(selected_models))
w = 0.18

fig, ax = plt.subplots(figsize=(9, 4.8))
for ci, cond in enumerate(plot_conds):
    vals = []
    for model in selected_models:
        if cond not in data[model]:
            vals.append(np.nan)
            continue
        vals.append(float(data[model][cond]["acc"].mean()))
    offset = (ci - (len(plot_conds) - 1) / 2) * w
    bars = ax.bar(
        x + offset,
        vals,
        w * 0.9,
        color=COND_COLORS[cond],
        alpha=0.86,
        label=COND_LABEL[cond],
    )
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

ax.set_xticks(x)
ax.set_xticklabels(selected_models, fontsize=9)
ax.set_ylabel("Mean VQA accuracy (variant C)", fontsize=10)
ax.set_title(
    "Image Ablation: Accuracy by Condition (complete VLMs)\n"
    f"Required conditions: blank/gray/noise, q={n_q_common}",
    fontsize=10,
)
ax.legend(fontsize=9, frameon=True, loc="upper right")
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
ax.grid(axis="y", color="#E0E0E0", lw=0.7)
fig.tight_layout()
save_fig(fig, PLOTS_DIR, f"accuracy_by_condition_vlm_complete{suffix}.png", dpi=180)
plt.close(fig)

# ── Plot 2: answer change vs blank (grouped bar) ────────────────────────────
print("Exporting answer-change-vs-blank plot…")
change_conds = [c for c in plot_conds if c != "blank"]
x2 = np.arange(len(selected_models))
w2 = 0.22

fig, ax = plt.subplots(figsize=(9, 4.8))
for ci, cond in enumerate(change_conds):
    vals = []
    for model in selected_models:
        if cond not in data[model]:
            vals.append(np.nan)
            continue
        blank_ans = data[model]["blank"].set_index("qid")["answer"]
        other = data[model][cond].set_index("qid")["answer"]
        common = blank_ans.index.intersection(other.index)
        changed = (blank_ans[common] != other[common]).mean() if len(common) else np.nan
        vals.append(changed)
    offset = (ci - (len(change_conds) - 1) / 2) * w2
    bars = ax.bar(
        x2 + offset,
        vals,
        w2 * 0.9,
        color=COND_COLORS[cond],
        alpha=0.86,
        label=COND_LABEL[cond],
    )
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{v*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

ax.set_xticks(x2)
ax.set_xticklabels(selected_models, fontsize=9)
ax.set_ylabel("Fraction of answers changed vs Blank (black)", fontsize=10)
ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.15))
ax.set_title(
    "Image Ablation: Answer Change Rate vs Blank Baseline (complete VLMs)\n"
    f"q={n_q_common}",
    fontsize=10,
)
ax.legend(fontsize=9, frameon=True, loc="upper right")
ax.grid(axis="y", color="#E0E0E0", lw=0.7)
fig.tight_layout()
save_fig(fig, PLOTS_DIR, f"answer_change_vs_blank_vlm_complete{suffix}.png", dpi=180)
plt.close(fig)

print("\nDone.")
print(f"Plots : {PLOTS_DIR}")
print(f"Tables: {TABLES_DIR}")
