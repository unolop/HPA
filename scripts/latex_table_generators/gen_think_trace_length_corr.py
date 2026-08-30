"""
Generate Table: Pearson r between per-question reasoning-trace length and HM SBERT.

Uses all 113 study questions × 3 variants (up to 339 pairs), inst_blind condition.
Excludes empty traces. Separate r and N columns per variant.

Output: latex/AAAI2026/LaTeX/tables/supp/think_trace_length_corr.tex
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = ROOT.parent / "human-prior-alignment"
sys.path.insert(0, str(ANALYSIS_ROOT / "analysis"))

from utils.load_data import load_hh_hm

LOGITS_BASE = ROOT / "evaluation/logits/backbone/pretrained"
OUT = ROOT / "latex/AAAI2026/LaTeX/tables/supp/think_trace_length_corr.tex"

MODEL_DIRS = {
    "Qwen3-0.6B_think": ("0.6B", "Qwen3-0.6B (think)"),
    "Qwen3-1.7B_think": ("1.7B", "Qwen3-1.7B (think)"),
    "Qwen3-4B_think":   ("4B",   "Qwen3-4B (think)"),
    "Qwen3-8B_think":   ("8B",   "Qwen3-8B (think)"),
    "Qwen3-32B_think":  ("32B",  "Qwen3-32B (think)"),
}
VARIANT_MAP = {"question": "C", "weaker_object": "B", "pronominalized": "A"}
VARIANT_ORDER = ["C", "B", "A"]
VARIANT_LABELS = {"C": "Original", "B": "Weaker", "A": "Pronominalized"}
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ── 1. Parse JSONL → (model, question_id, variant, think_words) ──────────────

def parse_jsonl(fpath: Path, model_name: str) -> list[dict]:
    rows = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["question_id"]
            for vkey, vlabel in VARIANT_MAP.items():
                ans = rec.get("generated_answers", {}).get(vkey, "") or ""
                m = THINK_RE.search(ans)
                think_text = m.group(1) if m else ""
                think_words = len(think_text.split()) if think_text.strip() else 0
                is_empty = think_text.strip() == ""
                rows.append({
                    "model_name":    model_name,
                    "question_id":   int(qid),
                    "variant":       vlabel,
                    "think_words":   think_words,
                    "is_empty":      is_empty,
                })
    return rows


print("Parsing JSONL files...", file=sys.stderr)
all_rows = []
for model_dir, (size_label, model_name) in MODEL_DIRS.items():
    fpath = LOGITS_BASE / model_dir / "vqa_1k_control_inst_blind.jsonl"
    if not fpath.exists():
        print(f"  WARNING: {fpath} not found", file=sys.stderr)
        continue
    rows = parse_jsonl(fpath, model_name)
    print(f"  {model_name}: {len(rows)} rows", file=sys.stderr)
    all_rows.extend(rows)

think_df = pd.DataFrame(all_rows)


# ── 2. Load pair_cache → HM SBERT per (model, question_id, variant) ──────────

print("Loading pair_cache...", file=sys.stderr)
_, hm_all = load_hh_hm(filtered=True)

think_names = [name for _, name in MODEL_DIRS.values()]
hm = (
    hm_all[hm_all["subject_2"].isin(think_names)]
    .groupby(["subject_2", "question_id", "variant"])["sbert_score"]
    .mean()
    .reset_index()
    .rename(columns={"subject_2": "model_name", "sbert_score": "hm_sbert"})
)
hm["question_id"] = hm["question_id"].astype(int)

study_qids = set(hm["question_id"].unique())
print(f"  Study question IDs: {len(study_qids)}", file=sys.stderr)


# ── 3. Merge, filter to study questions + non-empty traces ───────────────────

df = think_df[think_df["question_id"].isin(study_qids)].copy()
df = df.merge(hm, on=["model_name", "question_id", "variant"], how="left")
df_nonzero = df[~df["is_empty"]].copy()


# ── 4. Compute Pearson r per model × variant ─────────────────────────────────

def pearson(x, y):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[mask], y[mask]
    n = int(mask.sum())
    if n < 5:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x, y)
    return float(r), float(p), n


def fmt_r(r, p):
    """Format r value: bold if significant, with stars."""
    if np.isnan(r):
        return "---"
    sign = "+" if r >= 0 else "-"
    digits = f"{abs(r):.3f}".split(".")[1]
    s = f"{sign}.{digits}"
    star = ""
    if not np.isnan(p):
        if p < 0.01:
            star = "^{**}"
        elif p < 0.05:
            star = "^{*}"
    if not np.isnan(p) and p < 0.05:
        return f"$\\mathbf{{{s}{star}}}$"
    return f"${s}$"


results = {}
for model_dir, (size_label, model_name) in MODEL_DIRS.items():
    results[model_name] = {}
    sub = df_nonzero[df_nonzero["model_name"] == model_name]
    for v in VARIANT_ORDER:
        sv = sub[sub["variant"] == v]
        r, p, n = pearson(sv["think_words"].values.astype(float),
                          sv["hm_sbert"].values.astype(float))
        results[model_name][v] = (r, p, n)

    print(f"  {model_name}:", file=sys.stderr)
    for v in VARIANT_ORDER:
        rv, pv, nv = results[model_name][v]
        print(f"    {v}: r={rv:.3f}, p={pv:.4f}, n={nv}", file=sys.stderr)


# ── 5. Build and write LaTeX ─────────────────────────────────────────────────

tex_rows = []
for model_dir, (size_label, model_name) in MODEL_DIRS.items():
    cells = [size_label]
    for v in VARIANT_ORDER:
        r, p, n = results[model_name][v]
        cells.append(fmt_r(r, p))
        cells.append(str(n))
    tex_rows.append(" & ".join(cells) + r" \\")

tex = (
    r"\begin{table}[h]" "\n"
    r"\centering\small\setlength{\tabcolsep}{5pt}" "\n"
    r"\begin{tabular}{l cc cc cc}" "\n"
    r"\toprule" "\n"
    r"& \multicolumn{2}{c}{\textbf{Original}} & \multicolumn{2}{c}{\textbf{Weaker}} & \multicolumn{2}{c}{\textbf{Pronominalized}} \\" "\n"
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
    r"\textbf{Size} & $r$ & $N$ & $r$ & $N$ & $r$ & $N$ \\" "\n"
    r"\midrule" "\n"
    + "\n".join(tex_rows) + "\n"
    r"\bottomrule" "\n"
    r"\end{tabular}" "\n"
    r"\caption{Pearson $r$ between reasoning-trace length and $\sHM$ per question--variant pair, "
    r"for Qwen3 models with thinking enabled (free-text questions only, N$\leq$113, "
    r"where $N$ excludes abstained and empty-trace pairs). "
    r"Stars: ${}^{**}p{<}.01$, ${}^{*}p{<}.05$; significant values in bold.}" "\n"
    r"\label{tab:think_trace_length_corr}" "\n"
    r"\end{table}"
)

OUT.write_text(tex)
print(f"\nWritten to {OUT}", file=sys.stderr)
print("\n" + tex)
