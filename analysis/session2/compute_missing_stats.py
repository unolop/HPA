"""
compute_missing_stats.py
Computes missing statistical tests for the paper.

Run with:
    conda run -n zero python3 analysis/session2/compute_missing_stats.py
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

EXPORTS = Path("/home/david/Desktop/yuna/HPA/analysis/session2/exports")
THINK_EDA = Path("/home/david/Desktop/yuna/HPA/figures/think_eda")
LOGITS_DIR = Path("/home/david/Desktop/yuna/HPA/evaluation/logits")

OUTPUT_FILE = EXPORTS / "missing_stat_tests.txt"

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


def section(title: str) -> None:
    log()
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading pair_cache_cleaned.parquet (inst_blind / abstention-filtered)...", file=sys.stderr)
pair_ib = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
pair_blind = pd.read_parquet(EXPORTS / "pair_cache_cleaned_blind.parquet")

hm = pair_ib[pair_ib["pair_type"] == "HM"].copy()
hm_blind = pair_blind[pair_blind["pair_type"] == "HM"].copy()

print(f"  HM inst_blind rows: {len(hm):,}", file=sys.stderr)
print(f"  HM blind rows:      {len(hm_blind):,}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Parameter count mapping
# ---------------------------------------------------------------------------
PARAM_MAP: dict[str, float] = {
    # VLM
    "Qwen3-VL-2B": 2, "Qwen3-VL-4B": 4, "Qwen3-VL-8B": 8, "Qwen3-VL-32B": 32,
    "InternVL-1B": 1, "InternVL-2B": 2, "InternVL-8B": 8,
    "LLaVA-Vicuna": 7, "LLaVA-Vicuna-13B": 13, "LLaVA-Mistral": 7, "LLaVA-1.5-7B": 7,
    # VLM backbone decoder
    "Qwen3-VL-2B (LM)": 2, "Qwen3-VL-4B (LM)": 4, "Qwen3-VL-8B (LM)": 8, "Qwen3-VL-32B (LM)": 32,
    "InternVL-1B (LM)": 1, "InternVL-2B (LM)": 2, "InternVL-8B (LM)": 8,
    "LLaVA-Vicuna (LM)": 7, "LLaVA-Vicuna-13B (LM)": 13, "LLaVA-Mistral (LM)": 7, "LLaVA-1.5 (LM)": 7,
    # Standalone LLM (no think)
    "Qwen3-0.6B": 0.6, "Qwen3-1.7B": 1.7, "Qwen3-4B": 4, "Qwen3-8B": 8, "Qwen3-32B": 32,
    "Qwen2.5-7B-Instruct": 7, "Mistral-7B": 7, "Phi-3.5-mini": 3.8, "Vicuna-7B": 7, "Vicuna-13B": 13,
    # Standalone LLM (think)
    "Qwen3-0.6B (think)": 0.6, "Qwen3-1.7B (think)": 1.7, "Qwen3-4B (think)": 4,
    "Qwen3-8B (think)": 8, "Qwen3-32B (think)": 32,
}

GROUP_ORDER = ["VLM", "VLM backbone decoder", "standalone LLM", "standalone LLM (think)"]

# ============================================================
# TEST 1: Scale vs HM SBERT (Pearson r per group)
# ============================================================
section("TEST 1: Scale vs HM SBERT within groups (Pearson r)")

log("Using pair_cache_cleaned.parquet (inst_blind, abstention-filtered).")
log("Per-model mean HM SBERT (averaged across all variants and questions).")
log()

for grp in GROUP_ORDER:
    grp_df = hm[hm["subject_group_2"] == grp]
    model_means = grp_df.groupby("subject_2")["sbert_score"].mean()

    xs: list[float] = []
    ys: list[float] = []
    model_names: list[str] = []
    missing: list[str] = []

    for model, mean_sbert in model_means.items():
        if model in PARAM_MAP:
            xs.append(PARAM_MAP[model])
            ys.append(mean_sbert)
            model_names.append(model)
        else:
            missing.append(model)

    n = len(xs)
    log(f"Group: {grp} (n={n} models)")

    if missing:
        log(f"  WARNING: no param mapping for: {missing}")

    if n < 3:
        log(f"  Too few models ({n}) — skipping Pearson r.")
        log()
        continue

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    r, pval = stats.pearsonr(xs_arr, ys_arr)

    log(f"  Pearson r = {r:.4f}  p = {pval:.4e}  n = {n}")
    log("  Model breakdown:")
    for mn, x, y in sorted(zip(model_names, xs, ys), key=lambda t: t[1]):
        log(f"    {mn:<30s}  params={x:.1f}B  mean_HM_SBERT={y:.4f}")
    log()

# ============================================================
# TEST 2: Thinking vs non-thinking HM SBERT (paired Wilcoxon)
# ============================================================
section("TEST 2: Thinking vs non-thinking HM SBERT (paired Wilcoxon)")

log("Using pair_cache_cleaned.parquet (inst_blind, abstention-filtered).")
log("Per-question mean HM SBERT (averaged over human raters AND variants).")
log()

THINK_PAIRS = [
    ("Qwen3-0.6B", "Qwen3-0.6B (think)"),
    ("Qwen3-1.7B", "Qwen3-1.7B (think)"),
    ("Qwen3-4B", "Qwen3-4B (think)"),
    ("Qwen3-8B", "Qwen3-8B (think)"),
    ("Qwen3-32B", "Qwen3-32B (think)"),
]

# Get per-question means (average over variants and human raters)
per_q_model = (
    hm.groupby(["question_id", "subject_2"])["sbert_score"].mean()
)

all_nothink_vals: list[float] = []
all_think_vals: list[float] = []
all_qids: list[int] = []

for base, think in THINK_PAIRS:
    size_label = base.replace("Qwen3-", "")

    try:
        base_s = per_q_model.xs(base, level="subject_2")
        think_s = per_q_model.xs(think, level="subject_2")
    except KeyError as e:
        log(f"  {size_label}: MISSING data for {e} — skipping")
        log()
        continue

    # Align on common question IDs
    common_idx = base_s.index.intersection(think_s.index)
    if len(common_idx) < 5:
        log(f"  {size_label}: too few matched questions ({len(common_idx)}) — skipping")
        continue

    b_vals = base_s.loc[common_idx].values
    t_vals = think_s.loc[common_idx].values

    diffs = t_vals - b_vals
    if np.all(diffs == 0):
        log(f"  {size_label}: all differences are zero — Wilcoxon test not applicable.")
        log(f"    mean_nothink = {np.mean(b_vals):.4f}  mean_think = {np.mean(t_vals):.4f}")
        log()
        all_nothink_vals.extend(b_vals)
        all_think_vals.extend(t_vals)
        continue

    stat, pval = stats.wilcoxon(b_vals, t_vals, alternative="two-sided")
    delta = float(np.mean(t_vals) - np.mean(b_vals))

    log(f"  {size_label}:")
    log(f"    mean_nothink = {np.mean(b_vals):.4f}")
    log(f"    mean_think   = {np.mean(t_vals):.4f}")
    log(f"    delta (think - nothink) = {delta:+.4f}")
    log(f"    Wilcoxon W = {stat:.1f}  p = {pval:.4e}  n_questions = {len(common_idx)}")
    log()

    all_nothink_vals.extend(b_vals)
    all_think_vals.extend(t_vals)

# Pooled across all sizes
if len(all_nothink_vals) >= 5:
    pool_diffs = np.array(all_think_vals) - np.array(all_nothink_vals)
    if np.all(pool_diffs == 0):
        log("  POOLED: all differences are zero — Wilcoxon not applicable.")
        log(f"    mean = {np.mean(all_nothink_vals):.4f}")
    else:
        stat_pool, pval_pool = stats.wilcoxon(all_nothink_vals, all_think_vals, alternative="two-sided")
        delta_pool = float(np.mean(all_think_vals) - np.mean(all_nothink_vals))
        log(f"  POOLED (all 5 sizes):")
        log(f"    mean_nothink = {np.mean(all_nothink_vals):.4f}")
        log(f"    mean_think   = {np.mean(all_think_vals):.4f}")
        log(f"    delta (think - nothink) = {delta_pool:+.4f}")
        log(f"    Wilcoxon W = {stat_pool:.1f}  p = {pval_pool:.4e}  n_total = {len(all_nothink_vals)}")
else:
    log("  POOLED: insufficient data")

# ============================================================
# TEST 3: Instruction effect on HM SBERT (paired Wilcoxon per group)
# ============================================================
section("TEST 3: Instruction effect on HM SBERT (paired Wilcoxon per group)")

log("pair_cache_cleaned.parquet  = inst_blind condition (abstention-filtered)")
log("pair_cache_cleaned_blind.parquet = blind condition (abstention-filtered)")
log()
log("For each group: per-question mean HM SBERT (across human raters AND variants),")
log("matched by (question_id, subject_2).  Two-sided Wilcoxon.")
log()

for grp in GROUP_ORDER:
    ib_grp = hm[hm["subject_group_2"] == grp]
    bl_grp = hm_blind[hm_blind["subject_group_2"] == grp]

    # Per (question_id, subject_2) mean
    ib_q = ib_grp.groupby(["question_id", "subject_2"])["sbert_score"].mean()
    bl_q = bl_grp.groupby(["question_id", "subject_2"])["sbert_score"].mean()

    common_idx = ib_q.index.intersection(bl_q.index)
    if len(common_idx) < 5:
        log(f"  {grp}: too few matched (q, model) pairs ({len(common_idx)}) — skipping")
        continue

    ib_vals = ib_q.loc[common_idx].values
    bl_vals = bl_q.loc[common_idx].values

    delta = float(np.mean(ib_vals) - np.mean(bl_vals))
    diffs = ib_vals - bl_vals
    log(f"  Group: {grp}")
    log(f"    mean_blind      = {np.mean(bl_vals):.4f}")
    log(f"    mean_inst_blind = {np.mean(ib_vals):.4f}")
    log(f"    delta (inst_blind - blind) = {delta:+.4f}")
    log(f"    n_pairs = {len(common_idx)}")

    if np.all(diffs == 0):
        log(f"    Wilcoxon: all differences are zero — not applicable.")
    else:
        stat, pval = stats.wilcoxon(bl_vals, ib_vals, alternative="two-sided")
        log(f"    Wilcoxon W = {stat:.1f}  p = {pval:.4e}")
    log()

# ============================================================
# TEST 4: Trace length vs token confidence (Pearson r)
# ============================================================
section("TEST 4: Trace length vs token confidence (Pearson r)")

log("Looking for think-model JSONL files with trace lengths and logprobs.")
log()

# Find all think model JSONL files
think_jsonl_files: list[Path] = []
for subdir in LOGITS_DIR.rglob("*.jsonl"):
    fname = subdir.name
    parent_str = str(subdir)
    # Check if it's from a think-mode directory
    if any(d in parent_str for d in ["backbone_think", "think"]):
        think_jsonl_files.append(subdir)

if not think_jsonl_files:
    # Try to find any Qwen3 think model files
    for subdir in LOGITS_DIR.rglob("*.jsonl"):
        fname = subdir.name
        if "think" in str(subdir.parent):
            think_jsonl_files.append(subdir)

log(f"Found {len(think_jsonl_files)} think-mode JSONL file(s).")
for f in think_jsonl_files:
    log(f"  {f}")
log()

# Build per-record (trace_length, mean_logprob) dataset from all found files
THINK_TOKEN_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
FINAL_ANS_RE = re.compile(r"</think>\s*(.*)", re.DOTALL)

trace_lengths: list[int] = []
mean_logprobs: list[float] = []
mean_final_logprobs: list[float] = []


def count_tokens_approx(text: str) -> int:
    """Approximate token count by splitting on whitespace."""
    return len(text.split())


def extract_logprobs_for_key(logits_dict: dict, key: str) -> list[float]:
    """Return list of top logprob values for each token position."""
    if key not in logits_dict:
        return []
    content = logits_dict[key].get("content", [])
    return [tok["logprob"] for tok in content if "logprob" in tok]


# Process files (limit to avoid OOM)
MAX_RECORDS_PER_FILE = 5000
processed_files = 0

for jsonl_path in think_jsonl_files:
    try:
        with open(jsonl_path) as fh:
            file_records = 0
            for raw_line in fh:
                if file_records >= MAX_RECORDS_PER_FILE:
                    break
                record = json.loads(raw_line.strip())

                gen_answers = record.get("generated_answers", {})
                gen_logits = record.get("generated_logits", {})

                # Use 'question' variant (original prompt)
                for key in list(gen_answers.keys()):
                    answer_text = gen_answers.get(key, "")
                    if not isinstance(answer_text, str):
                        continue

                    # Extract think trace length
                    think_match = THINK_TOKEN_RE.search(answer_text)
                    if think_match:
                        think_trace = think_match.group(1)
                        trace_len = count_tokens_approx(think_trace)
                    else:
                        # No think block — skip for this analysis
                        continue

                    # Extract all logprobs from this key
                    all_lps = extract_logprobs_for_key(gen_logits, key)
                    if not all_lps:
                        continue

                    mean_lp = float(np.mean(all_lps))

                    # Try to separate final answer logprobs from think logprobs
                    final_match = FINAL_ANS_RE.search(answer_text)
                    if final_match:
                        final_text = final_match.group(1).strip()
                        # Estimate: final answer tokens ≈ last few tokens
                        n_final_tokens = max(1, count_tokens_approx(final_text))
                        n_final_tokens = min(n_final_tokens, len(all_lps))
                        final_lps = all_lps[-n_final_tokens:]
                        mean_final_lp = float(np.mean(final_lps))
                    else:
                        mean_final_lp = mean_lp

                    trace_lengths.append(trace_len)
                    mean_logprobs.append(mean_lp)
                    mean_final_logprobs.append(mean_final_lp)

                    file_records += 1

        processed_files += 1
        log(f"  Processed {jsonl_path.name} from {jsonl_path.parent.name}: {file_records} records")

    except Exception as e:
        log(f"  Error processing {jsonl_path}: {e}")

log()

if len(trace_lengths) >= 3:
    arr_len = np.array(trace_lengths)
    arr_lp = np.array(mean_logprobs)
    arr_final_lp = np.array(mean_final_logprobs)

    r_all, p_all = stats.pearsonr(arr_len, arr_lp)
    r_final, p_final = stats.pearsonr(arr_len, arr_final_lp)

    log(f"  n = {len(arr_len):,} observations (think-trace token count vs mean token logprob)")
    log()
    log(f"  Pearson r (trace_length vs mean_logprob ALL tokens):")
    log(f"    r = {r_all:.4f}  p = {p_all:.4e}")
    log()
    log(f"  Pearson r (trace_length vs mean_logprob FINAL ANSWER tokens):")
    log(f"    r = {r_final:.4f}  p = {p_final:.4e}")
    log()
    log(f"  Descriptive stats:")
    log(f"    trace_length: mean={arr_len.mean():.1f}  median={np.median(arr_len):.1f}  sd={arr_len.std():.1f}")
    log(f"    mean_logprob: mean={arr_lp.mean():.4f}  sd={arr_lp.std():.4f}")

    # Also check think_length_correlations.csv which already has computed rho
    lc_path = THINK_EDA / "think_length_correlations.csv"
    if lc_path.exists():
        lc_df = pd.read_csv(lc_path)
        log()
        log("  Pre-computed Spearman rho from think_length_correlations.csv")
        log("  (trace_length vs final_words, pooled across variants):")
        lc_all = lc_df[lc_df["scope"] == "all"]
        if not lc_all.empty:
            for _, row in lc_all.iterrows():
                log(f"    {row['model']}: rho={row['rho']:.4f}  p={row['p_value']:.4e}  n={row['n']}")
        else:
            for _, row in lc_df.head(10).iterrows():
                log(f"    {row['model']} [{row['scope']}]: rho={row['rho']:.4f}  p={row['p_value']:.4e}  n={row['n']}")
else:
    log(f"  Insufficient data ({len(trace_lengths)} records) — computing from summary CSV instead.")

    # Fallback: use already-computed summary from EDA
    lc_path = THINK_EDA / "think_length_correlations.csv"
    if lc_path.exists():
        lc_df = pd.read_csv(lc_path)
        log()
        log("  Pre-computed Spearman rho from think_length_correlations.csv")
        log("  (trace_length vs final_words):")
        for _, row in lc_df.iterrows():
            log(f"    {row['model']} [{row['scope']}]: rho={row['rho']:.4f}  p={row['p_value']:.4e}  n={row['n']}")
    else:
        log("  think_length_correlations.csv not found — Test 4 not available.")

# ============================================================
# TEST 5: Within-class variance comparison (backbone vs VLMs)
# ============================================================
section("TEST 5: Within-class variance comparison (backbone decoders vs VLMs)")

log("Using pair_cache_cleaned.parquet (inst_blind, abstention-filtered).")
log("Per (question_id, subject_2) mean HM SBERT (across human raters).")
log("Then: within-question SD across models in each group.")
log()

target_groups = ["VLM", "VLM backbone decoder"]

# Per (question, model) mean across human raters
q_model_means: dict[str, pd.Series] = {}
raw_scores: dict[str, np.ndarray] = {}

for grp in target_groups:
    grp_df = hm[hm["subject_group_2"] == grp]
    per_qm = grp_df.groupby(["question_id", "subject_2"])["sbert_score"].mean()
    q_model_means[grp] = per_qm
    raw_scores[grp] = per_qm.values
    log(f"  {grp}: {len(per_qm):,} (question, model) pairs, {grp_df['subject_2'].nunique()} models")

log()

# Per-question SD across models within each group
for grp in target_groups:
    per_qm = q_model_means[grp]
    per_q_sd = per_qm.groupby("question_id").std(ddof=1)
    log(f"  {grp}:")
    log(f"    Mean within-question SD across models = {per_q_sd.mean():.4f}")
    log(f"    Median within-question SD             = {per_q_sd.median():.4f}")
    log(f"    N questions with >= 2 models           = {(per_qm.groupby('question_id').count() >= 2).sum()}")
    log()

# Levene's test on raw per-(question, model) SBERT values
vlm_scores = raw_scores.get("VLM", np.array([]))
bb_scores = raw_scores.get("VLM backbone decoder", np.array([]))

if len(vlm_scores) > 1 and len(bb_scores) > 1:
    lev_stat, lev_p = stats.levene(vlm_scores, bb_scores)
    log(f"  Levene's test (VLM vs backbone decoder raw scores):")
    log(f"    F = {lev_stat:.4f}  p = {lev_p:.4e}")
    log(f"    n_VLM = {len(vlm_scores):,}  n_backbone = {len(bb_scores):,}")

    # Also Levene on per-question SDs
    vlm_q_sd = q_model_means["VLM"].groupby("question_id").std(ddof=1).dropna()
    bb_q_sd = q_model_means["VLM backbone decoder"].groupby("question_id").std(ddof=1).dropna()

    # Align on common questions
    common_q = vlm_q_sd.index.intersection(bb_q_sd.index)
    if len(common_q) >= 5:
        lev2_stat, lev2_p = stats.levene(vlm_q_sd.loc[common_q].values, bb_q_sd.loc[common_q].values)
        log()
        log(f"  Levene's test on per-question SDs (matched questions, n={len(common_q)}):")
        log(f"    F = {lev2_stat:.4f}  p = {lev2_p:.4e}")
        log(f"    Mean VLM within-Q SD     = {vlm_q_sd.loc[common_q].mean():.4f}")
        log(f"    Mean backbone within-Q SD = {bb_q_sd.loc[common_q].mean():.4f}")
else:
    log("  Levene's test: insufficient data for one or both groups.")

# ============================================================
# Final: write output
# ============================================================
log()
log("=" * 70)
log("  Done.")
log("=" * 70)

output = "\n".join(lines)
OUTPUT_FILE.write_text(output)
print(f"\nSaved to: {OUTPUT_FILE}", file=sys.stderr)
