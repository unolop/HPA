"""
Generate LaTeX table for scale analysis of VLM/LM models.
Output: latex/AAAI2026/LaTeX/tables/scale_alignment_ranking.tex
"""
import pandas as pd
import numpy as np
import re
import os
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

DATA_DIR = "analysis/session2/exports/"
OUT_PATH = "latex/AAAI2026/LaTeX/tables/scale_alignment_ranking.tex"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
pairs  = pd.read_csv(os.path.join(DATA_DIR, "answer_pairs_sbert_text.csv"))
inst   = pd.read_csv(os.path.join(DATA_DIR, "responses_model_inst_blind.csv"))
blind  = pd.read_csv(os.path.join(DATA_DIR, "responses_model_blind.csv"))
human  = pd.read_csv(os.path.join(DATA_DIR, "answers_human_q113.csv"))

# ── Model roster ───────────────────────────────────────────────────────────────
# (display_name, size_str, family, type_str)
ROSTER = [
    # InternVL VLM
    ("InternVL-1B",         "1B",  "InternVL",     "VLM"),
    ("InternVL-2B",         "2B",  "InternVL",     "VLM"),
    ("InternVL-8B",         "8B",  "InternVL",     "VLM"),
    # InternVL LM
    ("InternVL-1B (LM)",    "1B",  "InternVL",     "LM"),
    ("InternVL-2B (LM)",    "2B",  "InternVL",     "LM"),
    ("InternVL-8B (LM)",    "8B",  "InternVL",     "LM"),
    # Qwen3-VL VLM
    ("Qwen3-VL-2B",         "2B",  "Qwen3-VL",     "VLM"),
    ("Qwen3-VL-4B",         "4B",  "Qwen3-VL",     "VLM"),
    ("Qwen3-VL-8B",         "8B",  "Qwen3-VL",     "VLM"),
    # Qwen3-VL LM
    ("Qwen3-VL-2B (LM)",    "2B",  "Qwen3-VL",     "LM"),
    ("Qwen3-VL-4B (LM)",    "4B",  "Qwen3-VL",     "LM"),
    ("Qwen3-VL-8B (LM)",    "8B",  "Qwen3-VL",     "LM"),
    # LLaVA-Vicuna VLM
    ("LLaVA-Vicuna",        "7B",  "LLaVA-Vicuna", "VLM"),
    ("LLaVA-Vicuna-13B",    "13B", "LLaVA-Vicuna", "VLM"),
    # LLaVA-Vicuna LM
    ("LLaVA-Vicuna (LM)",   "7B",  "LLaVA-Vicuna", "LM"),
    ("LLaVA-Vicuna-13B (LM)","13B","LLaVA-Vicuna", "LM"),
    # Qwen3 SA
    ("Qwen3-0.6B",          "0.6B","Qwen3",        "SA"),
    ("Qwen3-1.7B",          "1.7B","Qwen3",        "SA"),
    ("Qwen3-4B",            "4B",  "Qwen3",        "SA"),
    ("Qwen3-8B",            "8B",  "Qwen3",        "SA"),
    # Qwen3 Think
    ("Qwen3-1.7B (think)",  "1.7B","Qwen3",        "Think"),
    ("Qwen3-4B (think)",    "4B",  "Qwen3",        "Think"),
    ("Qwen3-8B (think)",    "8B",  "Qwen3",        "Think"),
    # Vicuna SA
    ("Vicuna-7B",           "7B",  "Vicuna",       "SA"),
    ("Vicuna-13B",          "13B", "Vicuna",       "SA"),
]

# ── Available models in each CSV ───────────────────────────────────────────────
pairs_models  = set(pairs[pairs['pair_type']=='HM']['subject_2'].unique())
inst_models   = set(inst['model'].unique())
blind_models  = set(blind['model'].unique())

# ── Helpers ────────────────────────────────────────────────────────────────────
ABSTAIN_PATTERNS = [
    r'cannot', r"can't", r'unable', r'no image', r'without image',
    r'no visual', r'unanswerable', r'not possible', r'no context',
    r"don't have", r'i cannot', r'no picture', r'no photo',
    r'as an ai', r'sorry', r"don't know", r'no way to know',
    r'impossible to', r'difficult to answer', r'unable to determine',
    r'cannot determine', r'hard to say',
]
ABSTAIN_RE = re.compile('|'.join(ABSTAIN_PATTERNS), re.IGNORECASE)

def is_abstain(resp):
    if pd.isna(resp):
        return False
    return bool(ABSTAIN_RE.search(str(resp)))

def norm_yesno(resp):
    if pd.isna(resp):
        return None
    r = str(resp).strip().lower()
    if r == 'yes':
        return 'yes'
    if r == 'no':
        return 'no'
    return None

def norm_count(resp):
    if pd.isna(resp):
        return None
    m = re.search(r'\b(\d+)\b', str(resp))
    if not m:
        return None
    n = int(m.group(1))
    return min(n, 10)

def js_divergence(a_vals, b_vals, bins):
    """Compute JS divergence between two lists of discrete values over bins."""
    def dist(vals):
        counts = {b: 0 for b in bins}
        for v in vals:
            if v in counts:
                counts[v] += 1
        total = sum(counts.values())
        if total == 0:
            return np.ones(len(bins)) / len(bins)
        return np.array([counts[b] / total for b in bins])
    p = dist(a_vals)
    q = dist(b_vals)
    return float(jensenshannon(p, q) ** 2)

# ── Pre-compute human Y/N and count distributions ──────────────────────────────
human_yn_vals = [norm_yesno(r) for r in human[human['op']=='exist']['response']]
human_yn_vals = [v for v in human_yn_vals if v is not None]

count_bins = list(range(11))  # 0..10
human_cnt_vals = [norm_count(r) for r in human[human['op']=='count']['response']]
human_cnt_vals = [v for v in human_cnt_vals if v is not None]

# ── Free-text filter for SBERT ─────────────────────────────────────────────────
FREE_TEXT_OPS = ~pairs['op'].isin(['exist', 'count'])
hm_free = pairs[(pairs['pair_type']=='HM') & FREE_TEXT_OPS].copy()
hh_free = pairs[(pairs['pair_type']=='HH') & FREE_TEXT_OPS].copy()

# Per (question_id, variant) HH mean — shared across models
hh_per_q = hh_free.groupby(['question_id','variant'])['sbert_score'].mean().rename('sHH')

# Human reference values
HUMAN_SHH   = hh_free['sbert_score'].mean()   # pooled sHH across all questions/variants
HUMAN_R     = 1.0                              # perfect correlation
HUMAN_DJS_YN    = 0.271                        # human LOO median DJS Y/N
HUMAN_DJS_COUNT = 0.524                        # human LOO median DJS Count

# ── Compute metrics for each model ─────────────────────────────────────────────
results = {}

for (model, size, family, mtype) in ROSTER:
    row = {}

    # sHM
    if model in pairs_models:
        sub = hm_free[hm_free['subject_2'] == model]
        row['sHM'] = sub['sbert_score'].mean() if len(sub) > 0 else np.nan
    else:
        row['sHM'] = np.nan

    # r — Pearson(sHM_per_q, sHH_per_q)
    if model in pairs_models:
        sub = hm_free[hm_free['subject_2'] == model]
        hm_per_q = sub.groupby(['question_id','variant'])['sbert_score'].mean().rename('sHM')
        merged = pd.concat([hm_per_q, hh_per_q], axis=1).dropna()
        if len(merged) > 1:
            r, _ = pearsonr(merged['sHM'], merged['sHH'])
            row['r'] = r
        else:
            row['r'] = np.nan
    else:
        row['r'] = np.nan

    # DJS Y/N
    if model in inst_models:
        sub = inst[(inst['model'] == model) & (inst['op'] == 'exist')]
        model_yn = [norm_yesno(r) for r in sub['response']]
        model_yn = [v for v in model_yn if v is not None]
        if len(model_yn) > 0 and len(human_yn_vals) > 0:
            row['djs_yn'] = js_divergence(model_yn, human_yn_vals, ['yes','no'])
        else:
            row['djs_yn'] = np.nan
    else:
        row['djs_yn'] = np.nan

    # DJS Count
    if model in inst_models:
        sub = inst[(inst['model'] == model) & (inst['op'] == 'count')]
        model_cnt = [norm_count(r) for r in sub['response']]
        model_cnt = [v for v in model_cnt if v is not None]
        if len(model_cnt) > 0 and len(human_cnt_vals) > 0:
            row['djs_count'] = js_divergence(model_cnt, human_cnt_vals, count_bins)
        else:
            row['djs_count'] = np.nan
    else:
        row['djs_count'] = np.nan

    # Abstention w/o (blind)
    if model in blind_models:
        sub = blind[blind['model'] == model]
        row['abst_wo'] = sub['response'].apply(is_abstain).mean() * 100
    else:
        row['abst_wo'] = np.nan

    # Abstention w/ (inst_blind)
    if model in inst_models:
        sub = inst[inst['model'] == model]
        row['abst_w'] = sub['response'].apply(is_abstain).mean() * 100
    else:
        row['abst_w'] = np.nan

    results[model] = row

# ── Print all values (distances from human reference) ──────────────────────────
print(f"\nHuman references: sHH={HUMAN_SHH:.3f}, r=1.0, DJS Y/N={HUMAN_DJS_YN}, DJS Count={HUMAN_DJS_COUNT}")
print(f"\n{'Model':<30} {'|sHM-sHH|':>10} {'|r-1|':>7} {'|DJS YN-ref|':>13} {'|DJS Cnt-ref|':>14} {'Abst w/o':>9} {'Abst w/':>8}")
print("-" * 100)
for (model, size, family, mtype) in ROSTER:
    r = results[model]
    def fmt(v, decimals):
        return f"{v:.{decimals}f}" if not np.isnan(v) else "—"
    d_shm   = abs(r['sHM'] - HUMAN_SHH)         if not np.isnan(r['sHM'])       else float('nan')
    d_r     = abs(r['r'] - HUMAN_R)             if not np.isnan(r['r'])         else float('nan')
    d_yn    = abs(r['djs_yn'] - HUMAN_DJS_YN)   if not np.isnan(r['djs_yn'])    else float('nan')
    d_cnt   = abs(r['djs_count'] - HUMAN_DJS_COUNT) if not np.isnan(r['djs_count']) else float('nan')
    print(f"{model:<30} {fmt(d_shm,3):>10} {fmt(d_r,3):>7} {fmt(d_yn,3):>13} {fmt(d_cnt,3):>14} {fmt(r['abst_wo'],1)+'%':>9} {fmt(r['abst_w'],1)+'%':>8}")

# ── LaTeX generation ───────────────────────────────────────────────────────────
FAMILIES = ["InternVL", "Qwen3-VL", "LLaVA-Vicuna", "Qwen3", "Vicuna"]

def bold(val_str):
    return r"\textbf{" + val_str + "}"

def fmt_val(v, decimals, pct=False):
    if np.isnan(v):
        return "—"
    s = f"{v:.{decimals}f}"
    if pct:
        s += r"\%"
    return s

def make_tex_rows():
    lines = []
    first_family = True
    for family in FAMILIES:
        fam_rows = [(m, sz, fam, mt) for (m, sz, fam, mt) in ROSTER if fam == family]
        if not fam_rows:
            continue

        if not first_family:
            lines.append(r"    \midrule")
        first_family = False

        # Collect metrics for this family; compute distances for dist metrics
        fam_vals = {m: results[m] for (m, sz, fam, mt) in fam_rows}

        def dist_shm(rv):
            v = rv['sHM']
            return abs(v - HUMAN_SHH) if not np.isnan(v) else float('nan')
        def dist_r(rv):
            v = rv['r']
            return abs(v - HUMAN_R) if not np.isnan(v) else float('nan')
        def dist_yn(rv):
            v = rv['djs_yn']
            return abs(v - HUMAN_DJS_YN) if not np.isnan(v) else float('nan')
        def dist_cnt(rv):
            v = rv['djs_count']
            return abs(v - HUMAN_DJS_COUNT) if not np.isnan(v) else float('nan')

        # For distance metrics: smallest = best (min); for abstention: abst_wo max, abst_w min
        dist_metrics = {
            'd_yn':   (dist_yn,  min),
            'd_cnt':  (dist_cnt, min),
            'd_shm':  (dist_shm, min),
            'd_r':    (dist_r,   min),
            'abst_wo': (lambda rv: rv['abst_wo'], max),
            'abst_w':  (lambda rv: rv['abst_w'],  min),
        }
        best_vals = {}
        for key, (fn, bfn) in dist_metrics.items():
            vals = [fn(fam_vals[m]) for m in fam_vals if not np.isnan(fn(fam_vals[m]))]
            best_vals[key] = bfn(vals) if vals else None

        # Emit rows, grouping VLM rows first, then LM, then SA, then Think
        type_order = ['VLM', 'LM', 'SA', 'Think']
        for ttype in type_order:
            type_rows = [(m, sz, fam, mt) for (m, sz, fam, mt) in fam_rows if mt == ttype]
            if not type_rows:
                continue
            for i, (model, size, fam, mtype) in enumerate(type_rows):
                rv = results[model]
                cells = []
                # Distance columns
                for key, (fn, _) in [
                    ('d_yn',  (dist_yn,  min)),
                    ('d_cnt', (dist_cnt, min)),
                    ('d_shm', (dist_shm, min)),
                    ('d_r',   (dist_r,   min)),
                ]:
                    v = fn(rv)
                    s = fmt_val(v, 3, False)
                    if not np.isnan(v) and best_vals[key] is not None and abs(v - best_vals[key]) < 1e-9:
                        s = bold(s)
                    cells.append(s)
                # Abstention columns (raw %)
                for key, pct_raw in [('abst_wo', rv['abst_wo']), ('abst_w', rv['abst_w'])]:
                    s = fmt_val(pct_raw, 1, True)
                    if not np.isnan(pct_raw) and best_vals[key] is not None and abs(pct_raw - best_vals[key]) < 1e-9:
                        s = bold(s)
                    cells.append(s)

                # Family cell: multi-row for first row of family
                if i == 0 and ttype == 'VLM':
                    fam_count = len(fam_rows)
                    fam_cell = r"\multirow{" + str(fam_count) + r"}{*}{" + family + "}"
                else:
                    fam_cell = ""

                row_str = f"    {fam_cell} & {size} & {mtype} & " + " & ".join(cells) + r" \\"
                lines.append(row_str)

    return lines

rows = make_tex_rows()

tex = r"""\begin{table*}[ht]
\small
\centering
\caption{Distance from human reference across model families (instruction-blind condition).
$|\text{DJS}\,\text{Y/N} - \mu_H|$ and $|\text{DJS\,Count} - \mu_H|$: absolute deviation of model Jensen-Shannon divergence from the human LOO median ($\mu_H^{\text{Y/N}}=0.271$, $\mu_H^{\text{Cnt}}=0.524$); smaller = closer to human distribution.
$|\sHM - \sHH|$: absolute deviation of mean model--human SBERT from the pooled human--human SBERT baseline; smaller = closer to human agreement level.
$|r - 1|$: deviation of the per-question Pearson correlation from perfect alignment; smaller = better.
Abst\,w/o\% and Abst\,w/\%: raw abstention rates in blind and instruction-blind conditions.
Smallest distance per metric within each family is \textbf{bolded}; for abstention, highest w/o and lowest w/ are bolded.}
\label{tab:scale_alignment}
\begin{tabular}{lllllllll}
\toprule
Family & Size & Type & $|\text{DJS\,Y/N} - \mu_H|$ & $|\text{DJS\,Cnt} - \mu_H|$ & $|\sHM - \sHH|$ & $|r - 1|$ & Abst\,w/o\% & Abst\,w/\% \\
\midrule
"""
tex += "\n".join(rows)
tex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""

with open(OUT_PATH, "w") as f:
    f.write(tex)

print(f"\nDone. Written to {OUT_PATH}")
