import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(".")
EXPORTS = ROOT / "analysis/session2/exports"
sys.path.insert(0, str(ROOT))
from figures.helpers import filter_abstained_pairs

pc = pd.read_parquet(EXPORTS / "pair_cache_cleaned.parquet")
pc = filter_abstained_pairs(pc)
hh = pc[pc["pair_type"] == "HH"]
hm = pc[pc["pair_type"] == "HM"]
hh_q = hh.groupby("question_id")["sbert_score"].mean()

MODELS = [
    ("0.6B", "Qwen3-0.6B",  "Qwen3-0.6B (think)"),
    ("1.7B", "Qwen3-1.7B",  "Qwen3-1.7B (think)"),
    ("4B",   "Qwen3-4B",    "Qwen3-4B (think)"),
    ("8B",   "Qwen3-8B",    "Qwen3-8B (think)"),
    ("32B",  "Qwen3-32B",   "Qwen3-32B (think)"),
]

def get_stats(model):
    sub = hm[hm["subject_2"] == model]
    if len(sub) == 0:
        return None, None, None
    hm_q = sub.groupby("question_id")["sbert_score"].mean()
    common = hm_q.index.intersection(hh_q.index)
    x = hm_q[common].values
    y = hh_q[common].values
    rho, _ = spearmanr(x, y)
    r, _   = pearsonr(x, y)
    shm    = sub["sbert_score"].mean()
    return round(rho, 3), round(r, 3), round(shm, 3)

print("Size   nt_rho   t_rho  d_rho |  nt_r    t_r    nt_shm  t_shm")
for size, no_think, think in MODELS:
    rho_nt, r_nt, shm_nt = get_stats(no_think)
    rho_t,  r_t,  shm_t  = get_stats(think)
    delta = round(rho_t - rho_nt, 3) if (rho_t is not None and rho_nt is not None) else "---"
    print(f"{size:<6} {str(rho_nt):>7} {str(rho_t):>7} {str(delta):>7} | {str(r_nt):>6} {str(r_t):>6}  {str(shm_nt):>7} {str(shm_t):>7}")
