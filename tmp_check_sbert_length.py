from pathlib import Path

from scipy import stats

from helpers import load_human_subset, read_response_exports, load_pair_cache
from config import MODEL_GROUP
from utils.constants import MODEL_FAMILY


ROOT = Path("/home/david/Desktop/yuna/HPA")
participants, common_qids, human_df, _ = load_human_subset(
    ROOT, min_answers=348, translate=False, verbose=False
)


def run(include_yesno=False):
    exports = read_response_exports(ROOT, subset_qids=common_qids, variant="C")
    model_df = exports["model_inst_blind"].copy()
    pair = load_pair_cache(ROOT, include_yesno=include_yesno, verbose=False)
    pair = pair[
        (pair["variant"] == "C")
        & (pair["question_id"].isin(common_qids))
        & (pair["pair_type"] == "HM")
    ].copy()
    if "condition_2" in pair.columns:
        pair = pair[pair["condition_2"] == "inst_blind"].copy()

    hm = (
        pair.groupby(["question_id", "subject_2"])
        .agg(mean_sbert=("sbert_score", "mean"))
        .reset_index()
        .rename(columns={"subject_2": "model"})
    )
    joined = model_df.merge(hm, on=["question_id", "model"], how="inner")
    joined["acc_gap"] = joined["accuracy"] - joined["human_avg_acc"]
    joined["answer_len"] = joined["response"].fillna("").astype(str).str.split().str.len()

    ms = (
        joined.groupby("model")
        .agg(
            mean_sbert=("mean_sbert", "mean"),
            mean_acc_gap=("acc_gap", "mean"),
            mean_len=("answer_len", "mean"),
        )
        .reset_index()
    )
    ms["model_group"] = ms["model"].map(MODEL_GROUP)
    ms["model_family"] = ms["model"].map(MODEL_FAMILY)

    print(f"\nINCLUDE_YESNO={include_yesno} nq={joined['question_id'].nunique()} models={len(ms)}")
    for a, b, name in [
        ("mean_len", "mean_sbert", "len_vs_sbert"),
        ("mean_len", "mean_acc_gap", "len_vs_gap"),
    ]:
        x = ms[a].to_numpy()
        y = ms[b].to_numpy()
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        print(
            f"{name}: pearson={pr:.4f} p={pp:.6g} spearman={sr:.4f} p={sp:.6g}"
        )
    print(
        ms.sort_values("mean_len")[
            ["model", "model_group", "mean_len", "mean_sbert", "mean_acc_gap"]
        ].to_string(index=False)
    )


run(False)
run(True)
