
import json, argparse
from collections import defaultdict, Counter
from typing import Dict, List
from preprocess import build_human_distribution, merge_supports, ensure_distribution_from_answer, canonicalize
from metrics import js_div, kl_div, brier, top1_match, expected_calibration_error
from bootstrap import bootstrap_ci
from embedding import load_encoder, embed_texts, cluster_answers, canonical_from_clusters
from agreement import vqa_score

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def summarize(xs):
    mean, lo, hi = bootstrap_ci(xs)
    return {"mean": mean, "ci95": [lo, hi], "n": len(xs)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--synonyms", default=None)
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--binom_bins", type=int, default=10)
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--cluster_thresh", type=float, default=0.8)
    ap.add_argument("--use_semantic_clustering", action="store_true")
    args = ap.parse_args()

    synonyms = None
    if args.synonyms:
        with open(args.synonyms, "r", encoding="utf-8") as f:
            synonyms = json.load(f)

    encoder = None
    if args.use_semantic_clustering:
        encoder = load_encoder(args.embed_model)

    H = {}
    raw_answers = {}
    for ex in load_jsonl(args.human):
        qid = ex["qid"]
        ans = [a for a in ex.get("answers", [])]
        raw_answers[qid] = ans[:]
        if args.use_semantic_clustering and len(ans) > 1:
            vecs = embed_texts(encoder, ans)
            assign, cmap = cluster_answers(ans, vecs, thresh=args.cluster_thresh)
            can_map = canonical_from_clusters(ans, assign)
            ans = [can_map[a] for a in ans]
        H[qid] = build_human_distribution(ans, synonyms=synonyms)

    M = {}
    top_prob = {}
    top_match = {}
    vqa_scores = []
    for ex in load_jsonl(args.model):
        qid = ex["qid"]
        if "probs" in ex and ex["probs"]:
            probs = { canonicalize(k, synonyms): float(v) for k,v in ex["probs"].items() }
            Z = sum(probs.values()) or 1.0
            probs = {k: v/Z for k,v in probs.items()}
            M[qid] = probs
            top1 = max(probs.items(), key=lambda kv: kv[1])[0]
            if qid in raw_answers:
                vqa_scores.append(vqa_score(top1, raw_answers[qid]))
            top_prob[qid] = max(probs.values())
            human_majority = max(H.get(qid, {"<na>":1.0}).items(), key=lambda kv: kv[1])[0]
            top_match[qid] = 1.0 if top1 == human_majority else 0.0
        elif "answer" in ex:
            M[qid] = ensure_distribution_from_answer(ex["answer"])
            if qid in raw_answers:
                vqa_scores.append(vqa_score(ex["answer"], raw_answers[qid]))
            top_prob[qid] = 1.0
            human_majority = max(H.get(qid, {"<na>":1.0}).items(), key=lambda kv: kv[1])[0]
            top_match[qid] = 1.0 if list(M[qid].keys())[0] == human_majority else 0.0

    qids = sorted(set(H.keys()) & set(M.keys()))
    js_list, kl_list, br_list, t1_list = [], [], [], []
    for q in qids:
        supp = merge_supports(H[q], M[q])
        js_list.append(js_div(H[q], M[q], supp))
        kl_list.append(kl_div(H[q], M[q], supp))
        br_list.append(brier(H[q], M[q], supp))
        t1_list.append(top1_match(H[q], M[q]))

    confs = [top_prob[q] for q in qids if q in top_prob]
    corrs = [top_match[q] for q in qids if q in top_match]
    ece = expected_calibration_error(confs, corrs, n_bins=args.binom_bins) if confs else 0.0

    out = {
        "n_questions": len(qids),
        "metrics": {
            "JS": summarize(js_list),
            "KL": summarize(kl_list),
            "Brier": summarize(br_list),
            "Top1": summarize(t1_list),
            "VQA_score_top1": summarize(vqa_scores) if vqa_scores else {"mean":0,"ci95":[0,0],"n":0},
            "ECE": {"value": ece, "bins": args.binom_bins}
        },
        "semantic_clustering": {
            "enabled": bool(args.use_semantic_clustering),
            "threshold": args.cluster_thresh,
            "embed_model": args.embed_model if args.use_semantic_clustering else None
        }
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
