"""
Compute image-condition ablation metrics for InternVL-8B VLM.
Matches the format of the existing Qwen3-VL / LLaVA-Mistral ablation table.
Metrics: Acc, chrF, SBERT, Abstention rate, Response change rate vs blank.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))

from analysis.utils.abstention import classify
from analysis.utils.agreement import _pair_chrf, _normalize
from analysis.utils.load_session import clean_answer
from analysis.utils.vqa import score_answer_types

LOGIT_DIR = ROOT / 'evaluation/logits/vlm/pretrained/InternVL3_5-8B'
ANNOT_PATH = ROOT / 'dataset/vqa/v2_mscoco_val2014_annotations.json'

CONDITIONS = [
    ('Black',  'vqa_1k_control_blind.jsonl'),
    ('Gray',   'vqa_1k_control_blind_gray.jsonl'),
    ('Noise',  'vqa_1k_control_blind_noise.jsonl'),
    ('White',  'vqa_1k_control_blind_white.jsonl'),
]
VARIANT_KEY = 'question'  # variant C = original


def load_jsonl(path: Path) -> dict[int, dict]:
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            qid = int(ex['question_id'])
            data[qid] = ex
    return data


def load_annotations() -> dict[int, list[str]]:
    with open(ANNOT_PATH) as f:
        raw = json.load(f)
    ann = {}
    for item in raw.get('annotations', raw):
        qid = int(item['question_id'])
        answers = [a['answer'] for a in item.get('answers', [])]
        ann[qid] = answers
    return ann


def is_abstained(resp: str) -> bool:
    cls = classify(resp, None)
    return cls in ('hard_abstained', 'soft_abstained')


def vqa_accuracy(pred: str, gt_answers: list[str]) -> float:
    """Standard VQA accuracy: min(#matching / 3, 1)."""
    if not gt_answers:
        return 0.0
    pred_n = _normalize(pred)
    count = sum(1 for a in gt_answers if _normalize(a) == pred_n)
    return min(count / 3.0, 1.0)


def main():
    print("Loading annotations...")
    ann = load_annotations()

    print("Loading SBERT model...")
    sbert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Load all conditions
    condition_data: dict[str, dict[int, dict]] = {}
    for label, fname in CONDITIONS:
        path = LOGIT_DIR / fname
        condition_data[label] = load_jsonl(path)
        print(f"  {label}: {len(condition_data[label])} questions")

    blank_data = condition_data['Black']
    common_qids = set(blank_data.keys())
    for label, _ in CONDITIONS[1:]:
        common_qids &= set(condition_data[label].keys())
    common_qids &= set(ann.keys())
    qids = sorted(common_qids)
    print(f"Common questions: {len(qids)}")

    # Pre-encode blank responses for change rate
    blank_resps = {qid: clean_answer(str(blank_data[qid]['generated_answers'].get(VARIANT_KEY, '')))
                   for qid in qids}

    print("\nComputing metrics per condition...")
    print(f"{'Condition':<10} {'Acc':>6} {'chrF':>6} {'SBERT':>6} {'Abst%':>6} {'Chg%':>6}")
    print("-" * 45)

    rows = []
    for label, _ in CONDITIONS:
        data = condition_data[label]
        accs, chrfs, sbts, absts, changes = [], [], [], [], []

        resps_list = []
        gt_list = []
        for qid in qids:
            ex = data[qid]
            resp = clean_answer(str(ex['generated_answers'].get(VARIANT_KEY, '')))
            gt = ann.get(qid, [])

            accs.append(vqa_accuracy(resp, gt))
            chrfs.append(_pair_chrf(resp, gt[0]) if gt else 0.0)
            absts.append(float(is_abstained(resp)))
            changes.append(float(resp != blank_resps[qid]))

            resps_list.append(resp)
            gt_list.append(gt[0] if gt else '')

        # SBERT: encode all at once
        all_texts = resps_list + gt_list
        embs = sbert.encode(all_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
        resp_embs = embs[:len(resps_list)]
        gt_embs = embs[len(resps_list):]
        cos_scores = (resp_embs * gt_embs).sum(axis=1).tolist()
        sbts = cos_scores

        acc_m = np.mean(accs)
        chrf_m = np.mean(chrfs)
        sbt_m = np.mean(sbts)
        abst_m = np.mean(absts) * 100
        chg_m = np.mean(changes) * 100

        print(f"{label:<10} {acc_m:6.3f} {chrf_m:6.3f} {sbt_m:6.3f} {abst_m:6.1f} {chg_m:6.1f}")
        rows.append((label, acc_m, chrf_m, sbt_m, abst_m, chg_m))

    # Print LaTeX rows
    print("\nLaTeX rows for ablation table:")
    print("% InternVL-8B (with response change rate vs blank)")
    print(r"\midrule")
    print(r"\multirow{4}{*}{InternVL-8B}")
    for i, (label, acc, chrf, sbt, abst, chg) in enumerate(rows):
        prefix = " & " if i > 0 else ""
        star = r"\textbf" if label == max(rows, key=lambda r: r[3])[0] else ""
        print(f" & {label} & {acc:.3f} & {chrf:.3f} & {sbt:.3f} & {abst:.1f}\\% & {chg:.1f}\\% \\\\")


if __name__ == '__main__':
    main()
