"""
Compute image-condition ablation table for Qwen3-VL-8B, LLaVA-Mistral-7B, InternVL-8B.
Metrics: Acc, SBERT, Abstention%, Response-change% vs blank.
Replaces chrF with response change rate.
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
from analysis.utils.agreement import _normalize
from analysis.utils.load_session import clean_answer

ANNOT_PATH = ROOT / 'dataset/vqa/v2_mscoco_val2014_annotations.json'
VARIANT_KEY = 'question'

MODELS = [
    {
        'label': 'Qwen3-VL',
        'dir': ROOT / 'evaluation/logits/vlm/pretrained/Qwen3-VL-8B-Instruct',
        'conditions': [
            ('Black', 'vqa_1k_control_blind.jsonl'),
            ('Gray',  'vqa_1k_control_blind_gray.jsonl'),
            ('Noise', 'vqa_1k_control_blind_noise.jsonl'),
            ('White', 'vqa_1k_control_blind_white.jsonl'),
        ],
    },
    {
        'label': 'LLaVA-Mistral',
        'dir': ROOT / 'evaluation/logits/vlm/pretrained/llava-v1.6-mistral-7b-hf',
        'conditions': [
            ('Black', 'vqa_1k_control_blind.jsonl'),
            ('Gray',  'vqa_1k_control_blind_gray.jsonl'),
            ('Noise', 'vqa_1k_control_blind_noise.jsonl'),
            ('White', 'vqa_1k_control_blind_white.jsonl'),
        ],
    },
    {
        'label': 'InternVL-8B',
        'dir': ROOT / 'evaluation/logits/vlm/pretrained/InternVL3_5-8B',
        'conditions': [
            ('Black', 'vqa_1k_control_blind.jsonl'),
            ('Gray',  'vqa_1k_control_blind_gray.jsonl'),
            ('Noise', 'vqa_1k_control_blind_noise.jsonl'),
            ('White', 'vqa_1k_control_blind_white.jsonl'),
        ],
    },
]


def load_jsonl(path: Path) -> dict[int, dict]:
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            data[int(ex['question_id'])] = ex
    return data


def load_annotations() -> dict[int, list[str]]:
    with open(ANNOT_PATH) as f:
        raw = json.load(f)
    ann = {}
    for item in raw.get('annotations', raw):
        qid = int(item['question_id'])
        ann[qid] = [a['answer'] for a in item.get('answers', [])]
    return ann


def is_abstained(resp: str) -> bool:
    return classify(resp, None) in ('hard_abstained', 'soft_abstained')


def vqa_accuracy(pred: str, gt_answers: list[str]) -> float:
    if not gt_answers:
        return 0.0
    pred_n = _normalize(pred)
    count = sum(1 for a in gt_answers if _normalize(a) == pred_n)
    return min(count / 3.0, 1.0)


def compute_sbert_scores(resps: list[str], gts: list[str], model) -> list[float]:
    all_texts = resps + gts
    embs = model.encode(all_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    r_embs = embs[:len(resps)]
    g_embs = embs[len(resps):]
    return (r_embs * g_embs).sum(axis=1).tolist()


def bold(val: str, is_best: bool) -> str:
    return rf'\textbf{{{val}}}' if is_best else val


def underline(val: str, is_second: bool) -> str:
    return rf'\underline{{{val}}}' if is_second else val


def main():
    print("Loading annotations...")
    ann = load_annotations()

    print("Loading SBERT model (all-mpnet-base-v2)...")
    sbert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    all_results = []  # list of (model_label, condition, acc, sbert, abst, chg)

    for mconf in MODELS:
        label = mconf['label']
        mdir = mconf['dir']
        conditions = mconf['conditions']

        print(f"\n--- {label} ---")
        cond_data: dict[str, dict[int, dict]] = {}
        for cname, fname in conditions:
            path = mdir / fname
            cond_data[cname] = load_jsonl(path)
            print(f"  {cname}: {len(cond_data[cname])} questions")

        common = set(cond_data['Black'].keys())
        for c in cond_data.values():
            common &= set(c.keys())
        common &= set(ann.keys())
        qids = sorted(common)
        print(f"  Common: {len(qids)}")

        blank_resps = {
            qid: clean_answer(str(cond_data['Black'][qid]['generated_answers'].get(VARIANT_KEY, '')))
            for qid in qids
        }

        for cname, _ in conditions:
            data = cond_data[cname]
            resps, gts = [], []
            accs, absts, changes = [], [], []

            for qid in qids:
                resp = clean_answer(str(data[qid]['generated_answers'].get(VARIANT_KEY, '')))
                gt = ann.get(qid, [])
                resps.append(resp)
                gts.append(gt[0] if gt else '')
                accs.append(vqa_accuracy(resp, gt))
                absts.append(float(is_abstained(resp)))
                changes.append(float(resp != blank_resps[qid]))

            scores = compute_sbert_scores(resps, gts, sbert)
            row = (label, cname, np.mean(accs), np.mean(scores), np.mean(absts)*100, np.mean(changes)*100)
            all_results.append(row)
            print(f"  {cname}: Acc={row[2]:.3f} SBERT={row[3]:.3f} Abst={row[4]:.1f}% Chg={row[5]:.1f}%")

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Model':<16} {'Cond':<8} {'Acc':>6} {'SBERT':>6} {'Abst%':>6} {'Chg%':>6}")
    print("-"*50)
    for r in all_results:
        print(f"{r[0]:<16} {r[1]:<8} {r[2]:6.3f} {r[3]:6.3f} {r[4]:6.1f} {r[5]:6.1f}")

    # Generate LaTeX table
    print("\n% ===== UPDATED LATEX TABLE =====")
    print(r"""\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Image} & \textbf{Acc.} & \textbf{SBERT} & \textbf{Abst.} & \textbf{Chg.} \\
\midrule""")

    for mconf in MODELS:
        label = mconf['label']
        rows = [r for r in all_results if r[0] == label]
        # find best/second by SBERT
        sbts = [r[3] for r in rows]
        best_idx = int(np.argmax(sbts))
        second_idx = sorted(range(len(sbts)), key=lambda i: sbts[i])[-2]

        print(rf'\multirow{{4}}{{*}}{{{label}}}')
        for i, (_, cname, acc, sbt, abst, chg) in enumerate(rows):
            acc_s = f'{acc:.3f}'
            sbt_s = f'{sbt:.3f}'
            abst_s = f'{abst:.1f}\\%'
            chg_s = f'---' if cname == 'Black' else f'{chg:.1f}\\%'
            if i == best_idx:
                sbt_s = rf'\textbf{{{sbt_s}}}'
                acc_s = rf'\textbf{{{acc_s}}}'
            elif i == second_idx:
                sbt_s = rf'\underline{{{sbt_s}}}'
                acc_s = rf'\underline{{{acc_s}}}'
            print(f' & {cname} & {acc_s} & {sbt_s} & {abst_s} & {chg_s} \\\\')
        print(r'\midrule')

    print(r"""\end{tabular}
\caption{Image-condition ablation (original question variant). Chg.\ = response
change rate relative to the black image baseline; ``---'' for the baseline itself.
\textbf{Bold}: best SBERT/Acc within model; \underline{underline}: second best.}
\label{tab:image_ablation_main}
\end{table}""")


if __name__ == '__main__':
    main()
