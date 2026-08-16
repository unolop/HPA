"""
Compute image-condition ablation table for Qwen3-VL-8B, LLaVA-Mistral-7B, InternVL-8B.
Metrics: Acc, SBERT, Abstention%, Response-change% vs blank.
Shows blind and inst_blind side-by-side per image condition.
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

IMAGE_CONDITIONS = [
    ('Black', 'vqa_1k_control_blind.jsonl',       'vqa_1k_control_inst_blind.jsonl'),
    ('Gray',  'vqa_1k_control_blind_gray.jsonl',   'vqa_1k_control_inst_blind_gray.jsonl'),
    ('Noise', 'vqa_1k_control_blind_noise.jsonl',  'vqa_1k_control_inst_blind_noise.jsonl'),
    ('White', 'vqa_1k_control_blind_white.jsonl',  'vqa_1k_control_inst_blind_white.jsonl'),
]

MODELS = [
    {
        'label': 'Qwen3-VL',
        'dir': ROOT / 'evaluation/logits/vlm/pretrained/Qwen3-VL-8B-Instruct',
    },
    {
        'label': 'LLaVA-Mistral',
        'dir': ROOT / 'evaluation/logits/vlm/pretrained/llava-v1.6-mistral-7b-hf',
    },
    {
        'label': 'InternVL-8B',
        'dir': ROOT / 'evaluation/logits/vlm/pretrained/InternVL3_5-8B',
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


def compute_metrics(data: dict[int, dict], qids: list[int], ann: dict, sbert_model,
                    baseline_resps: dict[int, str] | None) -> tuple:
    resps, gts, accs, absts, changes = [], [], [], [], []
    for qid in qids:
        resp = clean_answer(str(data[qid]['generated_answers'].get(VARIANT_KEY, '')))
        gt = ann.get(qid, [])
        resps.append(resp)
        gts.append(gt[0] if gt else '')
        accs.append(vqa_accuracy(resp, gt))
        absts.append(float(is_abstained(resp)))
        if baseline_resps is not None:
            changes.append(float(resp != baseline_resps[qid]))
    scores = compute_sbert_scores(resps, gts, sbert_model)
    chg = np.mean(changes) * 100 if baseline_resps is not None else None
    return np.mean(accs), np.mean(scores), np.mean(absts) * 100, chg, {qid: resps[i] for i, qid in enumerate(qids)}


def main():
    print("Loading annotations...")
    ann = load_annotations()

    print("Loading SBERT model (all-mpnet-base-v2)...")
    sbert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # all_results[model_label][img_cond] = {blind: (acc,sbt,abst,chg), inst: (acc,sbt,abst,chg)}
    all_results: dict[str, dict] = {}

    for mconf in MODELS:
        label = mconf['label']
        mdir = mconf['dir']
        print(f"\n--- {label} ---")

        # Load all files
        blind_data, inst_data = {}, {}
        for cname, blind_file, inst_file in IMAGE_CONDITIONS:
            blind_data[cname] = load_jsonl(mdir / blind_file)
            inst_data[cname]  = load_jsonl(mdir / inst_file)
            print(f"  {cname}: blind={len(blind_data[cname])} inst={len(inst_data[cname])}")

        # Common question IDs across all conditions
        common = set(blind_data['Black'].keys())
        for d in list(blind_data.values()) + list(inst_data.values()):
            common &= set(d.keys())
        common &= set(ann.keys())
        qids = sorted(common)
        print(f"  Common: {len(qids)}")

        # Black baselines for change rate
        blind_black_resps = {
            qid: clean_answer(str(blind_data['Black'][qid]['generated_answers'].get(VARIANT_KEY, '')))
            for qid in qids
        }
        inst_black_resps = {
            qid: clean_answer(str(inst_data['Black'][qid]['generated_answers'].get(VARIANT_KEY, '')))
            for qid in qids
        }

        all_results[label] = {}
        for cname, _, _ in IMAGE_CONDITIONS:
            baseline_b = None if cname == 'Black' else blind_black_resps
            baseline_i = None if cname == 'Black' else inst_black_resps
            acc_b, sbt_b, abst_b, chg_b, _ = compute_metrics(blind_data[cname], qids, ann, sbert, baseline_b)
            acc_i, sbt_i, abst_i, chg_i, _ = compute_metrics(inst_data[cname],  qids, ann, sbert, baseline_i)
            all_results[label][cname] = {
                'blind': (acc_b, sbt_b, abst_b, chg_b),
                'inst':  (acc_i, sbt_i, abst_i, chg_i),
            }
            print(f"  {cname} blind: Acc={acc_b:.3f} SBERT={sbt_b:.3f} Abst={abst_b:.1f}% Chg={chg_b}")
            print(f"  {cname} inst:  Acc={acc_i:.3f} SBERT={sbt_i:.3f} Abst={abst_i:.1f}% Chg={chg_i}")

    # ── LaTeX table ──────────────────────────────────────────────────────────
    print("\n% ===== UPDATED LATEX TABLE =====")
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\begin{tabular}{ll cccc cccc}')
    lines.append(r'\toprule')
    lines.append(r' & & \multicolumn{4}{c}{\textbf{Blind}} & \multicolumn{4}{c}{\textbf{Blind+Instruction}} \\')
    lines.append(r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}')
    lines.append(r'\textbf{Model} & \textbf{Image} & Acc. & SBERT & Abst. & Chg. & Acc. & SBERT & Abst. & Chg. \\')
    lines.append(r'\midrule')

    for mconf in MODELS:
        label = mconf['label']
        res = all_results[label]

        # Find best SBERT per condition for bold/underline (across blind and inst separately)
        blind_sbts = [res[c]['blind'][1] for c, _, _ in IMAGE_CONDITIONS]
        inst_sbts  = [res[c]['inst'][1]  for c, _, _ in IMAGE_CONDITIONS]
        b_best = int(np.argmax(blind_sbts))
        b_2nd  = sorted(range(4), key=lambda i: blind_sbts[i])[-2]
        i_best = int(np.argmax(inst_sbts))
        i_2nd  = sorted(range(4), key=lambda i: inst_sbts[i])[-2]

        lines.append(rf'\multirow{{4}}{{*}}{{{label}}}')
        for idx, (cname, _, _) in enumerate(IMAGE_CONDITIONS):
            acc_b, sbt_b, abst_b, chg_b = res[cname]['blind']
            acc_i, sbt_i, abst_i, chg_i = res[cname]['inst']

            def fmt_sbt(v, best, second):
                s = f'{v:.3f}'
                if best: return rf'\textbf{{{s}}}'
                if second: return rf'\underline{{{s}}}'
                return s

            chg_b_s = '---' if chg_b is None else f'{chg_b:.1f}\\%'
            chg_i_s = '---' if chg_i is None else f'{chg_i:.1f}\\%'

            sbt_b_s = fmt_sbt(sbt_b, idx == b_best, idx == b_2nd)
            sbt_i_s = fmt_sbt(sbt_i, idx == i_best, idx == i_2nd)

            lines.append(
                f' & {cname}'
                f' & {acc_b:.3f} & {sbt_b_s} & {abst_b:.1f}\\% & {chg_b_s}'
                f' & {acc_i:.3f} & {sbt_i_s} & {abst_i:.1f}\\% & {chg_i_s} \\\\'
            )
        lines.append(r'\midrule')

    lines.append(r'\end{tabular}')
    lines.append(
        r'\caption{Image-condition ablation (original question variant) under blind and blind+instruction conditions. '
        r'Chg.\ = response change rate relative to the black image baseline of the same condition; ``---'' for the baseline itself. '
        r'\textbf{Bold}: best SBERT within model and condition; \underline{underline}: second best.}'
    )
    lines.append(r'\label{tab:image_ablation_main}')
    lines.append(r'\end{table*}')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
