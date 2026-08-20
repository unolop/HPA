#!/usr/bin/env python3
"""
Compute SBERT similarity statistics for 6 "flat" (collapsed-variant) questions
across all model inference files.
"""

import sys
import json
import os
import re
import glob
from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[2]
GLOB_PATTERN = str(BASE_DIR / "evaluation/logits/*/pretrained/*/vqa_1k_control_inst_blind.jsonl")
OUTPUT_CSV = BASE_DIR / "analysis/session2/exports/flat_question_stats.csv"

FLAT_QUESTIONS = {
    144162001: "C=A",
    153283009: "C=A",
    289943009: "C=B",
    31024002:  "C=B",
    436162001: "C=B",
    102872002: "C=B",
}

SBERT_MODEL = "all-mpnet-base-v2"

# ── Helper: extract bare question text from prompt ────────────────────────────

def extract_question_text(prompt: str) -> str:
    """Strip prompt wrapper; return just the question text."""
    # Look for text between 'Question: ' and ' Answer the question'
    m = re.search(r"Question:\s*(.+?)\s*Answer the question", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: everything after 'Question: ' on the first line
    m = re.search(r"Question:\s*(.+)", prompt)
    if m:
        return m.group(1).strip()
    return prompt.strip()

# ── Collect all JSONL paths, deduplicate by (model_name, model_type) ──────────

def collect_files(pattern: str) -> dict:
    """
    Returns dict: (model_name, model_type) -> Path of most-recent file.
    model_type derived from the first path segment after logits/.
    """
    all_paths = glob.glob(pattern)
    # normalise model_type: collapse backbone_think_tmp* -> backbone
    best = {}
    for path_str in all_paths:
        p = Path(path_str)
        # path structure: .../logits/<type_dir>/pretrained/<model_name>/...
        parts = p.parts
        logits_idx = None
        for i, part in enumerate(parts):
            if part == "logits":
                logits_idx = i
                break
        if logits_idx is None:
            continue
        type_dir = parts[logits_idx + 1]   # e.g. vlm, lm_decoder, backbone_think_tmp_4b
        model_name = parts[logits_idx + 3] # e.g. llava-1.5-7b-hf

        # Normalise model_type
        if type_dir.startswith("backbone"):
            model_type = "backbone"
        elif type_dir == "lm_decoder":
            model_type = "lm_decoder"
        elif type_dir == "vlm":
            model_type = "vlm"
        else:
            model_type = type_dir  # ablation etc.

        key = (model_name, model_type)
        mtime = os.path.getmtime(path_str)
        if key not in best or mtime > best[key][1]:
            best[key] = (Path(path_str), mtime)

    return {k: v[0] for k, v in best.items()}

# ── Load flat question entries from one file ──────────────────────────────────

def load_flat_entries(filepath: Path, target_ids: set) -> dict:
    """Returns dict: question_id -> entry dict."""
    result = {}
    with open(filepath) as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["question_id"]
            if qid in target_ids:
                result[qid] = entry
    return result

# ── SBERT utilities ───────────────────────────────────────────────────────────

def sbert_sim(model, a: str, b: str) -> float:
    embs = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading SBERT model...", file=sys.stderr)
    sbert = SentenceTransformer(SBERT_MODEL)

    print("Collecting JSONL files...", file=sys.stderr)
    file_map = collect_files(GLOB_PATTERN)
    print(f"  Found {len(file_map)} unique (model, type) combinations.", file=sys.stderr)
    for (name, mtype), path in sorted(file_map.items()):
        print(f"    [{mtype}] {name}: {path}", file=sys.stderr)

    target_ids = set(FLAT_QUESTIONS.keys())

    # Accumulate: qid -> list of (resp_C, resp_B, resp_A)
    # Also store question texts (should be same across models)
    q_texts = {}       # qid -> {"C": str, "B": str, "A": str}
    resp_data = defaultdict(list)   # qid -> [(resp_C, resp_B, resp_A)]
    n_models_per_q = defaultdict(int)

    for (model_name, model_type), fpath in file_map.items():
        entries = load_flat_entries(fpath, target_ids)
        for qid, entry in entries.items():
            # Extract question texts (store once)
            if qid not in q_texts:
                q_texts[qid] = {
                    "C": extract_question_text(entry.get("question", "")),
                    "B": extract_question_text(entry.get("weaker_object", "")),
                    "A": extract_question_text(entry.get("pronominalized", "")),
                }

            # Responses
            ga = entry.get("generated_answers", {})
            resp_C = str(ga.get("question", "")).strip()
            resp_B = str(ga.get("weaker_object", "")).strip()
            resp_A = str(ga.get("pronominalized", "")).strip()

            resp_data[qid].append((resp_C, resp_B, resp_A, model_name, model_type))
            n_models_per_q[qid] += 1

    # ── Compute statistics ──────────────────────────────────────────────────

    rows = []
    print("\n" + "="*70, file=sys.stderr)
    print("FLAT QUESTION STATISTICS", file=sys.stderr)
    print("="*70, file=sys.stderr)

    for qid in sorted(FLAT_QUESTIONS.keys()):
        collapse_type = FLAT_QUESTIONS[qid]
        texts = q_texts.get(qid, {"C": "?", "B": "?", "A": "?"})
        responses = resp_data.get(qid, [])
        n = len(responses)

        # Question-text SBERT similarities
        q_sbert_CB = sbert_sim(sbert, texts["C"], texts["B"])
        q_sbert_CA = sbert_sim(sbert, texts["C"], texts["A"])
        q_sbert_BA = sbert_sim(sbert, texts["B"], texts["A"])

        # Response SBERT similarities (per model, then mean)
        cb_sims, ca_sims, ba_sims = [], [], []
        for (resp_C, resp_B, resp_A, mn, mt) in responses:
            # Skip empty responses
            c = resp_C if resp_C else "N/A"
            b = resp_B if resp_B else "N/A"
            a = resp_A if resp_A else "N/A"
            cb_sims.append(sbert_sim(sbert, c, b))
            ca_sims.append(sbert_sim(sbert, c, a))
            ba_sims.append(sbert_sim(sbert, b, a))

        resp_CB_mean = float(np.mean(cb_sims)) if cb_sims else float("nan")
        resp_CA_mean = float(np.mean(ca_sims)) if ca_sims else float("nan")
        resp_BA_mean = float(np.mean(ba_sims)) if ba_sims else float("nan")

        rows.append({
            "question_id": qid,
            "question_text_C": texts["C"],
            "question_text_B": texts["B"],
            "question_text_A": texts["A"],
            "collapse_type": collapse_type,
            "n_models": n,
            "q_sbert_CB": round(q_sbert_CB, 4),
            "q_sbert_CA": round(q_sbert_CA, 4),
            "q_sbert_BA": round(q_sbert_BA, 4),
            "resp_sbert_CB_mean": round(resp_CB_mean, 4),
            "resp_sbert_CA_mean": round(resp_CA_mean, 4),
            "resp_sbert_BA_mean": round(resp_BA_mean, 4),
        })

        # Print summary
        print(f"\nQID {qid} [{collapse_type}] (n={n} models)", file=sys.stderr)
        print(f"  C: {texts['C']}", file=sys.stderr)
        print(f"  B: {texts['B']}", file=sys.stderr)
        print(f"  A: {texts['A']}", file=sys.stderr)
        print(f"  Q-text SBERT:  CB={q_sbert_CB:.4f}  CA={q_sbert_CA:.4f}  BA={q_sbert_BA:.4f}", file=sys.stderr)
        print(f"  Resp SBERT μ:  CB={resp_CB_mean:.4f}  CA={resp_CA_mean:.4f}  BA={resp_BA_mean:.4f}", file=sys.stderr)

        # Per-model breakdown
        print(f"  Per-model responses (C | B | A):", file=sys.stderr)
        for (resp_C, resp_B, resp_A, mn, mt) in responses:
            print(f"    [{mt}] {mn:<40s}  {resp_C!r:20s} | {resp_B!r:20s} | {resp_A!r}", file=sys.stderr)

    # ── Save CSV ────────────────────────────────────────────────────────────

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id", "question_text_C", "question_text_B", "question_text_A",
        "collapse_type", "n_models",
        "q_sbert_CB", "q_sbert_CA", "q_sbert_BA",
        "resp_sbert_CB_mean", "resp_sbert_CA_mean", "resp_sbert_BA_mean",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Saved to {OUTPUT_CSV}", file=sys.stderr)
    print(f"Rows: {len(rows)}", file=sys.stderr)

if __name__ == "__main__":
    main()
