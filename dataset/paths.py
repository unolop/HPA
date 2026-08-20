import os
from pathlib import Path

# Repo root: resolved relative to this file (dataset/paths.py → HPA/)
HPA_DIR = str(Path(__file__).resolve().parent.parent)

# External data root (VQA images live outside the repo).
# Override with HPA_ROOT env var if your data is elsewhere.
_external = os.environ.get("HPA_ROOT", str(Path(HPA_DIR).parent))

BLANK_IMAGE = f"{HPA_DIR}/dataset/blank_224.png"
GRAY_IMAGE  = f"{HPA_DIR}/dataset/gray_224.png"
NOISE_IMAGE = f"{HPA_DIR}/dataset/noise_224.png"
WHITE_IMAGE = f"{HPA_DIR}/dataset/white_224.png"
SPUBENCH_ANNOT = f"{HPA_DIR}/dataset/annotation.json"
VQA_5K_QIDS = f"{HPA_DIR}/dataset/s1_qids.json"

VQA_IMAGE_DIR = f"{_external}/data/val2014"
VQA_QUESTIONS = f"{_external}/data/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_ANNOT = f"{_external}/data/v2_mscoco_val2014_annotations.json"

VQA_1K = f"{HPA_DIR}/dataset/vqa/vqav2_1k_val.json"
VQA_1K_CONTROL = f"{HPA_DIR}/dataset/vqa/vqa1k_control.jsonl"

LOGITS_DIR = f"{HPA_DIR}/evaluation/logits"
SCORED_DIR = f"{HPA_DIR}/evaluation/scored"
