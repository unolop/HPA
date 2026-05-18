import os

ROOT_DIR = os.environ.get("HPA_ROOT", "/home/david/Desktop/yuna")
HPA_DIR = f"{ROOT_DIR}/HPA"

BLANK_IMAGE = f"{HPA_DIR}/dataset/blank_224.png"
GRAY_IMAGE  = f"{HPA_DIR}/dataset/gray_224.png"
NOISE_IMAGE = f"{HPA_DIR}/dataset/noise_224.png"
WHITE_IMAGE = f"{HPA_DIR}/dataset/white_224.png"
SPUBENCH_ANNOT = f"{HPA_DIR}/dataset/annotation.json"
VQA_5K_QIDS = f"{HPA_DIR}/dataset/s1_qids.json"

VQA_IMAGE_DIR = f"{ROOT_DIR}/data/val2014"
VQA_QUESTIONS = f"{ROOT_DIR}/data/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_ANNOT = f"{ROOT_DIR}/data/v2_mscoco_val2014_annotations.json"

VQA_1K = f"{HPA_DIR}/dataset/vqa/vqav2_1k_val.json"
VQA_1K_CONTROL = f"{HPA_DIR}/dataset/vqa/vqa1k_control.jsonl"

LOGITS_DIR = f"{HPA_DIR}/evaluation/logits"
SCORED_DIR = f"{HPA_DIR}/evaluation/scored"
