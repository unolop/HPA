# Session 1 — ACL Submission (archived)

**Datasets:** VQA v2 (5k subset) + MMstar  
**Models:** Pretrained + finetuned VLMs  
**Human study:** N=24, 641 questions, `inst_blind` condition  

## Contents

| Dir | Contents |
|-----|----------|
| `notebook/` | S1 analysis notebooks (may need path updates to re-run) |
| `figures/` | Figures generated for ACL paper |
| `csv/` | Human study data (`human_vqa.csv`), correlations, quadrant examples |
| `tables/` | LaTeX tables for ACL paper |
| `scripts/` | `make_tier_examples.py` |

## Notebooks

| File | Purpose |
|------|---------|
| `s1_model_pretrained.ipynb` | Pretrained model spurious bias analysis (VQA + MMstar) |
| `s1_model_finetuned.ipynb` | Finetuned model delta analysis |
| `s1_corr_vqa_mmstar.ipynb` | VQA ↔ MMstar correlation analysis |
| `s1_question_type.ipynb` | Question type clustering |

**Note:** S1 notebooks use `analyzer.py` (in `analysis/`). Run from `analysis/notebook/` or adjust `sys.path`.
