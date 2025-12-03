# Directory Reorganization Plan

## Current Issues

1. **Root directory clutter**: Training/evaluation scripts mixed with config files
2. **Mixed concerns**: Notebooks, scripts, and outputs in same directories
3. **Duplicate files**: `pipeline.sh` exists in both `analysis/` and `scripts/`
4. **Output directories scattered**: Analysis outputs in `analysis/`, figures in `figures/`, results in `results/`
5. **Cache files not ignored**: `__pycache__` directories tracked in git
6. **EDA files mixed**: Notebooks, plots, and data files together

## Proposed New Structure

```
HPA/
├── src/                          # All Python source code
│   ├── training/
│   │   ├── trainer.py           # From root
│   │   ├── train_supervised.py  # From root (3_train_soft_supervised.py)
│   │   └── ablations.py         # From root
│   ├── evaluation/
│   │   ├── evaluate.py          # From root (4_evaluate.py)
│   │   └── inference.py         # From root
│   ├── analysis/
│   │   ├── analyze_human_model.py     # From analysis/5_analyze_human_model.py
│   │   ├── analyze_calibration.py     # From analysis/6_analyze_calibration.py
│   │   └── visualize_results.py       # From analysis/7_visualize_results.py
│   ├── preprocessing/
│   │   ├── preprocess.py        # From eval/preprocess.py
│   │   ├── processor.py         # From eval/processor.py
│   │   ├── scoring.py           # From eval/scoring.py
│   │   └── util.py              # From eval/util.py
│   └── config.py                # From root
│
├── scripts/                      # All shell scripts
│   ├── pipeline.sh              # Keep the comprehensive one from scripts/
│   ├── analysis_pipeline.sh     # Rename from analysis/pipeline.sh
│   ├── run_inference.sh
│   ├── run_inference_llm.sh
│   ├── run_evalscope.sh
│   └── sft_swift.sh
│
├── notebooks/                    # All Jupyter notebooks
│   ├── eda/
│   │   ├── pilot_analysis.ipynb      # From eda/
│   │   ├── pilot_results.ipynb       # From eda/
│   │   └── sample_questions.ipynb    # From eda/
│   ├── eval/
│   │   ├── api_models.ipynb          # From eval/
│   │   ├── mmstar.ipynb              # From eval/
│   │   ├── progress.ipynb            # From eval/
│   │   └── spubench.ipynb            # From eval/
│   └── humans/
│       └── eda.ipynb                  # From results/humans/
│
├── data/                         # Input data (keep as is)
│   ├── pilot/
│   ├── s1_mmstar/
│   ├── s1_vqa_1k/
│   └── questions/                # From eda/questions/
│       └── s1.csv
│
├── outputs/                      # All outputs and results
│   ├── analysis/                 # Analysis results
│   │   └── calibration_InternVL3_5-8B_mmstar/
│   ├── figures/                  # Visualizations
│   │   ├── InternVL3_5-8B_mmstar/
│   │   └── InternVL3_5-8B_vqa_1k/
│   ├── eda_plots/                # From eda/plots/
│   ├── results/                  # Model and human results
│   │   ├── humans/
│   │   ├── processed/
│   │   └── swift/
│   └── temp/                     # Temporary outputs (git-ignored)
│
├── models/                       # Model checkpoints (keep as is)
│
├── .gitignore                    # Updated
├── README.md
└── REORGANIZATION_PLAN.md       # This file

```

## Changes Summary

### Moves
1. **Root Python files** → `src/training/` and `src/evaluation/`
2. **analysis/*.py** → `src/analysis/`
3. **eval/*.py** → `src/preprocessing/`
4. **All notebooks** → `notebooks/{eda,eval,humans}/`
5. **eda/plots/** → `outputs/eda_plots/`
6. **eda/questions/** → `data/questions/`
7. **results/** → `outputs/results/`
8. **figures/** → `outputs/figures/`
9. **analysis/calibration_*/** → `outputs/analysis/`

### Removals
- `analysis/__pycache__/` (delete)
- Duplicate `analysis/pipeline.sh` (move to `scripts/analysis_pipeline.sh`)

### .gitignore Updates
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.ipynb_checkpoints/
*.egg-info/

# Outputs (keep some results tracked, but ignore temp)
outputs/temp/
*.log

# Data
*.pth
checkpoint/
wandb/
.env
*.env

# Model directories
models/lxmert/
models/unilm/
data/vqav2_val.json
```

## Benefits

1. **Clear separation of concerns**
   - Source code in `src/`
   - Scripts in `scripts/`
   - Notebooks in `notebooks/`
   - Outputs in `outputs/`

2. **Easier navigation**
   - Related files grouped together
   - Purpose-based directory structure

3. **Better git hygiene**
   - Cache files ignored
   - Temp outputs ignored
   - Only source code and important results tracked

4. **Scalability**
   - Easy to add new analyses, models, or experiments
   - Clear where new files should go

## Migration Steps

1. Create new directory structure
2. Copy files to new locations (preserving git history where possible)
3. Update import paths in Python files
4. Update paths in shell scripts
5. Update .gitignore
6. Test key scripts to ensure they work
7. Remove old directories
8. Commit changes
