"""
Master figure export orchestrator.

Calls all modular export scripts in sequence and prints a summary of outputs.
Each script is self-contained: it loads its own data and writes to its own
figures/ subfolder following the naming convention in README.md.

Run from the repo root:
  conda run -n zero python analysis/run_paper_figures.py

To regenerate a single figure type, run its script directly, e.g.:
  conda run -n zero python analysis/export_agreement_scatter.py
  conda run -n zero python analysis/export_accuracy_scatter.py --agg model_groups
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / 'analysis'


def run(script: str, *args: str, label: str = ''):
    """Run a script and stream its output; abort the orchestrator on failure."""
    cmd = ['conda', 'run', '-n', 'zero', 'python', str(ANALYSIS / script)] + list(args)
    tag = label or script
    print(f'\n{"─"*70}')
    print(f'  {tag}')
    print('─' * 70)
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f'\nERROR: {script} exited with code {result.returncode}')
        sys.exit(result.returncode)


# ── Pair cache (must run first — all agreement figures depend on it) ──────────
run('build_pair_cache.py',
    label='Build / update pair_cache (new humans + new models)')

# ── Agreement figures ─────────────────────────────────────────────────────────
run('export_agreement_scatter.py',
    label='Agreement scatter: SBERT×exact per model + SBERT vs scale (Qwen3)')

run('export_agreement_variants.py',
    label='Agreement by variant: group-level lineplots + heatmaps')

run('export_agreement_variants.py', '--include_yesno',
    label='Agreement by variant (+ yes/no questions)')

run('export_agreement_variants_by_models.py',
    label='Agreement by variant: per-model lineplots')

run('export_agreement_heatmaps.py',
    label='Pairwise agreement heatmaps (all raters + group-mean)')

# ── Accuracy figures ──────────────────────────────────────────────────────────
run('export_accuracy_variants.py',
    label='Accuracy degradation across variants C→B→A')

run('export_accuracy_scatter.py', '--agg', 'model_groups',
    label='Per-question accuracy scatter (aggregated by group)')

run('export_accuracy_scatter.py', '--agg', 'model_family',
    label='Per-question accuracy scatter (aggregated by family)')

run('export_accuracy_quadrant.py',
    label='Per-question quadrant scatter (Jaccard coloured, blind + inst_blind)')

# ── Scale figures ─────────────────────────────────────────────────────────────
for metric in ('agreement', 'accuracy'):
    for agg in ('by_models', 'by_family', 'by_groups'):
        run('export_scale_plots.py', '--metric', metric, '--agg', agg,
            label=f'Scale plots: {metric} × {agg}')

# ── Instruction effect ────────────────────────────────────────────────────────
run('export_instruction_effect.py',
    label='Instruction effect: soft/hard abstention + response change rate')

# ── Confidence ────────────────────────────────────────────────────────────────
run('export_confidence_dist.py',
    label='Confidence analysis: condition shift + distributions + control ladders')

# ── Answer distributions ──────────────────────────────────────────────────────
run('export_answer_dist.py',
    label='Answer distributions: yes/no and number questions')

# ── Entity analysis ───────────────────────────────────────────────────────────
run('export_entity_analysis.py',
    label='Entity analysis: distribution, Pearson r, SBERT, degradation, '
          'instruction sensitivity + combined alignment figure')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '═' * 70)
print('All figures generated. Outputs in:')
print('  latex/AnonymousSubmission/LaTeX/figures/')
print()
print('Key paper figures:')
print('  agreement_scatter/  inst_blind_vC_sbert_exact_models*.png')
print('  agreement_scatter/  inst_blind_vC_sbert_scale_qwen3*.png')
print('  accuracy_scatter/   blind_vC_jaccard_vlm*.png')
print('  accuracy_scatter/   inst_blind_vC_jaccard_vlm_annotated*.png')
print('  agreement_heatmap/  vC_sbert_{all,groups}.png')
print('  instruction_effect/ soft_abstention_vC_*.png')
print('  instruction_effect/ response_change_vC_*.png')
print('  entity_analysis/    fig_hm_alignment*.png')
print()
print('Note: abstention_rates.png / abstention_collapse.png are generated')
print('  by analysis/session2/08_char_abstention.ipynb (run manually).')
print('═' * 70)
