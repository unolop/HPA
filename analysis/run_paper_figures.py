"""
Master figure export orchestrator.

Calls all modular export scripts in sequence and prints a summary of outputs.
Each script is self-contained: it loads its own data and writes to its own
figures/ subfolder following the naming convention in README.md.

Run from the repo root:
  conda run -n zero python analysis/run_paper_figures.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
FIGURES = ROOT / 'figures'

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true',
                    help='Pass --overwrite to figure exporters before regenerating plots.')
args = parser.parse_args()


def run(script: str, *args: str, label: str = ''):
    """Run a script and stream its output; abort the orchestrator on failure."""
    cmd = ['conda', 'run', '-n', 'zero', 'python', str(FIGURES / script)] + list(args)
    tag = label or script
    print(f'\n{"─"*70}')
    print(f'  {tag}')
    print('─' * 70)
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f'\nERROR: {script} exited with code {result.returncode}')
        sys.exit(result.returncode)


# ── Pair cache (must run first — all agreement figures depend on it) ──────────
run('../analysis/build_pair_cache.py',
    label='Build / update pair_cache (new humans + new models)')

# ── Agreement / alignment figures ────────────────────────────────────────────
run('hh_hm/variants_family.py', *(('--overwrite',) if args.overwrite else ()), '--include_yesno',
    label='Family-level HM agreement across control variants (+ yes/no)')

# ── Accuracy figures ──────────────────────────────────────────────────────────
run('accuracy_variants.py', *(('--overwrite',) if args.overwrite else ()),
    label='Accuracy degradation across variants C→B→A')


# ── Scale figures ─────────────────────────────────────────────────────────────
for metric in ('agreement', 'accuracy'):
    for agg in ('by_models', 'by_groups', 'by_group'):
        run('scale_plots.py', '--metric', metric, '--agg', agg,
            *(('--overwrite',) if args.overwrite else ()),
            label=f'Scale plots: {metric} × {agg}')

# ── Instruction effect ────────────────────────────────────────────────────────
run('instruction_effect.py', *(('--overwrite',) if args.overwrite else ()),
    label='Instruction effect: soft/hard abstention + response change rate')

# ── Confidence ────────────────────────────────────────────────────────────────
run('confidence_dist.py', *(('--overwrite',) if args.overwrite else ()),
    label='Confidence analysis: condition shift + distributions + control ladders')

# ── Answer distributions ──────────────────────────────────────────────────────
run('answer_dist.py', *(('--overwrite',) if args.overwrite else ()),
    label='Answer distributions: yes/no and number questions')

# ── Model-model coherence (supplementary) ────────────────────────────────────
run('model_coherence.py', *(('--overwrite',) if args.overwrite else ()),
    label='Model-model within-group coherence across variants (supplementary)')

# ── Entity analysis ───────────────────────────────────────────────────────────
run('entity_analysis.py', *(('--overwrite',) if args.overwrite else ()),
    label='Entity analysis: distribution, Pearson r, SBERT, degradation, '
          'instruction sensitivity + combined alignment figure')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '═' * 70)
print('All figures generated. Outputs in:')
print('  figures/')
print()
print('Key paper figures:')
print('  hh_hm/variants_family/  inst_blind_sbert_family_vABC_q113_yesno.png')
print('  instruction_effect/  soft_abstention_vC_*.png')
print('  instruction_effect/  response_change_vC_*.png')
print('  entity_analysis/     fig_hm_alignment*.png')
print()
print('Note: abstention_rates.png / abstention_collapse.png are generated')
print('  by analysis/session2/08_char_abstention.ipynb (run manually).')
print('═' * 70)
