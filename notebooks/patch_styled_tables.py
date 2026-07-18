"""
Patch all notebooks/*.ipynb to add colored heatmap styling to every table display.

Strategy:
  1. Insert a `show(df)` helper cell after the imports cell in each notebook.
  2. For each code cell that ends with a bare variable or .head() call,
     append show() on the next line (in-place).
  3. For each code cell that calls display(xxx), replace with show(xxx).

Run from repo root:
  python notebooks/patch_styled_tables.py
"""

import json
import re
import copy
from pathlib import Path

NB_DIR = Path(__file__).parent

HELPER_SOURCE = '''\
import pandas as pd
from IPython.display import display as _display

# ── Gradient column hints ─────────────────────────────────────────────────────
_CORR_COLS   = {'r', 'pearson r', 'spearman rho', 'all r', 'original',
                'weaker', 'pronominalized', 'hh sbert', 'hm sbert',
                'mean hh sbert', 'mean hm sbert', 'hh-hm',
                'delta c->a', r'c$\\to$a', 'blind accuracy'}
_PVAL_COLS   = {'p', 'pearson p', 'spearman p', 'p-value'}
_DIV_COLS    = {'js divergence', 'tv distance', "cramer's v", 'chi-square'}
_SIG_COLS    = {'sig.', 'significant', 'pearson sig', 'significant?'}

def show(df, caption=None):
    """Display a DataFrame with auto heatmap styling."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        _display(df)
        return
    s = df.style
    numeric = df.select_dtypes(include='number').columns.tolist()
    for col in numeric:
        key = col.lower().strip()
        vmin, vmax = df[col].min(), df[col].max()
        if vmin == vmax:
            continue
        if key in _CORR_COLS:
            lo = min(vmin, -0.05)
            s = s.background_gradient(subset=[col], cmap='RdYlGn', vmin=lo, vmax=max(vmax, 0.05))
        elif key in _PVAL_COLS:
            s = s.background_gradient(subset=[col], cmap='RdYlGn_r', vmin=0, vmax=0.2)
        elif key in _DIV_COLS:
            s = s.background_gradient(subset=[col], cmap='YlOrRd', vmin=vmin, vmax=vmax)
        elif 'acc' in key or 'mean acc' in key:
            s = s.background_gradient(subset=[col], cmap='Greens', vmin=vmin, vmax=vmax)
        elif 'conf' in key:
            s = s.background_gradient(subset=[col], cmap='Blues', vmin=vmin, vmax=vmax)
        elif 'slope' in key:
            s = s.background_gradient(subset=[col], cmap='PuOr', vmin=vmin, vmax=vmax)
        else:
            s = s.background_gradient(subset=[col], cmap='Blues', vmin=vmin, vmax=vmax)
    # Significance column
    sig_cols = [c for c in df.columns if c.lower().strip() in _SIG_COLS]
    for sc in sig_cols:
        def _color_sig(val):
            if str(val).strip().lower() in ('n.s.', 'false', 'no', ''):
                return 'color: #c62828; font-weight: bold'
            return 'color: #2e7d32; font-weight: bold'
        s = s.applymap(_color_sig, subset=[sc])
    # Rank column — highlight top rows
    if len(df) > 1 and df.index.dtype == int or True:
        pass  # row highlighting done via gradients on value cols
    if caption:
        s = s.set_caption(caption)
    s = s.format(lambda x: f"{x:.3f}" if isinstance(x, float) else x, na_rep="—")
    _display(s)
'''

HELPER_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "id": "styled-show-helper",
    "metadata": {},
    "outputs": [],
    "source": HELPER_SOURCE.splitlines(keepends=True),
}

# Target patterns: bare variable at end of cell or .head() call
BARE_VAR_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)(?:\.head\(\d*\))?\s*$')
# display(xxx) pattern
DISPLAY_RE  = re.compile(r'\bdisplay\((.+?)\)')


def get_last_line(src_lines):
    """Return stripped last non-empty line of a cell."""
    for line in reversed(src_lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            return stripped
    return ''


def patch_notebook(path: Path):
    with open(path) as f:
        nb = json.load(f)

    cells = nb['cells']
    patched = []
    helper_inserted = False

    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            patched.append(cell)
            continue

        src_lines = cell['source']
        src = ''.join(src_lines)

        # Insert helper after the first code cell (imports)
        if not helper_inserted and i > 0:
            h = copy.deepcopy(HELPER_CELL)
            patched.append(h)
            helper_inserted = True

        last_line = get_last_line(src_lines)

        # Case 1: cell ends with bare variable or xxx.head()
        m = BARE_VAR_RE.match(last_line)
        if m:
            var = m.group(1)
            # Append show() call
            new_cell = copy.deepcopy(cell)
            new_src = list(src_lines)
            # Remove trailing newline from last line if needed
            if new_src and new_src[-1].endswith('\n'):
                new_src[-1] = new_src[-1].rstrip('\n')
            new_src.append(f'\nshow({last_line})\n')
            new_cell['source'] = new_src
            patched.append(new_cell)
            continue

        # Case 2: cell contains display(xxx) — add show(xxx) after each
        if 'display(' in src:
            new_src = list(src_lines)
            additions = []
            for line in src_lines:
                stripped = line.strip()
                dm = DISPLAY_RE.match(stripped)
                if dm:
                    arg = dm.group(1).strip()
                    additions.append(f'show({arg})\n')
            if additions:
                if new_src and not new_src[-1].endswith('\n'):
                    new_src[-1] += '\n'
                new_src.extend(additions)
                new_cell = copy.deepcopy(cell)
                new_cell['source'] = new_src
                patched.append(new_cell)
                continue

        patched.append(cell)

    nb['cells'] = patched
    # Fix kernel to python3
    nb['metadata']['kernelspec'] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'  patched: {path.name}')


if __name__ == '__main__':
    for nb_path in sorted(NB_DIR.glob('*.ipynb')):
        if nb_path.name.startswith('_'):
            continue
        print(f'\n{nb_path.name}')
        patch_notebook(nb_path)
    print('\nDone.')
