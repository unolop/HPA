# Paper Figures — Status & Generation Guide

All figures for `latex/AnonymousSubmission/LaTeX/paper.tex`.
Place final files in `latex/AnonymousSubmission/LaTeX/` OR update `\graphicspath`.

**Format rule:** Use **PDF for plots/diagrams** (vector = crisp at any size).
Use **PNG for heatmaps/photos** (already raster, PNG is lossless).
Both compile fine in pdflatex. PDF is strictly better for publication quality on line charts.

---

## Figure Status

| Label | File needed | Status | Source |
|-------|------------|--------|--------|
| `fig:dist` | `fig_dist.pdf` | ❌ TODO | Regenerate from 1K blind logits |
| `fig:mg` | `fig_mg.pdf` | ❌ TODO | `analysis-quadrants.ipynb` |
| `fig:degradation` | `fig_degradation.pdf` | ❌ TODO | After scoring fix + pronominalized |
| `fig:gated` | `fig_collapse.pdf` | ❌ TODO | From collapse rate numbers below |
| `fig:interrater` | `ac1_heatmap.png` | ✓ EXISTS | `analysis/figures/ac1_heatmap.png` |
| `fig:finetune` | `fig_finetune.pdf` | ❌ TODO | `analysis-finetuned.ipynb` + k-fold |

---

## Generation Instructions

### fig:dist — Answer Distribution Biases
Reproduce ACL Figure 1 with 1K-question blind data.
- Data: `evaluation/logits/pretrained/*/vqa_1k_control_blind.jsonl`
- Plot: two stacked horizontal bars (top: yes/no, bottom: count)
- Bars: Model (avg across 4 models) | Humans | Ground Truth
- Key numbers:
  - Yes/no: Model 70% no / 30% yes → Human 53% no → GT 47% no
  - Count: Model 75% "0" → Human distributed → GT distributed
- Style: match ACL Figure 1 colors; save as PDF

### fig:mg — Multimodal Gain by Quadrant
Reproduce ACL Figure 2 (bar chart) with updated models (add Qwen3-VL 32B).
- Data: `analysis/notebook/analysis-quadrants.ipynb`
- X-axis: Shared Wrong | Human-Only | Shared Correct | Model-Only
- Y-axis: Mean MG ± bootstrap CI
- Two grouped bars: VQA (blue) + MMStar (orange)
- Save as PDF

### fig:degradation — Control Variant Degradation Curve
**BLOCKED on:** (1) scoring fix (data_issues.md §4), (2) pronominalized inference.
- After fixes: line plot, x = [original, deictic_rm, object_rm, weaker_obj, pronominalized]
- Two y-axes or two panels: accuracy (left) + mean logprob (right)
- One line per model family, with different line styles
- Expected: moderate drop at object_removed, largest at pronominalized
- Save as PDF (two-column width figure)

### fig:gated — Soft Abstention Collapse Rates
Ready to generate from known numbers — no data dependency.
- Bar chart, one bar per model
- Values: Qwen3-VL-8B=71.9%, LLaVA-Mistral=51.9%, LLaVA-Vicuna=28.3%, LLaVA-1.5=17.1%
- Color-code by model family (Qwen=blue, LLaVA=orange shades)
- Add horizontal line at 50% as reference
- X-axis label: model name; Y-axis: "Soft abstention collapse rate (%)"
- Save as PDF
- **This figure can be generated NOW** — no pending data needed

### fig:finetune — Alignment vs. Accuracy Scatter
Reproduce ACL Figure 3 with k-fold error bars.
- Data: `analysis/notebook/analysis-finetuned.ipynb` + k-fold ρ results
- X-axis: Spearman ρ (0.40–0.65); Y-axis: VQA accuracy (74–90%)
- Markers: ● Pretrained, ■ SFT, ▲ JS
- Colors: Blue = GT-trained, Orange = Human-blind-trained
- Add: error bars from 5-fold CV on SFT/JS markers
- Add: Qwen3-VL results alongside InternVL/LLaVA
- Red dashed vertical line = human baseline ρ
- Save as PDF (full-width, figure*)

---

## Quick Win: fig:gated Can Be Made Right Now

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # editable text in PDF

models = ['Qwen3-VL-8B', 'LLaVA v1.6\n(Mistral)', 'LLaVA v1.6\n(Vicuna)', 'LLaVA v1.5']
rates = [71.9, 51.9, 28.3, 17.1]
colors = ['#2196F3', '#FF9800', '#FF7043', '#E91E63']

fig, ax = plt.subplots(figsize=(4.5, 2.8))
bars = ax.bar(models, rates, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, label='50% reference')
ax.set_ylabel('Soft abstention collapse rate (%)')
ax.set_ylim(0, 85)
ax.legend(fontsize=8)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('fig_collapse.pdf', bbox_inches='tight')
plt.savefig('fig_collapse.png', bbox_inches='tight', dpi=150)
```
