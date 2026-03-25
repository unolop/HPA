# Paper TODOs

Consolidated checklist for `latex/AnonymousSubmission/LaTeX/paper.tex`.
This summary reflects the current draft after unsupported or conflicting claims
were softened in the LaTeX source.

## Must Fix Before Sharing Externally

- Reconcile the model inventory across the paper.
  - Main text, tables, and appendix currently mix Qwen3-VL, LLaVA, and InternVL coverage.
  - Pick one consistent scope and make every section match it.

- Reconcile human-condition wording with the actual analysis.
  - Current notes indicate the agreement analysis uses 20 participants on 374 VQA questions in `inst_blind`.
  - The draft still presents broader human claims around `N=24, 641 questions`.
  - State clearly which human subset is used for each result.

- Finish the control-question scoring fix before making degradation claims.
  - Use VQAv2 10-annotator soft scoring for both the original and control variants.
  - Do not interpret degradation curves until this is harmonized.
  - See `notes/data_issues.md`.

- Resolve control-data validity issues.
  - Qwen3 control files have self-referential `answers`.
  - LLaVA logits can join multiword answers without spaces.
  - Qwen3-VL-4B control outputs are unusable.
  - See `notes/data_issues.md`.

- Replace placeholders or remove the corresponding narrative.
  - `fig:dist`
  - `fig:mg`
  - `fig:degradation`
  - `fig:gated`
  - `fig:finetune`

## High Priority Analysis Work

- Run matched decoder-only baselines with the same inference pipeline.
  - Reuse the existing inference code and pass `None` / no image instead of the blank image path where the model class allows it.
  - This should give a cleaner VLM-vs-LLM comparison than only using blank-image blind inference.
  - Priority backbone matches:
    - Qwen3-VL -> `Qwen/Qwen3-8B`
    - LLaVA v1.5 / LLaVA v1.6 Vicuna -> `lmsys/vicuna-7b-v1.5`
    - LLaVA v1.6 Mistral -> `mistralai/Mistral-7B-Instruct-v0.2`
  - Keep prompt formatting as close as possible to the VLM prompts, just removing image handling.

- Confirm local inference behavior for decoder-only mode.
  - If the current code path does not accept `image=None`, add a decoder-only branch instead of forcing a fake image.
  - Avoid mixing blank-image results with true decoder-only results in the same table.

- Run k-fold evaluation for the fine-tuning section.
  - Replace single-split alignment numbers with mean ± std.
  - Add uncertainty bars to the alignment-vs-accuracy figure.

- Decide whether the fine-tuning section stays in the main paper.
  - Keep only if k-fold results are ready.
  - Otherwise reduce it to a brief preliminary note or move it out.

- Decide whether the new targeted human study stays in the main paper.
  - It is currently a plan, not a completed result.
  - Keep as future work unless data collection and analysis are complete.

- Compute confidence calibration metrics.
  - ECE
  - Reliability diagrams
  - Blind vs standard-condition comparison

- Run the corpus-frequency correlation promised by the framing.
  - Blind accuracy vs answer frequency in VQA training data
  - This is important for the McCoy-style mechanistic argument.

- Decide whether to remove InternVL from the paper entirely.
  - If Titan RTX cannot run it reliably, dropping it is cleaner than keeping partial results.
  - If you want a replacement, the safest current option is `Qwen/Qwen2.5-VL-7B-Instruct`:
    - Hugging Face model exists
    - Qwen family is already in the paper
    - ms-swift clearly supports Qwen3-VL / Llava families and broadly supports Qwen multimodal workflows
  - A secondary option is `microsoft/Phi-3.5-vision-instruct`, which exists on Hugging Face, but ms-swift support is less explicit from the current top-level support list than for Qwen/LLaVA.
  - Unless architectural diversity is essential, prefer replacing InternVL with Qwen2.5-VL-7B or simply dropping InternVL and strengthening the decoder-only baseline section.

## Draft Cleanup

- Remove all remaining `\TODO{...}` blocks before submission.

- Check every precise numeric claim against the current tables and notes.
  - Especially instruction-effect deltas
  - Human vs model numerical-question comparisons
  - Any model-family ranking statements

- Keep `Blind` and `Blind+Inst` terminology consistent.
  - Several notes and analyses still use `inst_blind`.

- Add one short methodological note wherever needed when comparing humans and models.
  - Human difficulty uses averaged VQA soft scores.
  - Model accuracy is binary per question.
  - See `notes/metrics_agreement.md`.

## Figures

- Fix figure paths and ensure compilation uses the correct relative paths from `latex/AnonymousSubmission/LaTeX/`.

- Generate `fig:gated` first.
  - It is the cleanest figure with the least dependency risk.

- Generate `fig:dist` and `fig:mg` next.
  - They support claims already central to the draft.

- Leave `fig:degradation` and `fig:finetune` until the blocked analyses are resolved.

## Source Notes

- Data issues: `notes/data_issues.md`
- Figure status: `notes/figures_todo.md`
- Agreement metrics: `notes/metrics_agreement.md`
- Abstention analysis: `notes/A1_abstention.md`

## Online Checks

- Hugging Face:
  - `Qwen/Qwen3-8B`
  - `lmsys/vicuna-7b-v1.5`
  - `mistralai/Mistral-7B-Instruct-v0.2`
  - `Qwen/Qwen2.5-VL-7B-Instruct`
  - `microsoft/Phi-3.5-vision-instruct`

- ms-swift:
  - Current repo front page explicitly lists support for Qwen3, Mistral, Qwen3-VL, and Llava families, which is enough to make the matched Qwen/Vicuna/Mistral decoder-only plan low-risk.
  - The same support list also includes InternVL3.5, but that does not help if the model is not runnable on your Titan RTX.
