#!/usr/bin/env bash
# Re-run Qwen3 standalone LLM think variants with qwen3_thinking template
# (0.6B, 4B, 8B had empty think traces because they used the default qwen3
#  template which prefills <think>\n\n</think>, suppressing actual reasoning)
#
# Strategy: run inference to a temp dir, then move results into *_think dirs.
set -euo pipefail

cd "$(dirname "$0")/.."

log() { echo "[$(date '+%H:%M:%S')] $*"; }

BACKBONE="evaluation/logits/backbone/pretrained"

run_think() {
    local size=$1
    local gpu=$2
    local model_name="Qwen3-${size}"
    local think_dir="${BACKBONE}/${model_name}_think"
    local tmp_dir="${BACKBONE}/${model_name}_think_new"

    # Back up old results
    if [ -d "$think_dir" ]; then
        local bak="${think_dir}_bak_$(date +%Y%m%d)"
        if [ ! -d "$bak" ]; then
            log "Backing up ${think_dir} → ${bak}"
            cp -r "$think_dir" "$bak"
        fi
    fi

    mkdir -p "$tmp_dir"

    for cond in _control_inst_blind _control_blind; do
        log "=== ${model_name} think ${cond} ==="
        CUDA_VISIBLE_DEVICES=$gpu conda run -n zero python3 evaluation/inference.py \
            --model "Qwen/${model_name}" \
            --model_type lm \
            --dataset vqa_1k \
            --condition "$cond" \
            --savedir "evaluation/logits/backbone_think_tmp" \
            --template_type qwen3_thinking

        # Move output to think dir
        local src="evaluation/logits/backbone_think_tmp/pretrained/${model_name}/vqa_1k${cond}.jsonl"
        if [ -f "$src" ]; then
            mv "$src" "${tmp_dir}/vqa_1k${cond}.jsonl"
            log "  → ${tmp_dir}/vqa_1k${cond}.jsonl"
        else
            log "  WARNING: expected output not found at ${src}"
            # Try to find it
            find evaluation/logits/backbone_think_tmp -name "*.jsonl" 2>/dev/null
        fi
    done

    # Verify non-empty think traces
    log "Verifying ${model_name} think traces..."
    conda run -n zero python3 -c "
import json, re
for cond in ['vqa_1k_control_inst_blind', 'vqa_1k_control_blind']:
    f = '${tmp_dir}/' + cond + '.jsonl'
    try:
        with open(f) as fh:
            lines = [json.loads(l) for l in fh]
        nonempty = sum(1 for l in lines
                       if re.search(r'<think>(.+?)</think>',
                                    l.get('generated_answers',{}).get('question',''),
                                    re.DOTALL)
                       and re.search(r'<think>(.+?)</think>',
                                     l.get('generated_answers',{}).get('question',''),
                                     re.DOTALL).group(1).strip())
        print(f'  {cond}: {nonempty}/{len(lines)} non-empty think traces')
    except FileNotFoundError:
        print(f'  {cond}: FILE NOT FOUND')
"

    # Replace old think dir with new
    if [ -d "$tmp_dir" ] && [ "$(ls -A "$tmp_dir")" ]; then
        rm -rf "$think_dir"
        mv "$tmp_dir" "$think_dir"
        log "Replaced ${think_dir}"
    fi
}

# Run sequentially — all fit in 24GB TITAN RTX
run_think "0.6B" 0
run_think "4B"   0
run_think "8B"   0

# Cleanup temp dir
rm -rf evaluation/logits/backbone_think_tmp

log "All done. Re-run the pair cache build to update downstream data."
