#!/usr/bin/env bash
# Evaluates every open model of the paper on both test sets (1,000 clean items and
# 2,000 flip/stay items), sharded across GPUs, and writes the per-item answers to
# results/per_item/<model>[_grpo_lora|_grpo_full][_perturb].jsonl, the files that
# musiclistenbench.scoring.paper_tables reads.
#
#   - the 8 open models, base weights
#   - the 4 trainable models with their GRPO checkpoints (LoRA and full fine-tuning),
#     taken from checkpoints/grpo_<run>/checkpoint-<step>
#
# A failure on one model does not abort the sweep; failures are listed at the end.
#
# Usage: bash scripts/eval_all_models.sh [gpu-list]     (default GPUs: 0,1,2,3,4,5,6,7)
#   STEP=1000 selects the checkpoint step (default 1000; Audio Flamingo 3 full
#   fine-tuning uses 500, see the paper's Section 4).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repository root

GPUS="${1:-0,1,2,3,4,5,6,7}"
STEP="${STEP:-1000}"
SUBDIR="per_item"
mkdir -p "${MLB_RESULTS_DIR:-results}/${SUBDIR}"

FAILED=()
run() {   # run <slug> <run_probe_gpus args...>
    local slug="$1"; shift
    for set_flag in "" "--perturb"; do
        local suffix=""; [[ -n "$set_flag" ]] && suffix="_perturb"
        echo ""
        echo "### ${slug}${suffix}"
        if ! bash scripts/run_probe_gpus.sh "$GPUS" "$@" $set_flag --out-name "${SUBDIR}/${slug}${suffix}"; then
            echo "!! FAILED: ${slug}${suffix}"
            FAILED+=("${slug}${suffix}")
        fi
    done
}

# checkpoint directories written by musiclistenbench.training
declare -A LORA_DIR=(
    [qwen2-audio]=checkpoints/grpo_qwen2_audio_lora
    [qwen2.5-omni]=checkpoints/grpo_omni_lora
    [audio-flamingo3]=checkpoints/grpo_af3_lora
    [phi4-multimodal]=checkpoints/grpo_phi4_lora
)
declare -A FULL_DIR=(
    [qwen2-audio]=checkpoints/grpo_qwen2_audio_fullft
    [qwen2.5-omni]=checkpoints/grpo_omni_fullft
    [audio-flamingo3]=checkpoints/grpo_af3_fullft
    [phi4-multimodal]=checkpoints/grpo_phi4_fullft
)
declare -A SLUG=(
    [qwen2-audio]=qwen2_audio [qwen2.5-omni]=qwen2_5_omni [audio-flamingo3]=audio_flamingo3
    [phi4-multimodal]=phi4_multimodal [kimi-audio]=kimi_audio [mimo-audio]=mimo_audio
    [fun-audio-chat]=fun_audio_chat [step-audio2]=step_audio2
)

# ---- 1. base weights of all eight open models --------------------------------
for m in qwen2-audio qwen2.5-omni audio-flamingo3 phi4-multimodal kimi-audio mimo-audio fun-audio-chat step-audio2; do
    run "${SLUG[$m]}" --backend "$m"
done

# ---- 2. GRPO checkpoints of the four trainable models -------------------------
for m in qwen2-audio qwen2.5-omni audio-flamingo3 phi4-multimodal; do
    lora="${LORA_DIR[$m]}/checkpoint-${STEP}"
    full_step="$STEP"; [[ "$m" == "audio-flamingo3" ]] && full_step=500
    full="${FULL_DIR[$m]}/checkpoint-${full_step}"
    if [[ -d "$lora" ]]; then run "${SLUG[$m]}_grpo_lora" --backend "$m" --lora-checkpoint "$lora"
    else echo "!! skip $m LoRA: $lora not found"; fi
    if [[ -d "$full" ]]; then run "${SLUG[$m]}_grpo_full" --backend "$m" --full-checkpoint "$full"
    else echo "!! skip $m full fine-tuning: $full not found"; fi
done

echo ""
echo "=== evaluation sweep complete ==="
if (( ${#FAILED[@]} > 0 )); then
    echo "${#FAILED[@]} run(s) failed:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "next: python -m musiclistenbench.scoring.paper_tables"
