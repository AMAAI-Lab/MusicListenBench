#!/usr/bin/env bash
# GRPO on the Slakh2100 training split, with the settings of the main runs (LoRA: 1000 steps, 8 prompts per step on
# 2 GPUs; full fine-tuning: 1000 steps, 500 for audio-flamingo3). The Slakh items come from
# musiclistenbench/other_sets/build_slakh.py.
#
#   bash scripts/train_slakh.sh lora <model> <gpu-ids> [slakh-dir]
#   bash scripts/train_slakh.sh full <model> <gpu-ids> [slakh-dir]
#
# <model>: qwen2-audio | qwen2.5-omni | audio-flamingo3 | phi4-multimodal
# Example: bash scripts/train_slakh.sh lora qwen2.5-omni 0,1 generated/slakh
set -euo pipefail
MODE=$1; MODEL=$2; GPUS=$3; D=${4:-generated/slakh}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OUT=checkpoints/slakh_${MODEL}_${MODE}
COMMON=(--model "$MODEL" --save-every 100 --prompts-per-step 4
        --train-json "$D/train.json" --audio-root "$D/audio_train" --out-dir "$OUT")
if [[ "$MODE" == "lora" ]]; then
  CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --multi_gpu --num_processes "$NPROC" \
    -m musiclistenbench.training.train_grpo --steps 1000 "${COMMON[@]}"
else
  STEPS=1000; [[ "$MODEL" == "audio-flamingo3" ]] && STEPS=500
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS \
    accelerate launch --multi_gpu --num_processes "$NPROC" \
    -m musiclistenbench.training.train_grpo_fullft --trainable language_model --grad-checkpointing --steps "$STEPS" "${COMMON[@]}"
fi
