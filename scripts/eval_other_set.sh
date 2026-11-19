#!/usr/bin/env bash
# Evaluates one model on the Slakh2100 test set or on the real-recordings test set and prints the scores.
# Both sets have the item-file layout of the main benchmark (eval.json, eval_perturb.json, audio_test/).
#
#   bash scripts/eval_other_set.sh <set-dir> <gpu-ids> <run-name> [options for run_probe_gpus.sh]
#
# Examples:
#   bash scripts/eval_other_set.sh generated/slakh 0,1 omni_zero --backend qwen2.5-omni
#   bash scripts/eval_other_set.sh generated/real 0,1 omni_slakh \
#        --backend qwen2.5-omni --lora-checkpoint checkpoints/slakh_qwen2.5-omni_lora/checkpoint-1000
set -euo pipefail
D=$1; GPUS=$2; NAME=$3; shift 3
bash scripts/run_probe_gpus.sh "$GPUS" "$@" --eval-json "$D/eval.json" --audio-root "$D/audio_test" \
  --out-name "per_item/$NAME"
bash scripts/run_probe_gpus.sh "$GPUS" "$@" --eval-json "$D/eval_perturb.json" --audio-root "$D/audio_test" \
  --score --out-name "per_item/${NAME}_perturb"
python -m musiclistenbench.scoring.convert_results --clean "results/per_item/$NAME.jsonl" \
  --flip-stay "results/per_item/${NAME}_perturb.jsonl" --model "$NAME" --version v1 --readout logprob \
  --used-training-split none --date "$(date +%F)" --out "results/per_item/$NAME.submission.jsonl"
python -m musiclistenbench.scoring.score_submission "results/per_item/$NAME.submission.jsonl" \
  --eval-json "$D/eval.json" --eval-perturb-json "$D/eval_perturb.json"
