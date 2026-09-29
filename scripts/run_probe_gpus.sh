#!/usr/bin/env bash
# Runs one model over an eval set, sharded across an explicit list of GPUs, merges
# the shards, and (for the flip/stay set) scores false-flip / miss rates.
#
# Usage:
#   bash scripts/run_probe_gpus.sh <gpu-ids> [options]
#
#   <gpu-ids>   comma-separated GPU ids, one model copy per GPU, e.g. 2,3,4,5,6,7
#
# MODEL (input model path -- pick at most one; default = base Qwen2-Audio-7B):
#   --lora-checkpoint PATH   a train_grpo LoRA adapter dir (e.g. .../checkpoint-500)
#   --full-checkpoint PATH   a standalone full-FT model dir  (e.g. .../checkpoint-250)
#
# MULTI-MODEL (evaluate a different audio-LLM instead of Qwen2-Audio):
#   --backend NAME           run any registered backend (musiclistenbench/backends/) via run_probe_mm.py
#                            (forced-choice logprob, one metric, same output schema).
#                            NAME: qwen2-audio | qwen2.5-omni | audio-flamingo3 |
#                            phi4-multimodal | fun-audio-chat | kimi-audio |
#                            mimo-audio | step-audio2. Combine with --lora-checkpoint /
#                            --full-checkpoint to score a GRPO-trained model.
#
# EVAL SET (which data):
#   --perturb                use the controlled flip/stay set: sets
#                            --eval-json data/eval_perturb.json,
#                            --audio-root data/audio_perturb, and runs
#                            score_perturb.py (false-flip / miss) after merging.
#   --eval-json PATH         eval json          (default: data/eval.json)
#   --audio-root PATH        audio root         (default: data/audio/)
#   --score                  run score_perturb.py on the merged results (implied by --perturb)
#
# OUTPUT (output path):
#   --out-name NAME          base name for result files under $MLB_RESULTS_DIR (default results/)
#                            -> results/NAME.jsonl (+ NAME.shard*.jsonl, logs).
#                            Default derived from model+set, e.g.
#                            results_qwen2_audio[_grpo|_fullft][_perturb].
#
# Examples:
#   bash scripts/run_probe_gpus.sh 2,3,4,5,6,7                         # base model, base set
#   bash scripts/run_probe_gpus.sh 2,3,4,5,6,7 --perturb               # base model, flip/stay set
#   bash scripts/run_probe_gpus.sh 0,1,2,3 --full-checkpoint checkpoints/grpo_qwen2_audio_fullft/checkpoint-1000 --perturb
#   bash scripts/run_probe_gpus.sh 0,1 --eval-json data/eval_perturb.json \
#        --audio-root data/audio_perturb --out-name results_myrun --score
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repository root
DATA_DIR="${MLB_DATA_DIR:-data}"
RESULTS_ROOT="${MLB_RESULTS_DIR:-results}"

usage() { awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "${BASH_SOURCE[0]}"; }

GPU_LIST="${1:-}"
if [[ -z "$GPU_LIST" || "$GPU_LIST" == "-h" || "$GPU_LIST" == "--help" ]]; then usage; exit 1; fi
shift
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_SHARDS=${#GPUS[@]}

CHECKPOINT_FLAG=()
CKPT_TAG=""                       # "" base | "_grpo" lora | "_fullft" full
BACKEND=""                        # non-empty -> multi-model runner (run_probe_mm.py)
EVAL_JSON=""
AUDIO_ROOT=""
OUT_NAME=""
SET_TAG=""
DO_SCORE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lora-checkpoint) CHECKPOINT_FLAG=("$1" "$2"); CKPT_TAG="_grpo"; shift 2;;
        --full-checkpoint) CHECKPOINT_FLAG=("$1" "$2"); CKPT_TAG="_fullft"; shift 2;;
        --perturb) EVAL_JSON="${DATA_DIR}/eval_perturb.json"
                   AUDIO_ROOT="${DATA_DIR}/audio_perturb"; SET_TAG="_perturb"; DO_SCORE=1; shift;;
        --eval-json)  EVAL_JSON="$2"; shift 2;;
        --audio-root) AUDIO_ROOT="$2"; shift 2;;
        --backend)    BACKEND="$2"; shift 2;;
        --out-name)   OUT_NAME="$2"; shift 2;;
        --score)      DO_SCORE=1; shift;;
        -h|--help)    usage; exit 0;;
        *) echo "unknown argument: $1" >&2; usage; exit 1;;
    esac
done

# Pick the runner: --backend -> multi-model runner (run_probe_mm.py, delegates to
# musiclistenbench/backends/<backend>, forced-choice logprob); otherwise the Qwen2-Audio runner
# (run_probe.py, supports LoRA/full checkpoints). --backend is base-weights-only.
if [[ -n "$BACKEND" ]]; then
    RUN_MODULE="musiclistenbench.eval.run_probe_mm"
    # CHECKPOINT_FLAG is empty (base) or (--lora-checkpoint PATH) / (--full-checkpoint PATH)
    # to score a TRAINED checkpoint of a trainable model.
    MODEL_FLAGS=(--backend "$BACKEND" "${CHECKPOINT_FLAG[@]}")
    MODEL_DESC="backend=$BACKEND ${CHECKPOINT_FLAG[*]}"
    DEFAULT_NAME="results_${BACKEND//[.-]/_}${CKPT_TAG}${SET_TAG}"
else
    RUN_MODULE="musiclistenbench.eval.run_probe"
    MODEL_FLAGS=("${CHECKPOINT_FLAG[@]}")
    MODEL_DESC="${CHECKPOINT_FLAG[*]:-<base Qwen2-Audio-7B>}"
    DEFAULT_NAME="results_qwen2_audio${CKPT_TAG}${SET_TAG}"
fi
# Distinct result-file name per (model, eval set) so a run never merges with stale
# shards from a different model/set. Override with --out-name for full control
# (may include a subdirectory, e.g. --out-name my_results/results_foo -- created
# automatically so trained-checkpoint sweeps can land in their own folder).
RESULT_NAME="${OUT_NAME:-$DEFAULT_NAME}"
RESULT_DIR="${RESULTS_ROOT}/$(dirname -- "$RESULT_NAME")"
RESULT_BASE="$(basename -- "$RESULT_NAME")"
mkdir -p "$RESULT_DIR"

# Only pass --eval-json/--audio-root when set; otherwise run_probe uses its defaults.
EVAL_FLAGS=()
[[ -n "$EVAL_JSON" ]]  && EVAL_FLAGS+=(--eval-json "$EVAL_JSON")
[[ -n "$AUDIO_ROOT" ]] && EVAL_FLAGS+=(--audio-root "$AUDIO_ROOT")

echo "model:     ${MODEL_DESC}"
echo "eval-json: ${EVAL_JSON:-<default eval.json>}"
echo "audio:     ${AUDIO_ROOT:-<default audio/>}"
echo "output:    ${RESULT_DIR}/${RESULT_BASE}.jsonl   (shards + logs share this name)"

pids=()
shard_files=()
for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    shard_file="${RESULT_DIR}/${RESULT_BASE}.shard${idx}.jsonl"
    shard_files+=("$shard_file")
    CUDA_VISIBLE_DEVICES="$gpu" python -m "$RUN_MODULE" \
        --device cuda:0 \
        --num-shards "$NUM_SHARDS" \
        --shard "$idx" \
        "${MODEL_FLAGS[@]}" \
        "${EVAL_FLAGS[@]}" \
        --out "$shard_file" \
        > "${RESULT_DIR}/run_probe.${RESULT_BASE}.shard${idx}.log" 2>&1 &
    pids+=($!)
done

echo "launched ${NUM_SHARDS} shards on GPUs ${GPU_LIST}; logs at ${RESULT_DIR}/run_probe.${RESULT_BASE}.shard*.log"
wait "${pids[@]}"

# Explicit file list, NOT a glob, so a stale shard from a previous run at a
# different NUM_SHARDS / model / set can never leak into this merge.
MERGED="${RESULT_DIR}/${RESULT_BASE}.jsonl"
python -m musiclistenbench.eval.merge_results "${shard_files[@]}" --out "$MERGED"

if [[ "$DO_SCORE" == "1" ]]; then
    echo ""
    echo "=== flip/stay metrics (false_flip = error on stays, miss = error on flips) ==="
    python -m musiclistenbench.scoring.score_perturb "$MERGED"
fi
