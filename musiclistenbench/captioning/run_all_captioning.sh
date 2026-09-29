#!/usr/bin/env bash
# MusicCaps captioning side-effect check: base vs GRPO (LoRA / full fine-tuning)
# for Qwen2-Audio-7B-Instruct and the Qwen2.5-Omni-7B thinker.
#
# Run from the repository root:
#   bash musiclistenbench/captioning/run_all_captioning.sh              # download all clips, 6 jobs on GPUs 2-7
#   GPUS="0,1" bash musiclistenbench/captioning/run_all_captioning.sh
#   CKPT_STEP=500 bash ...     # use checkpoint-500 instead of the latest checkpoint
#   SKIP_DOWNLOAD=1 bash ...   # clips already downloaded
#   SKIP_CAPTION=1 bash ...    # only (re)score existing captions
#
# Clips are downloaded from YouTube with yt-dlp (MusicCaps distributes only ids
# and captions), so 2,735 of the 2,858 evaluation clips were still available when
# we ran it. The clip list of that run is in results/captioning/clip_ids.txt.
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-python}"
TARGET="${TARGET:-2858}"
GPUS="${GPUS:-2,3,4,5,6,7}"
CKPT_STEP="${CKPT_STEP:-}"
MANIFEST="${MANIFEST:-musiccaps/manifest.jsonl}"
OUT="${MLB_RESULTS_DIR:-results}/captioning"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
mkdir -p "$OUT/logs"

# ---- step 1: download clips (resumable) --------------------------------
if [[ -z "${SKIP_DOWNLOAD:-}" ]]; then
    n_have=$( [[ -f "$MANIFEST" ]] && wc -l < "$MANIFEST" || echo 0 )
    if (( n_have < TARGET )); then
        echo "[step 1] downloading MusicCaps clips ($n_have/$TARGET so far)..."
        $PY -m musiclistenbench.captioning.download_musiccaps --target "$TARGET" 2>&1 | tee "$OUT/logs/download.log"
        n_have=$(wc -l < "$MANIFEST")
    else
        echo "[step 1] manifest already has $n_have clips -- skip"
    fi
    if (( n_have < 50 )); then
        echo "!! too few clips ($n_have) -- YouTube is probably rate-limiting. See musiccaps/failed.tsv"; exit 1
    fi
fi

# ---- step 2: caption with all 6 configurations --------------------------
JOBS=()
run_job () {  # model variant gpu
    local model="$1" variant="$2" gpu="$3"
    local tag="${model//./_}"; tag="${tag//-/_}_${variant}"
    local extra=()
    [[ -n "$CKPT_STEP" ]] && extra+=(--ckpt-step "$CKPT_STEP")
    echo "[step 2] ${model}/${variant} -> cuda:${gpu}"
    $PY -m musiclistenbench.captioning.run_captioning --model "$model" --variant "$variant" \
        --device "cuda:${gpu}" --manifest "$MANIFEST" \
        "${extra[@]}" > "$OUT/logs/caption_${tag}.log" 2>&1 &
    JOBS+=($!)
}

if [[ -z "${SKIP_CAPTION:-}" ]]; then
    if (( ${#GPU_ARR[@]} < 6 )); then
        echo "[step 2] ${#GPU_ARR[@]} GPU(s) given -- running configurations round-robin per GPU"
    fi
    i=0
    for model in qwen2-audio qwen2.5-omni; do
        for variant in base lora fullft; do
            run_job "$model" "$variant" "${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
            i=$((i+1))
        done
    done
    fail=0
    for pid in "${JOBS[@]}"; do
        wait "$pid" || fail=1
    done
    for model in qwen2-audio qwen2.5-omni; do
        for variant in base lora fullft; do
            tag="${model//./_}"; tag="${tag//-/_}_${variant}"
            n=$( [[ -f "$OUT/preds_${tag}.jsonl" ]] && wc -l < "$OUT/preds_${tag}.jsonl" || echo 0 )
            echo "  preds_${tag}.jsonl: $n captions (log: $OUT/logs/caption_${tag}.log)"
            grep -q "Traceback" "$OUT/logs/caption_${tag}.log" 2>/dev/null && echo "  !! error in ${tag} log"
        done
    done
    if (( fail )); then echo "!! at least one captioning job failed -- check logs above"; fi
fi

# ---- step 3: score -------------------------------------------------------
echo "[step 3] scoring..."
$PY -m musiclistenbench.captioning.score_captions 2>&1 | tee "$OUT/logs/score.log"
echo "done -> $OUT/summary.md"
