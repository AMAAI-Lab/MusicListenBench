#!/usr/bin/env bash
# Paid-API model evaluation over both MusicListenBench sets — the base benchmark
# (eval.json, 1000 items, accuracy) and the controlled flip/stay perturb set
# (eval_perturb.json, 2000 items, miss / false-flip via score_perturb.py).
#
# This is the no-GPU counterpart of run_probe_gpus.sh: paid backends are
# network-bound, so instead of one model copy per GPU it parallelises with
# run_probe_mm.py's --workers flag and needs no CUDA_VISIBLE_DEVICES.
#
# Backends (see run_probe_mm.py's PAID_BACKENDS):
#   gemini-flash   Google Gemini Flash tier (default model gemini-3.8-flash;
#                  key: GEMINI_API_KEY or GOOGLE_API_KEY)
#   gpt-audio      OpenAI audio GPT (default model gpt-audio-1.5;
#                  key: OPENAI_API_KEY)
#   openrouter     any OpenRouter model (default google/gemini-3.1-pro-preview;
#                  key: OPENROUTER_API_KEY; live only -- OpenRouter has no
#                  Batch API. The paper's Gemini 2.5 Pro and GPT-Audio-mini rows use
#                  --model-id google/gemini-2.5-pro and openai/gpt-audio-mini)
#
# Usage:
#   bash scripts/run_paid_eval.sh <backend> [options]
#
# Options:
#   --api-key KEY     pass the provider key explicitly (prefer exporting the
#                     env var instead — a CLI flag is visible in shell history
#                     and process listings)
#   --model-id ID     override the backend's default model (snapshot names
#                     turn over fast; re-check the provider's model page)
#   --workers N       parallel API calls (default 4; raise/lower to your rate
#                     limit tier — on 429s the backends retry with backoff)
#   --limit N         smoke test: only the first N items of each set
#   --only SET        run only 'base' or only 'perturb' (default: both)
#   --out-name NAME   result base name under results/ (default
#                     results_<backend>; perturb adds a _perturb suffix)
#   --resume          skip items already scored in the output files and append
#                     (a crashed paid run doesn't re-pay for finished items)
#
# Examples:
#   # 1. ALWAYS smoke-test first (8 items/set, a few cents — catches key/quota/
#   #    schema/logprob-support problems before spending on a full run):
#   export GEMINI_API_KEY=...
#   bash scripts/run_paid_eval.sh gemini-flash --limit 8 --out-name /tmp/smoke_gemini_flash
#   export OPENAI_API_KEY=...
#   bash scripts/run_paid_eval.sh gpt-audio --limit 8 --out-name /tmp/smoke_gpt_audio
#
#   # 2. Full runs (base 1000 + perturb 2000 items each):
#   bash scripts/run_paid_eval.sh gemini-flash
#   bash scripts/run_paid_eval.sh gpt-audio --workers 8
#
# Results land in $MLB_RESULTS_DIR (default results/) as results_<backend>[_perturb].jsonl (+ the
# perturb run's *_perturb_summary.json), same schema as every other backend,
# so merge_results.py, score_perturb.py and paper_tables.py consume them
# unchanged.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repository root

usage() { awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1 && /^$/{print ""; next} NR>1{exit}' "${BASH_SOURCE[0]}"; }

BACKEND="${1:-}"
if [[ -z "$BACKEND" || "$BACKEND" == "-h" || "$BACKEND" == "--help" ]]; then usage; exit 1; fi
shift

case "$BACKEND" in
  gemini-flash) KEY_ENVS=("GEMINI_API_KEY" "GOOGLE_API_KEY");;
  gpt-audio) KEY_ENVS=("OPENAI_API_KEY");;
  openrouter) KEY_ENVS=("OPENROUTER_API_KEY");;
  *)
    echo "unknown backend '$BACKEND' — expected gemini-flash | gpt-audio | openrouter" >&2; usage; exit 1;;
esac

API_KEY=""
MODEL_ID=""
WORKERS=4
LIMIT_FLAG=()
ONLY="both"
OUT_NAME=""
RESUME_FLAG=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)   API_KEY="$2"; shift 2;;
        --model-id)  MODEL_ID="$2"; shift 2;;
        --workers)   WORKERS="$2"; shift 2;;
        --limit)     LIMIT_FLAG=(--limit "$2"); shift 2;;
        --only)      ONLY="$2"; shift 2;;
        --out-name)  OUT_NAME="$2"; shift 2;;
        --resume)    RESUME_FLAG=(--resume); shift;;
        -h|--help)   usage; exit 0;;
        *) echo "unknown argument: $1" >&2; usage; exit 1;;
    esac
done
[[ "$ONLY" =~ ^(base|perturb|both)$ ]] || { echo "--only must be base|perturb|both" >&2; exit 1; }

# Key present at all? (never printed). --api-key wins, else any provider env var.
if [[ -z "$API_KEY" ]]; then
    key_found=0
    for v in "${KEY_ENVS[@]}"; do
        [[ -n "${!v:-}" ]] && key_found=1
    done
    if [[ "$key_found" != "1" ]]; then
        echo "!!! none of ${KEY_ENVS[*]} is set (and no --api-key given). Export one first:" >&2
        echo "      export ${KEY_ENVS[0]}=..." >&2
        exit 1
    fi
fi

# conda env with google-genai + openai installed (requirements.txt)
PY_ENV="${PY_ENV:-mlb}"
CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$PY_ENV" 2>/dev/null || echo "note: could not activate conda env '$PY_ENV', using current python"
fi

SAFE="${BACKEND//[.-]/_}"
if [[ -z "$OUT_NAME" ]]; then
    OUT_NAME="results_${SAFE}"
    if [[ -n "$MODEL_ID" ]]; then
        # Never silently mix two different models in one results file.
        OUT_NAME="${OUT_NAME}__$(echo "$MODEL_ID" | tr -c 'a-zA-Z0-9' '_' | sed 's/_\+/_/g; s/^_//; s/_$//')"
    fi
fi

BACKEND_FLAGS=(--backend "$BACKEND" --workers "$WORKERS")
[[ -n "$API_KEY"  ]] && BACKEND_FLAGS+=(--api-key "$API_KEY")
[[ -n "$MODEL_ID" ]] && BACKEND_FLAGS+=(--model-id "$MODEL_ID")
BACKEND_FLAGS+=("${RESUME_FLAG[@]}")

RESULTS_ROOT="${MLB_RESULTS_DIR:-results}"
BASE_OUT="${RESULTS_ROOT}/${OUT_NAME}.jsonl"
PERTURB_OUT="${RESULTS_ROOT}/${OUT_NAME}_perturb.jsonl"
mkdir -p "$(dirname "$BASE_OUT")" "$(dirname "$PERTURB_OUT")"

run_set () {   # run_set <label> <eval-json> <audio-root> <out>
    local label="$1" eval_json="$2" audio_root="$3" out="$4"
    local log="${out%.jsonl}.log"
    echo "=== [$BACKEND] $label set -> $out (log: $log) ==="
    python -m musiclistenbench.eval.run_probe_mm \
        "${BACKEND_FLAGS[@]}" "${LIMIT_FLAG[@]}" \
        --eval-json "$eval_json" --audio-root "$audio_root" \
        --out "$out" 2>&1 | tee "$log"
}

if [[ "$ONLY" == "base" || "$ONLY" == "both" ]]; then
    run_set "base" data/eval.json data/audio "$BASE_OUT"
fi
if [[ "$ONLY" == "perturb" || "$ONLY" == "both" ]]; then
    run_set "flip/stay perturb" data/eval_perturb.json data/audio_perturb "$PERTURB_OUT"
    echo ""
    echo "=== [$BACKEND] flip/stay metrics (false_flip = error on stays, miss = error on flips) ==="
    python -m musiclistenbench.scoring.score_perturb "$PERTURB_OUT"
fi
