"""Multi-model runner for the probe sets (base `eval.json` and flip/stay
`eval_perturb.json`) — the benchmark-wide evaluator.

Unlike `run_probe.py` (which hardcodes Qwen2-Audio and reports both a free-generated
letter and a forced-choice logprob), this evaluates ANY registered audio-LLM backend in
`musiclistenbench/backends/*_backend.py` through the ONE uniform interface every backend implements:

    score_options(question, clip_paths, options) -> {option: logprob}

and reports the parse-free **forced-choice logprob** reading only. Two consequences that
matter for a cross-model benchmark:

  * Every model is scored on identical footing — no model is penalised for failing to
    emit a clean 'A'/'B' (a formatting artefact, not a perception failure), which is the
    fair metric and pre-empts an "unfair evaluation" objection.
  * The metric is fully deterministic (it reads logprobs; it never samples/decodes), so
    there is no decoding-temperature choice and no train-vs-eval decoding mismatch to
    reconcile.

Each backend's own chat template + audio front-end is reused (not a hand-built Qwen
prompt), so adding a model is just adding its `musiclistenbench/backends/<name>_backend.py` adapter.

Output schema matches `run_probe.py`, so `merge_results.py` and `score_perturb.py`
consume the results unchanged (the single reading fills both the generated-letter and
logprob fields — one metric, reported once).

Paid API backends (`PAID_BACKENDS` below: Gemini Flash, OpenAI gpt-audio) implement the
same interface over a network call instead of local weights, scored by LETTER ANSWER
rather than logprobs (gemini-3.8-flash rejects logprob params outright — live-confirmed
400 "Logprobs is not enabled for this model"; the decision was made to drop logprobs
from both paid models): Gemini uses enum-constrained decoding (the model can only emit
"A"/"B" — the same constrained-decoding pattern used for every Gemini call in the paper), gpt-audio uses
temperature-0 generation + a word-bounded letter parse. Their score dicts put 0.0 on
the answer letter and the sentinel on the other, so this runner's argmax reproduces the
emitted/parsed letter and every downstream consumer stays unchanged; `logprob_margin`
is a constant for these rows — read `correct`, not the margin. They add four flags,
invalid for local backends: `--api-key` (else the provider env var: GEMINI_API_KEY /
GOOGLE_API_KEY, OPENAI_API_KEY), `--model-id` (override the backend's default model —
Flash/GPT snapshots turn over fast), `--workers N` (parallel API calls; the run is
network-bound, not GPU-bound) and `--resume` (skip items already in --out and append,
so a crashed paid run doesn't re-pay for finished items). `scripts/run_paid_eval.sh`
wraps both sets (base + flip/stay) + scoring for a paid backend in one command.

Run (see `run_probe_gpus.sh --backend NAME` for the sharded launcher):
    python -m musiclistenbench.eval.run_probe_mm --backend audio-flamingo3 \
        --eval-json data/eval_perturb.json \
        --audio-root data/audio_perturb \
        --out results/results_audio_flamingo3_perturb.jsonl
"""

import argparse
import importlib
import json
import os

from musiclistenbench import paths

# Registry of evaluation backends (one adapter module per model).
BACKENDS = {
    "qwen2.5-omni": "musiclistenbench.backends.qwen_omni_backend",
    "qwen2-audio": "musiclistenbench.backends.qwen2_audio_backend",
    "audio-flamingo3": "musiclistenbench.backends.audio_flamingo3_backend",
    "fun-audio-chat": "musiclistenbench.backends.fun_audio_chat_backend",
    "kimi-audio": "musiclistenbench.backends.kimi_audio_backend",
    "mimo-audio": "musiclistenbench.backends.mimo_audio_backend",
    "step-audio2": "musiclistenbench.backends.step_audio2_backend",
    "phi4-multimodal": "musiclistenbench.backends.phi4_multimodal_backend",
    # commercial API models (evaluation only), scored by the letter they answer with
    "gemini-flash": "musiclistenbench.backends.gemini_flash_backend",
    "gpt-audio": "musiclistenbench.backends.gpt_audio_backend",
    "openrouter": "musiclistenbench.backends.openrouter_backend",
}
# Backends that hit a paid HTTP API instead of local weights: base-weights-only
# (no --*-checkpoint), no GPU needed, and the only ones accepting --api-key /
# --model-id / --workers>1 below.
PAID_BACKENDS = {"gemini-flash", "gpt-audio", "openrouter"}
LETTERS = ["A", "B"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS),
                        help="which audio-LLM backend in musiclistenbench/backends/ to evaluate")
    parser.add_argument("--lora-checkpoint", default=None,
                        help="optional: merge a train_grpo LoRA adapter into the base backend "
                             "before scoring, i.e. evaluate a TRAINED LoRA checkpoint")
    parser.add_argument("--full-checkpoint", default=None,
                        help="optional: load a full-FT save_pretrained dir instead of base weights "
                             "(evaluate a TRAINED full-FT checkpoint); mutually exclusive with --lora-checkpoint")
    parser.add_argument("--eval-json", default=paths.EVAL_JSON)
    parser.add_argument("--audio-root", default=paths.AUDIO_DIR)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="cap items, for a smoke test")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split the eval set across N parallel processes (one per GPU)")
    parser.add_argument("--shard", type=int, default=0, help="which shard, 0-indexed")
    parser.add_argument("--api-key", default=None,
                        help="paid backends only: provider API key. Prefer the env var "
                             "(GEMINI_API_KEY/GOOGLE_API_KEY, OPENAI_API_KEY) -- a CLI flag "
                             "is visible in shell history and process listings")
    parser.add_argument("--model-id", default=None,
                        help="paid backends only: override the backend's default model id "
                             "(e.g. a newer Flash / GPT-audio snapshot)")
    parser.add_argument("--workers", type=int, default=1,
                        help="paid backends only: N parallel API calls (the run is "
                             "network-bound; local GPU backends must stay at 1)")
    parser.add_argument("--resume", action="store_true",
                        help="skip items whose voice path already has a line in --out and "
                             "append instead of truncating (safe rerun after a crash; a "
                             "truncated tail line is dropped with everything after it)")
    args = parser.parse_args()

    if args.out is None:
        safe = args.backend.replace(".", "_").replace("-", "_")
        args.out = os.path.join(paths.RESULTS_DIR, f"results_{safe}.jsonl")

    if args.lora_checkpoint and args.full_checkpoint:
        parser.error("pass at most one of --lora-checkpoint / --full-checkpoint")
    if args.backend not in PAID_BACKENDS and (args.api_key or args.model_id or args.workers != 1):
        parser.error("--api-key / --model-id / --workers apply only to the paid API "
                     f"backends: {', '.join(sorted(PAID_BACKENDS))}")
    if args.backend in PAID_BACKENDS and (args.lora_checkpoint or args.full_checkpoint):
        parser.error(f"{args.backend} is a paid API model -- no local checkpoints to load")
    backend = importlib.import_module(BACKENDS[args.backend])
    if args.full_checkpoint:
        # Point the backend at the full-FT checkpoint dir (it holds model + processor)
        # instead of the base repo; the backend's model/processor CLASSES are unchanged.
        if not hasattr(backend, "MODEL_ID"):
            raise SystemExit(f"--full-checkpoint unsupported for backend {args.backend!r} (no MODEL_ID)")
        backend.MODEL_ID = args.full_checkpoint
        print(f"loading full-FT checkpoint -> {args.full_checkpoint}")
    load_kwargs = {"device": args.device}
    if args.backend in PAID_BACKENDS:
        # Paid backends' load() additionally takes the key/model overrides (and
        # ignores `device` -- no GPU involved).
        load_kwargs.update(api_key=args.api_key, model_id=args.model_id)
    backend.load(**load_kwargs)
    if args.lora_checkpoint:
        # Fold a trained LoRA adapter into the base model held in the backend's singleton
        # state, so score_options() runs the TRAINED checkpoint. Same merge_and_unload path
        # run_probe.py uses for Qwen; VERIFY the adapter matches this backend's base model.
        from peft import PeftModel
        st = backend.load()
        if not (isinstance(st, dict) and "model" in st):
            raise SystemExit(f"--lora-checkpoint unsupported for backend {args.backend!r} "
                             f"(its load() doesn't expose _state['model'])")
        st["model"] = PeftModel.from_pretrained(st["model"], args.lora_checkpoint).merge_and_unload()
        st["model"].eval()
        print(f"merged LoRA checkpoint -> {args.lora_checkpoint}")

    with open(args.eval_json) as f:
        items = json.load(f)
    if args.limit is not None:
        items = items[: args.limit]
    if args.num_shards > 1:
        items = items[args.shard::args.num_shards]

    out_mode = "w"
    if args.resume and os.path.exists(args.out):
        # Keep only cleanly-parsed complete lines; a truncated tail line (crash mid-write)
        # and anything after it are dropped so they can't corrupt the appended rerun.
        done_voices, kept_lines = set(), []
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_voices.add(json.loads(line)["voice"])
                    kept_lines.append(line)
                except (json.JSONDecodeError, KeyError):
                    break
        with open(args.out, "w") as f:
            f.writelines(l + "\n" for l in kept_lines)
        out_mode = "a"
        before = len(items)
        items = [it for it in items if it["voice"][0] not in done_voices]
        print(f"--resume: {len(done_voices)} items already scored, {len(items)}/{before} to go")

    def score_item(item):
        voice_relpath = item["voice"][0]
        task = voice_relpath.split("/")[0]
        question = item["conversations"][0]["value"].replace("<audio>", "").strip()
        gold = item["conversations"][1]["value"].strip().upper()
        clip_path = os.path.join(args.audio_root, voice_relpath)

        logprobs = backend.score_options(question, [clip_path], LETTERS)
        predicted = max(logprobs, key=logprobs.get)
        ranked = sorted(logprobs.values(), reverse=True)
        margin = (ranked[0] - ranked[1]) if len(ranked) > 1 else None
        correct = predicted == gold

        # One forced-choice reading fills BOTH the generated-letter fields and the
        # logprob fields, so merge_results.py / score_perturb.py stay unchanged and
        # their `_gen` and `_lp` columns coincide (a single metric, reported once).
        record = {
            "voice": voice_relpath, "task": task, "question": question,
            "gold": gold, "raw_text": predicted, "predicted": predicted, "correct": correct,
            "logprob_A": logprobs["A"], "logprob_B": logprobs["B"], "logprob_margin": margin,
            "logprob_predicted": predicted, "logprob_correct": correct,
            "backend": args.backend,
        }
        if args.model_id:
            record["model_id"] = args.model_id   # paid overrides; batch rows record it too
        return task, record

    stats = {}
    n_written = 0

    def on_record(task, record):
        nonlocal n_written
        s = stats.setdefault(task, {"n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += int(record["correct"])
        out_f.write(json.dumps(record) + "\n")
        n_written += 1
        if n_written % 25 == 0:
            out_f.flush()
            print(f"[{args.backend} shard {args.shard}] [{n_written}/{len(items)}] running...", flush=True)

    with open(args.out, out_mode) as out_f:
        if args.workers > 1:
            # Paid-API mode: the backend's client is thread-safe and each call is an
            # independent HTTP request. pool.map yields results in input order, so the
            # output file stays in eval-set order and stats/progress update in the main
            # thread only. An exception propagates on iteration; already-written lines
            # are complete jsonl rows, and --resume finishes the rest on rerun.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for task, record in pool.map(score_item, items):
                    on_record(task, record)
        else:
            for item in items:
                on_record(*score_item(item))
        out_f.flush()

    note = f" (shard {args.shard}/{args.num_shards})" if args.num_shards > 1 else ""
    print(f"\n{args.backend} results{note}:")
    print(f"{'task':<12} {'n':>5} {'acc':>10}")
    tn = tc = 0
    for task, s in sorted(stats.items()):
        print(f"{task:<12} {s['n']:>5} {s['correct'] / s['n']:>10.3f}")
        tn += s["n"]; tc += s["correct"]
    print(f"{'overall':<12} {tn:>5} {(tc / tn if tn else float('nan')):>10.3f}")
    print(f"\nper-item results -> {args.out}")


if __name__ == "__main__":
    main()
