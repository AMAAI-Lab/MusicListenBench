"""Batch-API driver for the paid backends (gemini-flash, gpt-audio) over
the MusicListenBench sets — the 50%-off, no-rate-limit alternative to live
`run_paid_eval.sh` / `run_probe_mm.py --workers N`.

Same two sets, same single LETTER-ANSWER metric as the live paid mode (no
logprobs anywhere — gemini-3.8-flash rejects them outright per the live
smoke; Gemini answers arrive enum-constrained, gpt-audio answers are parsed
word-bounded), same output schema (as run_probe_mm.py rows + a "channel"
and "model_id" field), so merge_results.py / score_perturb.py consume
batch rows unchanged. The provider backends (musiclistenbench/backends/gemini_flash_backend.py,
musiclistenbench/backends/gpt_audio_backend.py) expose:

    build_batch_record/line(key, question, clip_paths, options)
    submit_batch(records, jsonl_dir)   -> [job ids]   (chunked under the
                                            provider's input-file cap)
    poll_jobs(job_ids)                 -> {job_id: status}
    fetch_jobs(job_ids)                -> {key: response-or-None}

Stages (a submit/collect split — batch
turnaround is minutes-to-24h, so submit returns immediately):

  submit   build the JSONL records for the chosen set(s) and submit one batch
           job per size chunk. --dry-run builds + reports (items, chunks,
           payload MB, model) without touching the network — works keyless.
           --wait keeps polling and collects when done.
  status   print the current provider status of every recorded job.
  collect  poll recorded jobs; when ALL are terminal and succeeded, download,
           parse, score, and write the results file (in eval-set order,
           joined on the short k<idx> key). Missing/errored requests are
           reported and simply left out — then fill them live with
           `run_probe_mm --backend <b> --resume --out <same file>`, which
           appends exactly the missing voices. Re-running collect MERGES
           with an existing results file (batch answers win, unrelated rows
           kept), so live-filled items survive.

The API key is never written to the jobs-state file — export the provider env
var (GEMINI_API_KEY/GOOGLE_API_KEY, OPENAI_API_KEY) or pass --api-key per
invocation. Model is fixed per run: --model-id overrides the backend default
and is baked into every request + recorded in the state file and result rows
(OpenAI batch files are single-model; we satisfy that by construction).

Usage (from the repo root, mlb env):
    # keyless rehearsal — chunk plan + payload size, nothing submitted:
    python -m musiclistenbench.eval.run_paid_batch submit --backend gemini-flash --dry-run
    python -m musiclistenbench.eval.run_paid_batch submit --backend gpt-audio --dry-run --set perturb

    # real submission (BOTH sets by default):
    export GEMINI_API_KEY=...
    python -m musiclistenbench.eval.run_paid_batch submit --backend gemini-flash [--model-id ...]
    # ... later ...
    python -m musiclistenbench.eval.run_paid_batch status
    python -m musiclistenbench.eval.run_paid_batch collect            # writes + scores results
    python -m musiclistenbench.eval.run_paid_batch collect --wait     # block until done

⚠️ Smoke-test LIVE first with the exact --model-id you'll batch
(run_paid_eval.sh --limit 8): batch validation is all-or-nothing per file and
cannot adapt request parameters the way the live backends do.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from musiclistenbench import paths

ROOT = paths.REPO_ROOT
RESULTS = paths.RESULTS_DIR
JOBS_DIR = os.path.join(RESULTS, "paid_batch_jobs")

BACKENDS = {
    "gemini-flash": ("musiclistenbench.backends.gemini_flash_backend", "build_batch_record"),
    "gpt-audio": ("musiclistenbench.backends.gpt_audio_backend", "build_batch_line"),
}
SETS = {  # set name -> (eval json, audio root)
    "base": (paths.EVAL_JSON, paths.AUDIO_DIR),
    "perturb": (paths.EVAL_PERTURB_JSON, paths.AUDIO_PERTURB_DIR),
}
LETTERS = ("A", "B")

# Provider status values that mean "no more results will arrive".
_TERMINAL = {"succeeded", "failed", "cancelled", "canceled", "expired"}
_SUCCEEDED = {"succeeded", "completed"}


def _norm_state(status):
    s = str(status).lower()
    if s.startswith("job_state_"):
        s = s[len("job_state_"):]
    return s


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def load_backend(name, api_key=None, model_id=None):
    module_name, _ = BACKENDS[name]
    mod = importlib.import_module(module_name)
    mod.load(api_key=api_key, model_id=model_id)
    return mod


def load_items(set_name, limit=None):
    eval_json, audio_root = SETS[set_name]
    with open(eval_json) as f:
        items = json.load(f)
    if limit is not None:
        items = items[:limit]
    return items, audio_root


def build_records(mod, backend_name, items, audio_root):
    """One provider record per item, keyed k<idx> (short: OpenAI custom_id
    length-caps out around voice-path lengths)."""
    _, build_fn_name = BACKENDS[backend_name]
    build = getattr(mod, build_fn_name)
    records = []
    for idx, item in enumerate(items):
        question = item["conversations"][0]["value"].replace("<audio>", "").strip()
        clip = os.path.join(audio_root, item["voice"][0])
        records.append(build(f"k{idx}", question, [clip], LETTERS))
    return records


def chunk_count(records, chunk_bytes):
    """Same greedy packing math as the backends' submit_batch — used for the
    dry-run plan without building chunk files."""
    chunks = cur = 0
    cur_bytes = 0
    for rec in records:
        n = len(json.dumps(rec)) + 1
        if cur and cur_bytes + n > chunk_bytes:
            chunks += 1
            cur, cur_bytes = 0, 0
        cur += 1
        cur_bytes += n
    if cur:
        chunks += 1
    return chunks


def out_paths(args, set_name):
    safe = args.backend.replace(".", "_").replace("-", "_")
    out_name = args.out_name or f"results_{safe}"
    if args.model_id and not args.out_name:
        mangled = "".join(c if c.isalnum() else "_" for c in args.model_id).strip("_")
        out_name += f"__{mangled}"
    out_jsonl = os.path.join(RESULTS, f"{out_name}{'' if set_name == 'base' else '_perturb'}.jsonl")
    jobs_file = os.path.join(JOBS_DIR, f"{out_name}.{set_name}.jobs.json")
    return out_jsonl, jobs_file


def sets_to_run(args):
    return ["base", "perturb"] if args.set == "both" else [args.set]


# ------------------------------------------------------------------ submit --

def cmd_submit(args):
    os.makedirs(JOBS_DIR, exist_ok=True)
    for set_name in sets_to_run(args):
        items, audio_root = load_items(set_name, args.limit)
        mod = load_backend(args.backend, args.api_key, args.model_id)
        records = build_records(mod, args.backend, items, audio_root)
        payload_mb = sum(len(json.dumps(r)) for r in records) / 1e6
        cap_mb = mod.BATCH_CHUNK_BYTES / 1e6
        n_chunks = chunk_count(records, mod.BATCH_CHUNK_BYTES)
        model_id = mod._state.get("model_id") if hasattr(mod, "_state") else None
        print(f"[{args.backend}/{set_name}] model={model_id} items={len(records)} "
              f"payload={payload_mb:.0f}MB -> {n_chunks} chunk(s) of <= {cap_mb:.0f}MB "
              f"(50%-off batch tier)")
        if args.dry_run:
            print(f"[{args.backend}/{set_name}] dry-run: nothing submitted")
            continue

        out_jsonl, jobs_file = out_paths(args, set_name)
        # an --out-name with a subdirectory puts both files under dirs that may
        # not exist yet -- create them BEFORE submitting, not after (this run
        # already learned that the hard way)
        os.makedirs(os.path.dirname(jobs_file), exist_ok=True)
        os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
        chunk_dir = os.path.join(JOBS_DIR, os.path.splitext(os.path.basename(jobs_file))[0])
        job_ids = mod.submit_batch(records, chunk_dir)
        state = {
            "backend": args.backend,
            "model_id": model_id,
            "set": set_name,
            "n_items": len(records),
            "jobs": job_ids,
            "out": out_jsonl,
            "submitted_at": _now(),
        }
        with open(jobs_file, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[{args.backend}/{set_name}] jobs {job_ids} recorded -> {jobs_file}")
        print(f"[{args.backend}/{set_name}] results will land in {out_jsonl}")

        if args.wait:
            wait_and_collect(mod, state, jobs_file, args.poll_interval)


# --------------------------------------------------------- status / collect --

def _jobs_files(args):
    if args.jobs_file:
        return [args.jobs_file]
    if not os.path.isdir(JOBS_DIR):
        raise SystemExit("no recorded batch jobs — run `submit` first")
    files = sorted(f for f in os.listdir(JOBS_DIR) if f.endswith(".jobs.json"))
    if not files:
        raise SystemExit(f"no *.jobs.json under {JOBS_DIR} — run `submit` first")
    return [os.path.join(JOBS_DIR, f) for f in files]


def cmd_status(args):
    for jobs_file in _jobs_files(args):
        with open(jobs_file) as f:
            state = json.load(f)
        if args.backend and state["backend"] != args.backend:
            continue
        mod = load_backend(state["backend"], args.api_key)
        statuses = mod.poll_jobs(state["jobs"])
        print(f"{os.path.basename(jobs_file)} [{state['backend']} {state['model_id']} "
              f"set={state['set']} n={state['n_items']} submitted={state['submitted_at']}]")
        for jid, st in statuses.items():
            print(f"  {jid}: {st}{'  (terminal)' if _norm_state(st) in _TERMINAL else ''}")


def wait_and_collect(mod, state, jobs_file, poll_interval):
    print(f"[{state['backend']}/{state['set']}] polling every {poll_interval}s "
          f"(Ctrl-C safe — rerun collect later; jobs live server-side)")
    while True:
        statuses = mod.poll_jobs(state["jobs"])
        line = ", ".join(f"{_norm_state(s)}" for s in statuses.values())
        print(f"  {_now()}  {line}", flush=True)
        if all(_norm_state(s) in _TERMINAL for s in statuses.values()):
            break
        time.sleep(poll_interval)
    collect_state(mod, state, jobs_file)


def score_answer(mod, backend_name, response):
    """Provider response -> (logprobs, raw_text, channel)."""
    if backend_name == "gemini-flash":
        return mod.score_response_dict(response, LETTERS)
    return mod.score_choice_dict(response, LETTERS)


def collect_state(mod, state, jobs_file):
    backend_name = state["backend"]
    statuses = mod.poll_jobs(state["jobs"])
    bad = {j: s for j, s in statuses.items() if _norm_state(s) not in _SUCCEEDED}
    if bad:
        print(f"WARNING [{backend_name}/{state['set']}]: not-succeeded jobs (skipped, their "
              f"items will count as missing): {bad}")
    answers = mod.fetch_jobs(state["jobs"])

    items, _ = load_items(state["set"], state["n_items"])
    eval_json, _ = SETS[state["set"]]

    # Merge base: existing rows in the results file (if any) survive unless
    # this batch answers the same item — batch answers win, live-filled rows
    # for items the batch never answered are kept.
    out_path = state["out"]
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        existing[r["voice"]] = r
                    except json.JSONDecodeError:
                        pass

    per_task = {}
    channels = {}
    rows = []
    seen_voices = set()
    n_new = n_updated = n_missing = 0
    for idx, item in enumerate(items):
        voice = item["voice"][0]
        seen_voices.add(voice)
        resp = answers.get(f"k{idx}")
        if resp is None:
            n_missing += 1
            if voice in existing:
                rows.append(existing[voice])  # keep a live-filled row
            continue
        logprobs, raw_text, channel = score_answer(mod, backend_name, resp)
        predicted = max(logprobs, key=logprobs.get)
        ranked = sorted(logprobs.values(), reverse=True)
        margin = (ranked[0] - ranked[1]) if len(ranked) > 1 else None
        gold = item["conversations"][1]["value"].strip().upper()
        correct = predicted == gold
        task = voice.split("/")[0]
        row = {
            "voice": voice, "task": task,
            "question": item["conversations"][0]["value"].replace("<audio>", "").strip(),
            "gold": gold, "raw_text": raw_text if isinstance(raw_text, str) else predicted,
            "predicted": predicted, "correct": correct,
            "logprob_A": logprobs["A"], "logprob_B": logprobs["B"], "logprob_margin": margin,
            "logprob_predicted": predicted, "logprob_correct": correct,
            "backend": backend_name, "model_id": state["model_id"], "channel": channel,
        }
        if voice in existing:
            n_updated += 1
        else:
            n_new += 1
        rows.append(row)
        channels[channel] = channels.get(channel, 0) + 1

    # rows not covered by this batch or existing -> nothing; write file in eval order
    n_extra_kept = 0
    for voice, r in existing.items():
        if voice not in seen_voices:
            rows.append(r)   # e.g. live-filled items beyond a --limit'd batch's range
            n_extra_kept += 1
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    for r in rows:
        s = per_task.setdefault(r["task"], {"n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += int(r["correct"])

    print(f"\ncollected {len(rows)} items -> {out_path}")
    print(f"  answered: {n_new} new + {n_updated} updated; missing/errored: {n_missing} "
          f"({n_extra_kept} pre-existing out-of-range rows kept)")
    if channels:
        print(f"  channels: {channels}  (enum = constrained output was the letter; "
              f"parse = word-bounded letter; tied = unusable, scored wrong)")
    if n_missing:
        print(f"  to fill the {n_missing} missing items live:\n"
              f"    python -m musiclistenbench.eval.run_probe_mm --backend {backend_name} "
              f"[--model-id {state['model_id']}] --resume --eval-json {eval_json} "
              f"--audio-root {SETS[state['set']][1]} --out {out_path}")

    print(f"\n{backend_name} ({state['model_id']}) batch results [{state['set']}]:")
    print(f"{'task':<12} {'n':>5} {'acc':>10}")
    tn = tc = 0
    for task, s in sorted(per_task.items()):
        print(f"{task:<12} {s['n']:>5} {s['correct'] / s['n']:>10.3f}")
        tn += s["n"]
        tc += s["correct"]
    print(f"{'overall':<12} {tn:>5} {(tc / tn if tn else float('nan')):>10.3f}")

    if state["set"] == "perturb" and rows:
        print("\n=== flip/stay metrics ===")
        subprocess.run([sys.executable, "-m", "musiclistenbench.scoring.score_perturb", out_path],
                       cwd=ROOT, check=False)


def cmd_collect(args):
    for jobs_file in _jobs_files(args):
        with open(jobs_file) as f:
            state = json.load(f)
        if args.backend and state["backend"] != args.backend:
            continue
        mod = load_backend(state["backend"], args.api_key)
        if args.wait:
            wait_and_collect(mod, state, jobs_file, args.poll_interval)
        else:
            collect_state(mod, state, jobs_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p, backend_required):
        # status/collect read the backend from each recorded jobs-state file, not this flag —
        # required only for submit (which has no jobs file yet to read it from).
        p.add_argument("--backend", required=backend_required, choices=sorted(BACKENDS))
        p.add_argument("--api-key", default=None,
                       help="provider API key (prefer the env var: GEMINI_API_KEY/"
                            "GOOGLE_API_KEY, OPENAI_API_KEY)")
        p.add_argument("--model-id", default=None,
                       help="override the backend's default model (baked into every request "
                            "and recorded; OpenAI batch files are single-model by construction)")
        p.add_argument("--jobs-file", default=None,
                       help="explicit jobs-state file (default: every *.jobs.json under "
                            "results/paid_batch_jobs/)")

    p = sub.add_parser("submit", help="build + submit batch job(s)")
    common(p, backend_required=True)
    p.add_argument("--set", default="both", choices=["base", "perturb", "both"])
    p.add_argument("--limit", type=int, default=None, help="cap items (smoke submissions)")
    p.add_argument("--out-name", default=None,
                   help="results base name under results/ (default results_<backend>"
                        "[__<mangled model id>]; perturb adds _perturb)")
    p.add_argument("--dry-run", action="store_true",
                   help="build records + print the chunk plan, submit nothing (works keyless)")
    p.add_argument("--wait", action="store_true", help="after submitting, poll and collect")
    p.add_argument("--poll-interval", type=int, default=60)

    p = sub.add_parser("status", help="print provider status of recorded jobs")
    common(p, backend_required=False)

    p = sub.add_parser("collect", help="download+score finished jobs into the results file")
    common(p, backend_required=False)
    p.add_argument("--wait", action="store_true",
                   help="poll until ALL jobs are terminal, then collect")
    p.add_argument("--poll-interval", type=int, default=60)

    args = parser.parse_args()
    if args.cmd == "submit":
        cmd_submit(args)
    elif args.cmd == "status":
        cmd_status(args)
    else:
        cmd_collect(args)


if __name__ == "__main__":
    main()
