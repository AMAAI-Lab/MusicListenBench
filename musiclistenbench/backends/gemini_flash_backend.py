"""Paid API backend: Google Gemini (Flash tier), scored by CONSTRAINED
LETTER ANSWER (enum-constrained decoding), exposing the same
`score_options(question, clip_paths, options)` interface as the local
backends (see qwen2_audio_backend.py) so it drops straight into
run_probe_mm.py with no changes to that file's per-item loop,
merge_results.py, or score_perturb.py.

Why constrained decoding, not logprobs: gemini-3.8-flash rejects logprob
parameters outright -- confirmed live (2026-09-25): response_logprobs=True
returns `400 INVALID_ARGUMENT. 'Logprobs is not enabled for this model'`.
The same wall was hit earlier ("the closed Gemini API has no
teacher-forced logprob equivalent") and its solution is reused verbatim
here: every call constrains decoding to the task's exact option set via
response_mime_type="text/x.enum" + an enum response_schema -- the model can
ONLY emit "A" or "B", so the answer is a hard, always-valid label, never a
parse artefact. temperature=0 (determinism matters again now that nothing
reads logprobs) and thinking_level "low" (gemini-3.8-flash rejects "minimal":
`400 INVALID_ARGUMENT. 'Thinking level MINIMAL is not supported for this
model'`) so invisible reasoning can't eat the answer.

The {option: logprob} return shape keeps run_probe_mm.py's interface intact:
the emitted letter gets 0.0, the other the FALLBACK_LOGPROB sentinel, so the
runner's argmax reproduces the emitted letter exactly (and logprob_margin
is a constant, meaningless -- read `correct`, not the margin). Batch rows
additionally record HOW each answer was read via the `channel` field:
'enum' (constrained output was exactly the letter), 'parse' (word-bounded
regex rescue of off-spec output), or 'tied' (nothing usable -- scored wrong
by construction, which IS the failure mode being measured).

Partially verified live 2026-09-25: gemini-3.8-flash rejected the logprob
parameters with 400 INVALID_ARGUMENT "Logprobs is not enabled for this
model" (which is exactly what motivated the constrained-decoding redesign);
the enum path itself still wants its own cheap smoke run before a full pass:
    python -m musiclistenbench.eval.run_probe_mm --backend gemini-flash \\
        --eval-json data/eval.json --audio-root data/audio \\
        --limit 8 --out /tmp/gemini_flash_smoke.jsonl
Needs `pip install google-genai` (already in requirements.txt) and a
GEMINI_API_KEY or GOOGLE_API_KEY env var -- or pass --api-key on the command
line, though the env var is preferable (a CLI flag is visible in shell
history and process listings).

Batch mode (50% off, up-to-24h turnaround): `build_batch_record` /
`submit_batch` / `poll_jobs` / `fetch_jobs`. Audio goes INLINE (base64 in the
JSONL request) rather than via Files-API URIs, unlike a Files-API design --
that harness reuses a handful of clips across thousands of requests (upload
once, reference many), while MusicListenBench's every item has a UNIQUE combined
wav, so uploading would mean one Files-API upload per item anyway and inline
is strictly cheaper. Each request is ~0.7-1.1MB base64, comfortably under the
20MB inline-per-request cap; the JSONL input file caps at 2GB, so submit_batch
chunks records at BATCH_CHUNK_BYTES (default 1GB) into one batch job per chunk.
The docs allow "any request configurations you would use in a standard
non-batch request" in each JSONL request, so the enum-constrained
generation_config rides along; fetch_jobs reads the downloaded result file
with the SAME constrained-letter logic as live mode (one metric, both
modes). Result JSON key casing varies by endpoint revision (proto3 JSON
accepts both camelCase and snake_case), so the batch parser accepts either.
Rows preserve eval-set order by joining on a short `k<idx>` key.

Default-model VERIFY note: "gemini-3.8-flash" was Google's current
production Flash-tier model as of 2026-09 (GA'd 2026-09-02, per
https://ai.google.dev/gemini-api/docs/latest-model). Model names turn over
fast -- re-check https://ai.google.dev/gemini-api/docs/models if this looks
stale, and override with --model-id rather than editing this file.
"""

import base64
import json
import re
import time
from pathlib import Path

MODEL_ID = "gemini-3.8-flash"
THINKING_BUDGET = 1024     # only used for non-flash overrides; flash models get thinking_level "low"
FALLBACK_LOGPROB = -1000.0  # sentinel score for the non-emitted letter (and for 'tied' rows)
MAX_RETRIES = 5
BATCH_CHUNK_BYTES = 1_000_000_000   # 1GB, half the Batch API's 2GB input-file cap (headroom)

# google.genai raises ClientError whose str() starts with e.g. "400 INVALID_ARGUMENT."
# Retrying these is pure waste (~30s of backoff per item for a permanent error).
_NON_RETRYABLE_MARKERS = ("INVALID_ARGUMENT", "NOT_ENABLED", "UNAUTHENTICATED",
                          "PERMISSION_DENIED", "NOT_FOUND", "FAILED_PRECONDITION")

_state = {}


def load(device=None, api_key=None, model_id=None):
    """`device` is accepted (and ignored) so run_probe_mm.py can call this the
    same way it calls every local backend's load(device=...). The HTTP client
    is constructed lazily by _ensure_client(), so keyless offline uses
    (building batch records, --dry-run) work without GEMINI_API_KEY set."""
    if _state:
        if model_id:
            _state["model_id"] = model_id
        return _state
    _state["api_key"] = api_key
    _state["model_id"] = model_id or MODEL_ID
    _state["client"] = None   # built on first network use
    return _state


def _ensure_client():
    if _state.get("client") is None:
        from google import genai
        _state["client"] = (genai.Client(api_key=_state.get("api_key")) if _state.get("api_key")
                            else genai.Client())   # reads GEMINI_API_KEY / GOOGLE_API_KEY
    return _state["client"]


def _generation_config_dict(options):
    """enum-constrained output (the model can only emit one of `options`),
    temperature 0, low thinking on flash tiers (gemini-3.8-flash rejects
    "minimal" -- see module docstring). Returned as a plain dict so the SAME
    object serves the live SDK call (splat into GenerateContentConfig) and
    the batch JSONL generation_config field."""
    thinking = ("thinking_level", "low") if "flash" in _state["model_id"] \
        else ("thinking_budget", THINKING_BUDGET)
    return {
        "temperature": 0,
        "response_mime_type": "text/x.enum",
        "response_schema": {"type": "STRING", "enum": list(options)},
        "thinking_config": {thinking[0]: thinking[1]},
    }


def _gen_config(options):
    from google.genai import types
    return types.GenerateContentConfig(**_generation_config_dict(options))


def _is_non_retryable(exc):
    """4xx client errors (except 429 rate limits) will never succeed on
    retry -- fail fast instead of burning the backoff budget on each item."""
    status = getattr(exc, "code", None)
    status = getattr(status, "value", status)   # grpc status enum -> int, if it's one
    if isinstance(status, int) and 400 <= status < 500 and status != 429:
        return True
    return any(m in str(exc) for m in _NON_RETRYABLE_MARKERS)


def _call_with_retry(client, model_id, parts, config):
    delay = 1
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(model=model_id, contents=parts, config=config)
        except Exception as exc:   # rate limits / transient 5xx from a real network call
            if _is_non_retryable(exc):
                raise   # e.g. 400 INVALID_ARGUMENT -- permanent, retrying is pure waste
            last_exc = exc
            print(f"[gemini-flash] API call failed (attempt {attempt + 1}/{MAX_RETRIES}): "
                  f"{exc!r}; retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise last_exc


def _letter_scores(text, options):
    """Constrained output -> ({option: score}, channel). 'enum' when the
    text IS the letter (the normal case -- decoding is constrained to it);
    'parse' for a word-bounded regex rescue of off-spec output; 'tied' when
    nothing usable came back (scored wrong by construction)."""
    t = (text or "").strip().upper()
    if t in options:
        return {o: (0.0 if o == t else FALLBACK_LOGPROB) for o in options}, "enum"
    m = re.compile("|".join(rf"\b{re.escape(o)}\b" for o in options)).search(text or "")
    if m:
        hit = m.group(0).upper()
        return {o: (0.0 if o == hit else FALLBACK_LOGPROB) for o in options}, "parse"
    print(f"WARNING [gemini-flash]: no letter in constrained output {text!r}; "
          f"tied fallback scores", flush=True)
    return {o: FALLBACK_LOGPROB for o in options}, "tied"


def _text_from_response(resp):
    t = getattr(resp, "text", None)
    if t:
        return t
    try:
        cands = resp.candidates or []
        parts = getattr(cands[0].content, "parts", None) or []
        return " ".join(p.text for p in parts if getattr(p, "text", None))
    except Exception:
        return ""


def score_options(question, clip_paths, options, sr=None):
    """clip_paths: list of wav paths (1 item for this probe -- build_probe.py
    already concatenates each trial's two clips into one combined wav).
    options: e.g. ("A", "B"). Returns {option: score}: the emitted letter
    gets 0.0, the other the sentinel, so the runner's argmax reproduces the
    emitted letter. `sr` is accepted for interface parity with the local
    backends; unused -- the API reads the wav file's own header, no
    resampling needed."""
    from google.genai import types

    st = load()
    client, model_id = _ensure_client(), st["model_id"]

    parts = [types.Part.from_bytes(data=open(p, "rb").read(), mime_type="audio/wav")
             for p in clip_paths]
    parts.append(types.Part.from_text(text=question))

    resp = _call_with_retry(client, model_id, parts, _gen_config(options))
    if not resp.candidates:
        print("WARNING [gemini-flash]: no candidates in response (possibly safety-filtered); "
              "falling back to tied scores for all options", flush=True)
        return {opt: FALLBACK_LOGPROB for opt in options}
    scores, _channel = _letter_scores(_text_from_response(resp), options)
    return scores


# ---------------------------------------------------------------- batch ----

def _get(d, *names, default=None):
    """First present key among `names` -- proto3 JSON accepts both camelCase
    and snake_case, and different endpoint revisions emit different ones."""
    for n in names:
        if isinstance(d, dict) and n in d:
            return d[n]
    return default


def _text_from_batch_response(response_dict):
    """Visible text of one downloaded-batch response dict. Key casing varies
    by endpoint revision (proto3 JSON accepts both camelCase and snake_case)."""
    cands = _get(response_dict, "candidates", default=[]) or []
    if not cands:
        return ""
    parts = _get(cands[0], "content", default={}) or {}
    return "".join(p.get("text", "") for p in (parts.get("parts") or []))


def score_response_dict(response_dict, options):
    """Batch-mode twin of the live scoring: one downloaded response dict ->
    (scores dict, raw_text, channel) via the SAME _letter_scores logic the
    live path uses (one metric, both modes). Channel: 'enum' (constrained
    output was exactly the letter), 'parse' (regex rescue), or 'tied'."""
    text = _text_from_batch_response(response_dict)
    scores, channel = _letter_scores(text, options)
    return scores, text, channel


def build_batch_record(key, question, clip_paths, options, sr=None):
    """One Batch-API JSONL record for one item: {"key", "request"} with the
    audio inline (base64) and the SAME enum-constrained generation_config as
    the live call (_generation_config_dict(options), served as plain JSON
    here). Docs: batch requests take "any request configurations you would
    use in a standard non-batch request"."""
    parts = []
    for p in clip_paths:
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        parts.append({"inline_data": {"mime_type": "audio/wav", "data": data}})
    parts.append({"text": question})

    return {"key": key,
            "request": {
                "contents": [{"role": "user", "parts": parts}],
                "generation_config": _generation_config_dict(options),
            }}


def submit_batch(records, jsonl_dir, chunk_bytes=BATCH_CHUNK_BYTES):
    """Pack records into <=chunk_bytes JSONL chunk files (2GB Batch input-file
    cap; default chunks at 1GB for upload-speed headroom) and submit one
    batch job per chunk via the Files API. Returns the LIST of job names."""
    client = _ensure_client()
    jsonl_dir = Path(jsonl_dir)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    job_names = []
    chunk, chunk_bytes_used, chunk_i = [], 0, 0

    def flush():
        nonlocal chunk, chunk_bytes_used, chunk_i
        if not chunk:
            return
        path = jsonl_dir / f"chunk{chunk_i}.jsonl"
        with open(path, "w") as f:
            for rec in chunk:
                f.write(json.dumps(rec) + "\n")
        up = client.files.upload(file=str(path), config={"mime_type": "jsonl"})
        job = client.batches.create(model=_state["model_id"], src=up.name,
                                    config={"display_name": path.stem})
        job_names.append(job.name)
        print(f"[gemini-flash] submitted chunk {chunk_i} ({len(chunk)} requests) -> {job.name}",
              flush=True)
        chunk, chunk_bytes_used, chunk_i = [], 0, chunk_i + 1

    for rec in records:
        line = json.dumps(rec)
        if chunk and chunk_bytes_used + len(line) > chunk_bytes:
            flush()
        chunk.append(rec)
        chunk_bytes_used += len(line) + 1
    flush()
    return job_names


def poll_jobs(job_names):
    """{job_name: state} using the Batch API's own JOB_STATE_* values."""
    client = _ensure_client()
    out = {}
    for name in job_names:
        state = client.batches.get(name=name).state
        out[name] = str(getattr(state, "name", state))
    return out


def fetch_jobs(job_names):
    """Download + parse finished jobs. Returns {key: response_dict or None}
    (the raw GenerateContentResponse dict per request -- score_response_dict
    does the letter reading; None for errored requests / lines carrying a
    status object instead of a response). Not-yet-succeeded jobs contribute
    nothing; the caller decides how to handle absent keys."""
    client = _ensure_client()
    results = {}
    for name in job_names:
        job = client.batches.get(name=name)
        state = str(getattr(job.state, "name", job.state))
        if state != "JOB_STATE_SUCCEEDED":
            print(f"[gemini-flash] batch {name} not succeeded (state={state}); skipped", flush=True)
            continue
        out_file = getattr(job.dest, "file_name", None)
        if not out_file:
            print(f"[gemini-flash] batch {name} succeeded but has no result file; skipped",
                  flush=True)
            continue
        raw = client.files.download(file=out_file)
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec.get("key")
            resp = rec.get("response")
            results[key] = resp if isinstance(resp, dict) and resp.get("candidates") else None
    return results
