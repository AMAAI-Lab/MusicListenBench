"""Paid API backend: OpenAI audio-capable GPT (gpt-audio-1.5), scored by
GENERATED LETTER (temperature-0 generation + word-bounded parse), exposing
the same `score_options(question, clip_paths, options)` interface as the
local backends (see qwen2_audio_backend.py) so it drops straight into
run_probe_mm.py alongside every other backend.

No logprobs, by decision (2026-09-25): the Gemini backend's live smoke hit
`400 INVALID_ARGUMENT. 'Logprobs is not enabled for this model'` on
gemini-3.8-flash, and the call was made to drop logprob scoring from BOTH
paid backends so the two rows share one methodology. Chat Completions has
no enum-constrained output for audio models (the Gemini backend's trick),
so this backend asks for a short temperature-0 completion of the
one-letter answer and scores by a WORD-BOUNDED letter parse: an 'a' inside
'answer' or a 'b' inside 'but' never wins, and prose with no standalone
letter gets the tied sentinel (scored wrong by construction -- that IS the
failure mode being measured, the same policy applies to every paid model).

MAX_OUTPUT_TOKENS=16 (not 5): room for a chatty preamble ("The answer is:
A") so the token budget can't truncate the answer away before the letter
appears; output cost stays negligible because temperature-0 letter answers
are typically 1 token anyway.

The {option: logprob} return shape keeps run_probe_mm.py's interface intact:
the parsed letter gets 0.0, the other the FALLBACK_LOGPROB sentinel, so the
runner's argmax reproduces the parsed letter (and logprob_margin is a
constant, meaningless -- read `correct`, not the margin). Batch rows
additionally record HOW each answer was read via the `channel` field:
'parse' (word-bounded letter found) or 'tied' (nothing usable).

Adaptive request knobs (`--model-id` friendliness): newer OpenAI models keep
tightening parameter rules (some reject `max_tokens` in favour of
`max_completion_tokens`, some reject any `temperature` != 1). Rather than
hard-failing on the first such response, `_call_with_retry` adapts the
request shape once per incompatibility and retries immediately; genuinely
non-retryable client errors (400 bad model id, 401 auth, ...) fail fast
instead of burning the backoff budget, and only transient errors (rate
limit, 5xx, network) consume the exponential-backoff retries.

Batch mode (50% off, 24h turnaround): `build_batch_line` / `submit_batch` /
`poll_jobs` / `fetch_jobs` follow OpenAI's Batch API --
inline base64 audio in the JSONL (chat completions has no
file-reference form), greedily packed into <=180MB chunk files under the
Batch API's 200MB upload cap, one batch job per chunk, `submit_batch`
returning the LIST of job ids. Per the Batch API docs, each line's `body`
takes "the same parameters as the underlying endpoint" -- so modalities /
temperature / the token-budget param round-trip, and fetch_jobs scores the
downloaded `choices[0].message.content` with the SAME word-bounded letter
parse as live mode (one metric, both modes). NOTE: the adaptive knobs can't
fire inside a batch job (validation is server-side and all-or-nothing per
file), so smoke-test live (`run_paid_eval.sh --limit 8`) with the exact
--model-id you'll batch, first. Batch rows preserve eval-set order by joining
on a short `k<idx>` custom_id (voice paths can exceed the custom_id length
cap).

Smoke-test on a few items before a full run:
    python -m musiclistenbench.eval.run_probe_mm --backend gpt-audio \\
        --eval-json data/eval.json --audio-root data/audio \\
        --limit 8 --out /tmp/gpt_audio_smoke.jsonl
Needs `pip install openai` (in requirements.txt) and an OPENAI_API_KEY env
var -- or pass --api-key on the command line, though the env var is
preferable (a CLI flag is visible in shell history and process listings).

Default-model VERIFY note: "gpt-audio-1.5" is the audio-input model OpenAI's
own "Audio in Chat Completions" guide canonicalizes as of 2026-09
(developers.openai.com/api/docs/guides/audio-chat-completions) — successor of
the gpt-4o-audio-preview line. Model
names turn over fast; re-check the docs if this looks stale, and override
with --model-id rather than editing this file (the adaptive knobs above are
what make that safe).
"""

import base64
import json
import re
import time
from pathlib import Path

MODEL_ID = "gpt-audio-1.5"
MAX_OUTPUT_TOKENS = 16      # room for a chatty preamble before the letter; T=0 answers are ~1 token
FALLBACK_LOGPROB = -1000.0  # sentinel score for the non-parsed letter (and for 'tied' rows)
MAX_RETRIES = 5
BATCH_CHUNK_BYTES = 180_000_000   # stay under OpenAI Batch's 200MB input-file cap with headroom

_state = {}


def load(device=None, api_key=None, model_id=None):
    """`device` is accepted (and ignored) so run_probe_mm.py can call this the
    same way it calls every local backend's load(device=...). The HTTP client
    is constructed lazily by _ensure_client(), so keyless offline uses
    (building batch records, --dry-run) work without OPENAI_API_KEY set."""
    if _state:
        if model_id:
            _state["model_id"] = model_id
        return _state
    _state["api_key"] = api_key
    _state["model_id"] = model_id or MODEL_ID
    # Adaptive request knobs, flipped by _adapt() when this model rejects them.
    # Batch jobs can't adapt mid-file (validation is all-or-nothing), so the
    # param name is env-overridable for models that want max_completion_tokens:
    #   export GPT_AUDIO_MAX_TOKENS_PARAM=max_completion_tokens
    import os as _os
    _state["max_tokens_param"] = _os.environ.get("GPT_AUDIO_MAX_TOKENS_PARAM", "max_tokens")
    _state["use_temperature"] = True
    _state["client"] = None   # built on first network use
    return _state


def _ensure_client():
    if _state.get("client") is None:
        from openai import OpenAI
        _state["client"] = (OpenAI(api_key=_state.get("api_key")) if _state.get("api_key")
                            else OpenAI())   # reads OPENAI_API_KEY
    return _state["client"]


def _request_kwargs():
    kw = {"modalities": ["text"], _state["max_tokens_param"]: MAX_OUTPUT_TOKENS}
    if _state["use_temperature"]:
        kw["temperature"] = 0.0   # determinism: nothing reads logprobs any more
    return kw


def _is_non_retryable(exc):
    """4xx client errors (except 429 rate limits) will never succeed on
    retry -- fail fast instead of burning the backoff budget on each item."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status != 429:
        return True
    msg = str(exc).lower()
    return any(m in msg for m in ("invalid_request_error", "invalid argument",
                                  "model_not_found", "not found", "unauthorized"))


def _adapt(exc):
    """True if the request shape can be fixed and retried immediately (a
    parameter this model doesn't accept); False for transient/unknown errors."""
    msg = str(exc).lower()
    if "max_completion_tokens" in msg and _state["max_tokens_param"] != "max_completion_tokens":
        _state["max_tokens_param"] = "max_completion_tokens"
        return True
    if "temperature" in msg and _state["use_temperature"]:
        _state["use_temperature"] = False
        return True
    return False


def _letter_scores(text, options):
    """Alias kept for symmetry with the gemini backend's naming; see
    _parse_fallback_dict (the actual implementation)."""
    return _parse_fallback_dict(text, options)


def _parse_fallback_dict(raw, options):
    """THE scoring path (name kept from the logprob era): generated text ->
    {option: score} via a word-bounded letter match. An 'a' inside 'answer'
    or a 'b' inside 'but' never wins; prose with no standalone letter gets
    the tied sentinel (scored wrong; that IS the failure mode being
    measured)."""
    raw = (raw or "").strip()
    bounded = re.compile("|".join(rf"\b{re.escape(o)}\b" for o in options), re.IGNORECASE)
    m = bounded.search(raw)
    if m is None:
        print(f"WARNING [gpt-audio]: no letter parseable from {raw!r}; "
              f"falling back to tied scores for all options", flush=True)
        return {opt: FALLBACK_LOGPROB for opt in options}, "tied"
    hit = next(o for o in options if o.lower() == m.group(0).lower())
    return {o: (0.0 if o == hit else FALLBACK_LOGPROB) for o in options}, "parse"


def _call_with_retry(client, model_id, messages):
    delay = 1
    last_exc = None
    for attempt in range(MAX_RETRIES * 2):   # headroom: adaptations retry immediately too
        try:
            return client.chat.completions.create(
                model=model_id, messages=messages, **_request_kwargs())
        except Exception as exc:   # rate limits / transient 5xx / param rejections
            last_exc = exc
            if _adapt(exc):
                continue
            if _is_non_retryable(exc):
                raise   # e.g. 400 bad model id -- permanent, retrying is pure waste
            print(f"[gpt-audio] API call failed (attempt {attempt + 1}/{MAX_RETRIES * 2}): "
                  f"{exc!r}; retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise last_exc


def score_options(question, clip_paths, options, sr=None):
    """clip_paths: list of wav paths (1 item for this probe -- build_probe.py
    already concatenates each trial's two clips into one combined wav).
    options: e.g. ("A", "B"). Returns {option: score}: the parsed letter
    gets 0.0, the other the sentinel, so the runner's argmax reproduces the
    parsed letter. `sr` is accepted for interface parity with the local
    backends; unused -- the API reads the wav file's own header, no
    resampling needed."""
    st = load()
    client, model_id = _ensure_client(), st["model_id"]

    content = _audio_content(question, clip_paths)
    resp = _call_with_retry(client, model_id, [{"role": "user", "content": content}])
    if not resp.choices:
        print("WARNING [gpt-audio]: no choices in response (possibly safety-filtered); "
              "falling back to tied scores for all options", flush=True)
        return {opt: FALLBACK_LOGPROB for opt in options}
    return _parse_fallback_dict(resp.choices[0].message.content, options)[0]


def _parse_fallback(choice, options):
    """Kept for callers that hand a live choice object (unused internally
    since the letter-mode pivot; batch uses _parse_fallback_dict)."""
    return _parse_fallback_dict(choice.message.content, options)[0]


# ---------------------------------------------------------------- batch ----

def _audio_content(question, clip_paths):
    """The user message content list shared by live + batch request shapes."""
    content = []
    for p in clip_paths:
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        content.append({"type": "input_audio", "input_audio": {"data": data, "format": "wav"}})
    content.append({"type": "text", "text": question})
    return content


def score_choice_dict(choice_dict, options):
    """Batch-mode twin of the live scoring: one downloaded choices[0] dict ->
    (scores dict, raw_text, channel) via the SAME word-bounded letter parse
    as the live path (one metric, both modes). Channel: 'parse' (letter
    found) or 'tied' (nothing usable -- scored wrong by construction)."""
    text = (choice_dict.get("message") or {}).get("content")
    parsed, channel = _parse_fallback_dict(text, options)
    return parsed, text, channel


def build_batch_line(custom_id, question, clip_paths, options, sr=None):
    """One Batch-API JSONL line for one item. Body params mirror the live
    call exactly (modalities/text, temperature=0, small token budget, NO
    logprobs -- see module docstring) per "the parameters in each line's
    body field are the same as the parameters for the underlying endpoint".
    NOTE: unlike live mode, nothing can adapt mid-batch (validation is
    all-or-nothing per file) -- smoke-test live with the same --model-id
    first."""
    body = {
        "model": _state["model_id"],
        "modalities": ["text"],
        _state.get("max_tokens_param", "max_tokens"): MAX_OUTPUT_TOKENS,
        "messages": [{"role": "user", "content": _audio_content(question, clip_paths)}],
    }
    if _state.get("use_temperature", True):
        body["temperature"] = 0.0
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
            "body": body}


def submit_batch(lines, jsonl_dir, chunk_bytes=BATCH_CHUNK_BYTES):
    """Pack lines into <=chunk_bytes JSONL chunk files (200MB Batch upload
    cap) and submit one batch job per chunk. Returns the LIST of batch ids.
    `lines` are the dicts from build_batch_line."""
    client = _ensure_client()
    jsonl_dir = Path(jsonl_dir)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    batch_ids = []
    chunk, chunk_bytes_used, chunk_i = [], 0, 0

    def flush():
        nonlocal chunk, chunk_bytes_used, chunk_i
        if not chunk:
            return
        path = jsonl_dir / f"chunk{chunk_i}.jsonl"
        with open(path, "w") as f:
            f.writelines(l + "\n" for l in chunk)
        up = client.files.create(file=open(path, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
        batch_ids.append(batch.id)
        print(f"[gpt-audio] submitted chunk {chunk_i} ({len(chunk)} requests) -> {batch.id}",
              flush=True)
        chunk, chunk_bytes_used, chunk_i = [], 0, chunk_i + 1

    for line_dict in lines:
        line = json.dumps(line_dict)
        if chunk and chunk_bytes_used + len(line) > chunk_bytes:
            flush()
        chunk.append(line)
        chunk_bytes_used += len(line) + 1
    flush()
    return batch_ids


def _print_batch_errors(batch, batch_id):
    """Surface a failed batch's own error object (validation failures land
    here: 'failed' status = the input file was rejected before any request
    ran, so nothing was billed)."""
    errs = getattr(batch, "errors", None)
    items = list(getattr(errs, "data", None) or []) if errs else []
    if items:
        for e in items[:5]:
            print(f"[gpt-audio] batch {batch_id} error: code={e.code} param={e.param} "
                  f"message={str(e.message)[:300]}", flush=True)
    else:
        print(f"[gpt-audio] batch {batch_id} failed with no error object "
              f"(check dashboard or error_file_id)", flush=True)


def poll_jobs(batch_ids):
    """{batch_id: status} using the OpenAI Batch API's own status values
    (validating / in_progress / finalizing / completed / failed / expired
    / cancelling / cancelled)."""
    client = _ensure_client()
    out = {}
    for bid in batch_ids:
        batch = client.batches.retrieve(bid)
        out[bid] = str(batch.status)
        if out[bid] == "failed":
            _print_batch_errors(batch, bid)
    return out


def fetch_jobs(batch_ids):
    """Download + parse finished jobs. Returns {custom_id: choice_dict or
    None}: the raw choices[0] dict per request (score_choice_dict does the
    letter reading), None for requests the provider reports as errored.
    Failed / in-progress jobs contribute nothing; the caller decides how to
    handle absent keys."""
    client = _ensure_client()
    results = {}
    for batch_id in batch_ids:
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            print(f"[gpt-audio] batch {batch_id} not completed (status={batch.status}); skipped",
                  flush=True)
            continue
        for file_id, is_error in ((batch.output_file_id, False), (batch.error_file_id, True)):
            if not file_id:
                continue
            text = client.files.content(file_id).text
            for line in text.splitlines():
                rec = json.loads(line)
                cid = rec.get("custom_id")
                if is_error or rec.get("error") or not rec.get("response"):
                    results[cid] = None
                    continue
                choices = (rec["response"].get("body") or {}).get("choices") or []
                results[cid] = choices[0] if choices else None
    return results
