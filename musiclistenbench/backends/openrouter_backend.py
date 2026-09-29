"""Paid API backend: OpenRouter (any audio-capable chat model, default
google/gemini-3.1-pro-preview), scored by GENERATED LETTER -- same letter
mode as gpt_audio_backend (temp-0 generation + word-bounded parse), over the
OpenAI-compatible endpoint https://openrouter.ai/api/v1 with the openai SDK.

Gemini-native enum constraining (text/x.enum) isn't expressible in the
OpenAI chat format, so we TRY the closest equivalent -- response_format
json_schema with a string enum -- and drop it (adaptive, one retry) if the
model/provider rejects it, falling back to plain parsing. Reasoning models:
`reasoning: {effort: low}` + a generous max_tokens headroom, since thinking
tokens count toward the completion budget and would otherwise truncate the
answer (the same trap applies to Gemini).
OpenRouter has NO Batch API -> live only (run_paid_eval.sh / run_probe_mm).

Key: OPENROUTER_API_KEY (or --api-key). Model via --model-id, e.g.
"google/gemini-3.1-pro-preview" (default) or any other OpenRouter model id.
"""

import base64
import json
import re
import time

MODEL_ID = "google/gemini-3.1-pro-preview"
BASE_URL = "https://openrouter.ai/api/v1"
MAX_OUTPUT_TOKENS = 1024   # headroom for reasoning tokens; the letter itself is ~1 token
FALLBACK_LOGPROB = -1000.0
MAX_RETRIES = 5

# Per OpenAI's structured-outputs guide, response_format json_schema is for
# GPT-4o-and-later TEXT models; audio-family models reject it outright (and
# reasoning too). Matched by substring on the model id -> plain letter mode.
_PLAIN_LETTER_SUBSTRINGS = ("audio", "realtime", "transcribe", "whisper", "tts")

_state = {}


def load(device=None, api_key=None, model_id=None):
    """`device` ignored (parity with local backends); client built lazily so
    keyless offline use (tests, --dry-run) works."""
    if _state:
        if model_id:
            _state["model_id"] = model_id
        return _state
    _state["api_key"] = api_key
    _state["model_id"] = model_id or MODEL_ID
    _state["use_enum_schema"] = True
    _state["use_reasoning"] = True
    _state["use_temperature"] = True
    _state["max_tokens_param"] = "max_tokens"
    _state["client"] = None
    return _state


def _plain_letter_model():
    """True for model families that reject response_format/reasoning params
    (audio models per OpenAI's own docs) -- skip those knobs from request #1."""
    mid = (_state.get("model_id") or "").lower()
    return any(s in mid for s in _PLAIN_LETTER_SUBSTRINGS)


def _ensure_client():
    if _state.get("client") is None:
        import os
        from openai import OpenAI
        # Explicit --api-key wins; else read OPENROUTER_API_KEY directly. The
        # SDK will NOT pick it up on its own here — with base_url overridden it
        # looks for OPENAI_API_KEY, and an unset key sends no Authorization
        # header at all (live-confirmed 401 "Missing Authentication header").
        key = _state.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (or pass --api-key)")
        _state["client"] = OpenAI(base_url=BASE_URL, api_key=key)
    return _state["client"]


def _request_kwargs(options):
    kw = {_state.get("max_tokens_param", "max_tokens"): MAX_OUTPUT_TOKENS}
    if _state["use_temperature"]:
        kw["temperature"] = 0.0
    if _state["use_enum_schema"] and not _plain_letter_model():
        kw["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True,
                            "schema": {"type": "string", "enum": list(options)}},
        }
    if _state["use_reasoning"] and not _plain_letter_model():
        kw["reasoning"] = {"effort": "low"}
    return kw


def _is_non_retryable(exc):
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status != 429:
        return True
    msg = str(exc).lower()
    return any(m in msg for m in ("model_not_found", "not found", "unauthorized",
                                  "invalid_request_error", "no endpoints"))
def _error_text(exc):
    """Full error text incl. OpenRouter's nested metadata.raw (the provider's
    own message), which plain str(exc) may truncate or re-escape."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        try:
            import json as _json
            parts.append(_json.dumps(body))
        except Exception:
            parts.append(repr(body))
    return " ".join(parts).lower()


def _drop_all_optional():
    """Nuke every optional request knob. Used when a provider 400 rejects one
    of them: dropping one-at-a-time can still crash if the SAME retry trips on
    the next knob and the provider echoes the first error's text (live-observed
    with openai/gpt-audio-mini: 'response_format' rejection re-raised after the
    schema was already dropped). One clean retry with a minimal request is
    safer than diagnosing which knob the provider actually meant."""
    dropped = []
    for knob in ("use_enum_schema", "use_reasoning", "use_temperature"):
        if _state.get(knob):
            _state[knob] = False
            dropped.append(knob)
    return dropped


def _adapt(exc):
    """Drop optional constraint knobs when rejected; True = retry now.
    Matches on the NESTED provider message too (_error_text), not just str(exc)."""
    msg = _error_text(exc)
    dropped = []
    if ("response_format" in msg or "json_schema" in msg) and _state["use_enum_schema"]:
        _state["use_enum_schema"] = False
        dropped.append("response_format")
    if ("reasoning" in msg or "thinking" in msg) and _state["use_reasoning"]:
        _state["use_reasoning"] = False
        dropped.append("reasoning")
    if "temperature" in msg and _state["use_temperature"]:
        _state["use_temperature"] = False
        dropped.append("temperature")
    if ("max_completion_tokens" in msg
            and _state.get("max_tokens_param") != "max_completion_tokens"):
        _state["max_tokens_param"] = "max_completion_tokens"
        dropped.append("max_tokens->max_completion_tokens")
    if dropped:
        print(f"WARNING [openrouter]: provider rejected {dropped}; retrying without them",
              flush=True)
        return True
    # a param-rejection 400 that names no knob we sent -> drop everything once
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status == 400 and "invalid" in msg and "parameter" in msg:
        if _drop_all_optional():
            print("WARNING [openrouter]: unnamed parameter rejection (400); retrying with "
                  "a minimal request", flush=True)
            return True
    return False


def _call_with_retry(client, model_id, messages, options):
    delay = 1
    last_exc = None
    for attempt in range(MAX_RETRIES * 2):
        try:
            return client.chat.completions.create(
                model=model_id, messages=messages, **_request_kwargs(options))
        except Exception as exc:
            last_exc = exc
            if _adapt(exc):
                continue
            if _is_non_retryable(exc):
                raise
            print(f"[openrouter] API call failed (attempt {attempt + 1}/{MAX_RETRIES * 2}): "
                  f"{exc!r}; retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise last_exc


def _audio_content(question, clip_paths):
    content = []
    for p in clip_paths:
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        content.append({"type": "input_audio", "input_audio": {"data": data, "format": "wav"}})
    content.append({"type": "text", "text": question})
    return content


def _parse_letter(raw, options):
    """(scores, channel): word-bounded letter -> 'parse'; nothing -> 'tied'.
    json_schema-constrained replies may arrive quoted ('\"A\"') -- strip quotes."""
    raw = (raw or "").strip().strip('"').strip()
    bounded = re.compile("|".join(rf"\b{re.escape(o)}\b" for o in options), re.IGNORECASE)
    m = bounded.search(raw)
    if m is None:
        print(f"WARNING [openrouter]: no letter parseable from {raw!r}; "
              f"tied scores", flush=True)
        return {o: FALLBACK_LOGPROB for o in options}, "tied"
    hit = next(o for o in options if o.lower() == m.group(0).lower())
    return {o: (0.0 if o == hit else FALLBACK_LOGPROB) for o in options}, "parse"


def score_options(question, clip_paths, options, sr=None):
    st = load()
    client, model_id = _ensure_client(), st["model_id"]
    resp = _call_with_retry(client, model_id,
                            [{"role": "user", "content": _audio_content(question, clip_paths)}],
                            options)
    if not resp.choices:
        print("WARNING [openrouter]: no choices in response (possibly filtered); tied scores",
              flush=True)
        return {o: FALLBACK_LOGPROB for o in options}
    return _parse_letter(resp.choices[0].message.content, options)[0]
