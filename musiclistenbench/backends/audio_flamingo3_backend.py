"""Local backend: NVIDIA Audio Flamingo 3, scored by order-averaged
log-probability (Appendix A.6, A.8), mirroring qwen_omni_backend.py.

Natively supported by transformers (AudioFlamingo3ForConditionalGeneration,
added in a recent release) -- no repo clone or custom code needed, just an
up-to-date `transformers` install.

Verified quirk (2026-07, transformers 5.14.1): unlike Qwen2-Audio/Fun-Audio-
Chat, AF3's processor only accepts *one* audio per text sample --
`validate_inputs` raises `ValueError` if `len(text) != len(audio)`, and its
own chat template collapses any number of `{"type": "audio"}` content items
into a single `<sound>` placeholder anyway. Pair trials (found_pair,
harmony, stem_surgery -- anything comparing two clips) therefore get their
two clips concatenated (with a short silence gap) into one waveform and fed
through that single placeholder, rather than presented as two distinct
clips the way every other backend here does. That's a real semantic
difference for this model specifically, not a bug -- keep it in mind when
comparing AF3's pair-trial numbers against the other backends.

Second verified quirk: the checkpoint's audio tower is bf16 everywhere
except `embed_positions` (float32, per `_keep_in_fp32_modules_strict`); its
own forward() adds that fp32 tensor into the bf16 conv output without an
explicit cast, which throws a dtype mismatch a few layers later in bf16
mode. Loading in float32 (default here) sidesteps it entirely -- an 8B
model in fp32 is ~30GB, well within any of this project's GPUs, so there's
no real reason to fight for bf16 here.
"""

import torch
import torch.nn.functional as F

MODEL_ID = "nvidia/audio-flamingo-3-hf"

_state = {}


def load(device="cuda:0", dtype=torch.float32):
    if _state:
        return _state
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=device
    )
    model.eval()
    _state["processor"] = processor
    _state["model"] = model
    _state["device"] = device
    return _state


def sample_rate():
    return load()["processor"].feature_extractor.sampling_rate


def build_prompt_text(question):
    processor = load()["processor"]
    # AF3's own chat template renders exactly one <sound> tag no matter how
    # many audio content items are listed (it only checks whether any are
    # present), which matches the processor only ever accepting one audio
    # array per text sample -- see module docstring.
    content = [{"type": "audio", "path": "placeholder"}, {"type": "text", "text": question}]
    conversation = [{"role": "user", "content": content}]
    return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


def _common_prefix_len(ids_a, ids_b):
    n = min(len(ids_a), len(ids_b))
    i = 0
    while i < n and ids_a[i] == ids_b[i]:
        i += 1
    return i


@torch.no_grad()
def _answer_logprob(prompt_text, audio, candidate_word):
    st = load()
    processor, model, device = st["processor"], st["model"], st["device"]

    inputs_prompt = processor(text=prompt_text, audio=[audio], return_tensors="pt")
    full_text = prompt_text + " " + candidate_word
    inputs_full = processor(text=full_text, audio=[audio], return_tensors="pt")

    ids_prompt = inputs_prompt["input_ids"][0]
    ids_full = inputs_full["input_ids"][0]
    prefix_len = _common_prefix_len(ids_prompt, ids_full)
    targets = ids_full[prefix_len:]
    if len(targets) == 0:
        raise RuntimeError(f"candidate {candidate_word!r} tokenized to nothing new past the prompt prefix")

    inputs_full = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs_full.items()}
    out = model(**inputs_full)
    logits = out.logits[0]  # (seq, vocab)

    logprob = 0.0
    for i, tgt in enumerate(targets):
        pos = prefix_len - 1 + i
        lp = F.log_softmax(logits[pos].float(), dim=-1)[tgt.item()]
        logprob += lp.item()
    return logprob


def score_options(question, clip_paths, options, sr=None):
    """clip_paths: list of wav paths (1 or 2). options: (word_a, word_b).
    Returns {word: logprob}."""
    import numpy as np

    from musiclistenbench.backends import audio_io

    sr = sr or sample_rate()
    clips = [audio_io.load_mono(p, sr=sr) for p in clip_paths]
    if len(clips) > 1:
        gap = np.zeros(int(0.3 * sr), dtype=clips[0].dtype)
        audio = np.concatenate([clips[0], gap, clips[1]])
    else:
        audio = clips[0]
    prompt_text = build_prompt_text(question)
    return {opt: _answer_logprob(prompt_text, audio, opt) for opt in options}
