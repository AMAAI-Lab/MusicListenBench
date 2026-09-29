"""Local backend: Qwen2-Audio-7B-Instruct, scored by order-averaged
log-probability (Appendix A.6, A.8), mirroring qwen_omni_backend.py.

Loads once per process (module-level singleton) and exposes
`score_options(question, clip_paths, options)` -> {option: logprob}, which
run_eval.py calls once per (prompt, clip-order) presentation.
"""

import torch
import torch.nn.functional as F

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

_state = {}


def load(device="cuda:0", dtype=torch.bfloat16):
    if _state:
        return _state
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=device
    )
    model.eval()
    _state["processor"] = processor
    _state["model"] = model
    _state["device"] = device
    return _state


def sample_rate():
    return load()["processor"].feature_extractor.sampling_rate


def build_prompt_text(question, n_audio):
    processor = load()["processor"]
    content = [{"type": "audio", "audio_url": "placeholder"} for _ in range(n_audio)]
    content.append({"type": "text", "text": question})
    conversation = [{"role": "user", "content": content}]
    return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


def _common_prefix_len(ids_a, ids_b):
    n = min(len(ids_a), len(ids_b))
    i = 0
    while i < n and ids_a[i] == ids_b[i]:
        i += 1
    return i


@torch.no_grad()
def _answer_logprob(prompt_text, audios, candidate_word):
    st = load()
    processor, model, device = st["processor"], st["model"], st["device"]

    inputs_prompt = processor(text=prompt_text, audio=audios, return_tensors="pt", padding=True)
    full_text = prompt_text + " " + candidate_word
    inputs_full = processor(text=full_text, audio=audios, return_tensors="pt", padding=True)

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
    from musiclistenbench.backends import audio_io

    sr = sr or sample_rate()
    audios = [audio_io.load_mono(p, sr=sr) for p in clip_paths]
    prompt_text = build_prompt_text(question, len(audios))
    return {opt: _answer_logprob(prompt_text, audios, opt) for opt in options}
