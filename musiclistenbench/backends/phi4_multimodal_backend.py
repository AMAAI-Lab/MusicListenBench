"""Local backend: Microsoft Phi-4-multimodal-instruct (audio), scored by
forced-choice log-probability, mirroring qwen2_audio_backend.py.

Phi-4-multimodal's prompt tags and processor `audios=` kwarg differ from the Qwen
family. Smoke test:
    python -m musiclistenbench.eval.run_probe_mm --backend phi4-multimodal \
        --eval-json data/eval_perturb.json \
        --audio-root data/audio_perturb --limit 8 --out /tmp/phi4_smoke.jsonl
If it errors, check the `# VERIFY` lines against the model card of the checkpoint you use.
"""

import torch
import torch.nn.functional as F

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

_state = {}


def load(device="cuda:0", dtype=torch.bfloat16):
    if _state:
        return _state
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    model.eval()
    _state["processor"] = processor
    _state["model"] = model
    _state["device"] = device
    return _state


def sample_rate():
    fe = getattr(load()["processor"], "feature_extractor", None)
    return getattr(fe, "sampling_rate", 16000)   # VERIFY


def build_prompt_text(question):
    return f"<|user|><|audio_1|>{question}<|end|><|assistant|>"   # VERIFY: Phi-4 chat tags


def _common_prefix_len(ids_a, ids_b):
    n = min(len(ids_a), len(ids_b))
    i = 0
    while i < n and ids_a[i] == ids_b[i]:
        i += 1
    return i


@torch.no_grad()
def _answer_logprob(prompt_text, audio, sr, candidate_word):
    st = load()
    processor, model, device = st["processor"], st["model"], st["device"]
    # VERIFY: Phi-4 processor takes audios as a list of (array, sample_rate) tuples.
    inputs_prompt = processor(text=prompt_text, audios=[(audio, sr)], return_tensors="pt")
    inputs_full = processor(text=prompt_text + " " + candidate_word, audios=[(audio, sr)], return_tensors="pt")

    ids_prompt = inputs_prompt["input_ids"][0]
    ids_full = inputs_full["input_ids"][0]
    prefix_len = _common_prefix_len(ids_prompt, ids_full)
    targets = ids_full[prefix_len:]
    if len(targets) == 0:
        raise RuntimeError(f"candidate {candidate_word!r} tokenized to nothing new past the prompt")

    inputs_full = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs_full.items()}
    logits = model(**inputs_full).logits[0]
    logprob = 0.0
    for i, tgt in enumerate(targets):
        logprob += F.log_softmax(logits[prefix_len - 1 + i].float(), dim=-1)[tgt.item()].item()
    return logprob


def score_options(question, clip_paths, options, sr=None):
    """clip_paths: 1 or 2 wav paths. options: (word_a, word_b). Returns {word: logprob}."""
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
    return {opt: _answer_logprob(prompt_text, audio, sr, opt) for opt in options}
