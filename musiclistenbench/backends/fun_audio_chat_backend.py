"""Local backend: Fun-Audio-Chat-8B (FunAudioLLM/Alibaba), scored by
order-averaged log-probability (Appendix A.6, A.8), mirroring
qwen_omni_backend.py.

Fun-Audio-Chat is NOT a plain `transformers` AutoModel -- it registers its
own Auto-classes at import time via `funaudiochat.register`, shipped in the
model's own GitHub repo (not on PyPI). Clone the repo and point
FUN_AUDIO_CHAT_REPO at it (see setup commands below); it must be importable
as both `funaudiochat` (the Auto-class registration) and `utils` (the
`AUDIO_TEMPLATE`/`DEFAULT_S2T_PROMPT` constants used by their own example),
so the repo root goes on sys.path, matching how their own
examples/infer_s2t.py is run (from the repo root).

Confirmed against examples/infer_s2t.py for the single-clip prompt/generate
path. The direct forward()/logits path used here for teacher-forced scoring
is architecturally supported (modeling_funaudiochat.py's forward() returns
`text_logits`, and `.logits` aliases it) but no official example calls
forward() directly -- only .generate(). Two things to double check once
weights are downloaded:
  - AUDIO_TEMPLATE is only ever shown used once per conversation (n_audio=1)
    in the official example; repeating it per clip for pair trials
    (found-pair/harmony tasks use 2 clips) is this file's assumption, not
    a verified convention.
  - the `.logits` alias returns the text-vocab logits (vocab_size 151936
    per config.json's text_config), not the audio-codec logits (a separate
    ~6565-entry codebook) -- if `.logits` ever returns something with the
    wrong last dim, that's this alias breaking, not a bug in the scoring
    loop below.

Setup (needs its own env -- pins transformers==4.52.3, flash-attn, and a
specific torch/CUDA combo; do not install into the same env as the Qwen/
Audio-Flamingo-3 backends):
    git clone --recurse-submodules https://github.com/FunAudioLLM/Fun-Audio-Chat.git third_party/Fun-Audio-Chat
    pip install transformers==4.52.3 torch==2.8.0 torchaudio==2.8.0
    pip install flash-attn --no-build-isolation
    hf download FunAudioLLM/Fun-Audio-Chat-8B --local-dir third_party/Fun-Audio-Chat/pretrained_models/Fun-Audio-Chat-8B
"""

import os
import sys

import torch
import torch.nn.functional as F

from musiclistenbench.backends import config

REPO_DIR = os.environ.get("FUN_AUDIO_CHAT_REPO", os.path.join(config.PROJECT_ROOT, "third_party", "Fun-Audio-Chat"))
MODEL_PATH = os.environ.get(
    "FUN_AUDIO_CHAT_MODEL", os.path.join(REPO_DIR, "pretrained_models", "Fun-Audio-Chat-8B")
)

_state = {}


def load(device="cuda:0", dtype=torch.bfloat16):
    if _state:
        return _state
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
    from funaudiochat.register import register_funaudiochat
    from utils.constant import AUDIO_TEMPLATE, DEFAULT_S2T_PROMPT

    register_funaudiochat()
    hf_config = AutoConfig.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH, config=hf_config, torch_dtype=dtype, device_map=device
    )
    model.eval()
    _state["processor"] = processor
    _state["model"] = model
    _state["device"] = device
    _state["audio_template"] = AUDIO_TEMPLATE
    _state["system_prompt"] = DEFAULT_S2T_PROMPT
    return _state


def sample_rate():
    return 16000  # fixed by the model's own front end (librosa.load(..., sr=16000) in its examples)


def build_prompt_text(question, n_audio):
    st = load()
    audio_block = "\n".join([st["audio_template"]] * n_audio)
    conversation = [
        {"role": "system", "content": st["system_prompt"]},
        {"role": "user", "content": audio_block + "\n" + question},
    ]
    return st["processor"].apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


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

    inputs_prompt = processor(text=prompt_text, audio=audios, return_tensors="pt", return_token_type_ids=False)
    full_text = prompt_text + " " + candidate_word
    inputs_full = processor(text=full_text, audio=audios, return_tensors="pt", return_token_type_ids=False)

    ids_prompt = inputs_prompt["input_ids"][0]
    ids_full = inputs_full["input_ids"][0]
    prefix_len = _common_prefix_len(ids_prompt, ids_full)
    targets = ids_full[prefix_len:]
    if len(targets) == 0:
        raise RuntimeError(f"candidate {candidate_word!r} tokenized to nothing new past the prompt prefix")

    inputs_full = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs_full.items()}
    out = model(**inputs_full)
    logits = out.logits[0]  # aliases text_logits, vocab_size 151936 -- see modeling_funaudiochat.py

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
