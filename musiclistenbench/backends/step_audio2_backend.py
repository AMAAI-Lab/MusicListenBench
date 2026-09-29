"""Local backend: StepFun's Step-Audio-2-mini, scored by order-averaged
log-probability (Appendix A.6, A.8), mirroring qwen_omni_backend.py.

Step-Audio-2-mini is loaded via `trust_remote_code=True` (custom
`StepAudio2ForCausalLM` class shipped on its own HF repo) rather than a
built-in transformers class. Its forward() does a single full-sequence
pass and returns plain `.logits` -- like Qwen2-Audio, unlike Kimi-Audio/
MiMo-Audio which only expose incremental decode-step logits.

The one non-obvious part: Step-Audio2's forward() expects `input_ids` to
already contain the *exact* number of `<audio_patch>` placeholder tokens
that its audio encoder + adaptor will produce for a given clip (it splices
audio embeddings into those positions rather than expanding placeholders
itself, unlike Qwen2Audio's processor). `compute_token_num()` below
reproduces that count from the model's own utils.py so the placeholder
count always matches.

Setup (needs its own env -- pins transformers==4.49.0, incompatible with
the newer transformers used by the Qwen/Audio-Flamingo-3 backends):
    git clone https://github.com/stepfun-ai/Step-Audio2.git third_party/Step-Audio2
    pip install transformers==4.49.0 torchaudio librosa onnxruntime s3tokenizer diffusers hyperpyyaml
    hf download stepfun-ai/Step-Audio-2-mini --local-dir pretrained_models/Step-Audio-2-mini
"""

import os

import torch
import torch.nn.functional as F

from musiclistenbench.backends import config

REPO_DIR = os.environ.get("STEP_AUDIO2_REPO", os.path.join(config.PROJECT_ROOT, "third_party", "Step-Audio2"))
MODEL_PATH = os.environ.get(
    "STEP_AUDIO2_MODEL", os.path.join(config.PROJECT_ROOT, "pretrained_models", "Step-Audio-2-mini")
)

AUDIO_PATCH_TOKEN = "<audio_patch>"
AUDIO_START_TOKEN = "<audio_start>"
AUDIO_END_TOKEN = "<audio_end>"

_state = {}


def _ensure_repo_on_path():
    import sys

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)


def load(device="cuda:0", dtype=torch.bfloat16):
    if _state:
        return _state
    _ensure_repo_on_path()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["device"] = device
    return _state


def sample_rate():
    return 16000


def build_prompt_and_mels(question, audios, device):
    # log_mel_spectrogram/compute_token_num are used verbatim from the
    # repo's own utils.py rather than re-derived, since the mel filterbank
    # design (a Whisper-style precomputed filter set, not a standard
    # torchaudio melscale) is not something to reimplement by hand.
    from utils import compute_token_num, log_mel_spectrogram

    st = load()
    tokenizer = st["tokenizer"]

    mels = [log_mel_spectrogram(a, device=device) for a in audios]
    audio_blocks = []
    for mel in mels:
        n_tok = compute_token_num(mel.shape[1])
        audio_blocks.append(AUDIO_START_TOKEN + AUDIO_PATCH_TOKEN * n_tok + AUDIO_END_TOKEN)

    text = "\n".join(audio_blocks) + "\n" + question
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt_text, mels


def _common_prefix_len(ids_a, ids_b):
    n = min(len(ids_a), len(ids_b))
    i = 0
    while i < n and ids_a[i] == ids_b[i]:
        i += 1
    return i


@torch.no_grad()
def _answer_logprob(prompt_text, mels, candidate_word):
    from utils import padding_mels  # vendored from third_party/Step-Audio2/utils.py

    st = load()
    tokenizer, model, device = st["tokenizer"], st["model"], st["device"]

    ids_prompt = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
    full_text = prompt_text + " " + candidate_word
    ids_full = tokenizer(full_text, return_tensors="pt")["input_ids"][0]

    prefix_len = _common_prefix_len(ids_prompt.tolist(), ids_full.tolist())
    targets = ids_full[prefix_len:]
    if len(targets) == 0:
        raise RuntimeError(f"candidate {candidate_word!r} tokenized to nothing new past the prompt prefix")

    wavs, wav_lens = padding_mels(mels)
    wavs, wav_lens = wavs.to(device), wav_lens.to(device)
    input_ids = ids_full.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, wavs=wavs, wav_lens=wav_lens, attention_mask=attention_mask)
    logits = out.logits[0]  # (seq, vocab) -- full-sequence, single forward pass

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

    device = load()["device"]
    sr = sr or sample_rate()
    audios = [audio_io.load_mono(p, sr=sr) for p in clip_paths]
    prompt_text, mels = build_prompt_and_mels(question, audios, device)
    return {opt: _answer_logprob(prompt_text, mels, opt) for opt in options}
