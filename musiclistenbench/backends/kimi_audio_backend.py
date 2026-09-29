"""Local backend: Moonshot AI's Kimi-Audio-7B-Instruct, scored by
order-averaged log-probability (Appendix A.6, A.8), mirroring
qwen_omni_backend.py.

Kimi-Audio's architecture is a dual-stream design: every timestep carries
both an audio-token id and a text-token id in lockstep (they're summed as
embeddings, not concatenated), plus optional continuous Whisper features at
audio positions. There's no single `processor(text=..., audio=...)` call
like Qwen2-Audio -- prompt construction is delegated here to the model's own
`KimiAPromptManager` (kimia_infer/api/prompt_manager.py) rather than
reimplemented, since hand-rolling the dual-stream/special-token bookkeeping
would be easy to get subtly wrong. `forward()` still returns a plain
`text_logits` tensor over the *text* vocabulary (151936, i.e. the `lm_head`
branch) from a single full-sequence pass -- teacher-forced scoring here
works the same way as Qwen2-Audio's, just diffing the text-id stream instead
of a single token stream.

Verified from source (2026-07):
  - kimia_infer.api.prompt_manager.KimiAPromptManager and
    kimia_infer.utils.data.KimiAContent's to_tensor()/merge() API.
  - model.forward(input_ids=<audio ids>, text_input_ids=<text ids>,
    whisper_input_feature=..., is_continuous_mask=..., position_ids=...,
    past_key_values=None, return_dict=False) -> (audio_logits, text_logits,
    past_key_values), used verbatim in kimia_infer/api/kimia.py's own
    generation loop.
  - config.json: kimia_token_offset=152064. There is no
    `kimia_text_audiodelaytokens` field in config.json even though that's
    KimiAPromptManager's constructor arg name -- config.json instead has
    `kimia_mimo_audiodelaytokens: 6`, which this file assumes is the same
    value under the model-config naming vs. prompt-manager-arg naming.
    Double check this mapping if generation looks systematically shifted.

Setup (needs its own env -- pins torch==2.6.0/flash_attn==2.7.4.post1/
deepspeed, incompatible with the Qwen/Audio-Flamingo-3 backends' env):
    git clone https://github.com/MoonshotAI/Kimi-Audio.git third_party/Kimi-Audio
    cd third_party/Kimi-Audio && git submodule update --init --recursive && cd -
    pip install -r third_party/Kimi-Audio/requirements.txt
    hf download moonshotai/Kimi-Audio-7B-Instruct --local-dir third_party/Kimi-Audio/pretrained_models/Kimi-Audio-7B-Instruct
"""

import os
import sys

import torch
import torch.nn.functional as F

from musiclistenbench.backends import config

REPO_DIR = os.environ.get("KIMI_AUDIO_REPO", os.path.join(config.PROJECT_ROOT, "third_party", "Kimi-Audio"))
MODEL_PATH = os.environ.get(
    "KIMI_AUDIO_MODEL", os.path.join(config.PROJECT_ROOT, "pretrained_models", "Kimi-Audio-7B-Instruct")
)

_state = {}


def load(device="cuda:0", dtype=torch.bfloat16):
    if _state:
        return _state
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    from transformers import AutoModelForCausalLM
    from kimia_infer.api.prompt_manager import KimiAPromptManager

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=dtype, device_map=device
    )
    model.eval()

    audiodelaytokens = getattr(
        model.config, "kimia_text_audiodelaytokens", getattr(model.config, "kimia_mimo_audiodelaytokens")
    )
    prompt_manager = KimiAPromptManager(
        model_path=MODEL_PATH,
        kimia_token_offset=model.config.kimia_token_offset,
        kimia_text_audiodelaytokens=audiodelaytokens,
    )

    _state["model"] = model
    _state["prompt_manager"] = prompt_manager
    _state["device"] = device
    return _state


def sample_rate():
    return 16000


def _common_prefix_len(ids_a, ids_b):
    n = min(len(ids_a), len(ids_b))
    i = 0
    while i < n and ids_a[i] == ids_b[i]:
        i += 1
    return i


@torch.no_grad()
def _answer_logprob(clip_paths, question, candidate_word):
    st = load()
    model, prompt_manager, device = st["model"], st["prompt_manager"], st["device"]

    base_messages = [{"role": "user", "message_type": "audio", "content": p} for p in clip_paths]
    base_messages.append({"role": "user", "message_type": "text", "content": question})

    prompt_only = prompt_manager.get_prompt(base_messages, output_type="text", add_assistant_start_msg=True)
    full = prompt_manager.get_prompt(
        base_messages + [{"role": "assistant", "message_type": "text", "content": candidate_word}],
        output_type="text",
        add_assistant_start_msg=False,
    )

    prompt_audio_ids, prompt_text_ids, _prompt_cont, *_ = prompt_only.to_tensor()
    full_audio_ids, full_text_ids, full_is_continuous, *_ = full.to_tensor()
    # to_tensor() already returns [1, seq] batched tensors (see
    # kimia_infer/utils/data.py's to_tensor() and kimia.py's own generate()
    # loop, which passes them straight to model.forward() with no further
    # unsqueeze) -- don't add another batch dim here.

    prefix_len = _common_prefix_len(prompt_text_ids[0].tolist(), full_text_ids[0].tolist())
    targets = full_text_ids[0, prefix_len:]
    if len(targets) == 0:
        raise RuntimeError(f"candidate {candidate_word!r} tokenized to nothing new past the prompt prefix")

    seq_len = full_text_ids.shape[1]
    audio_ids = full_audio_ids.to(device)
    text_ids = full_text_ids.to(device)
    is_continuous_mask = full_is_continuous.to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    whisper_feat = full.continuous_feature if full.continuous_feature else None

    _audio_logits, text_logits, _cache = model.forward(
        input_ids=audio_ids,
        text_input_ids=text_ids,
        whisper_input_feature=whisper_feat,
        is_continuous_mask=is_continuous_mask,
        position_ids=position_ids,
        past_key_values=None,
        return_dict=False,
    )
    logits = text_logits[0]  # (seq, text_vocab)

    logprob = 0.0
    for i, tgt in enumerate(targets):
        pos = prefix_len - 1 + i
        lp = F.log_softmax(logits[pos].float(), dim=-1)[tgt.item()]
        logprob += lp.item()
    return logprob


def score_options(question, clip_paths, options, sr=None):
    """clip_paths: list of wav paths (1 or 2). options: (word_a, word_b).
    Returns {word: logprob}."""
    return {opt: _answer_logprob(clip_paths, question, opt) for opt in options}
