"""Local backend: Xiaomi's MiMo-Audio-7B-Instruct, scored by order-averaged
log-probability (Appendix A.6, A.8), mirroring qwen_omni_backend.py.

Verified against the real XiaomiMiMo/MiMo-Audio repo (third_party/MiMo-Audio,
cloned 2026-09), by reading mimo_audio/mimo_audio.py's own
`MimoAudio.__init__`/`preprocess_input`/`get_audio_understanding_sft_prompt`
and mimo_audio/modeling_mimo_audio.py's `MiMoAudioForCausalLM.forward`/
`_prepare_input_embeds` directly (not guessed from the model card).

Import note: the cloned repo's own top-level package is named `src`. `musiclistenbench/backends/mimo_audio` and
`musiclistenbench/backends/mimo_audio_tokenizer` are symlinks into the cloned repo (see setup
commands below) so `from musiclistenbench.backends.mimo_audio...` resolves as a real subpackage
of *this* package, matching the relative imports inside those files
(e.g. `mimo_audio.py` does `from ..mimo_audio_tokenizer import
MiMoAudioTokenizer`, which only works if both are siblings under one package).

Two load-bearing details confirmed from source, both different from a naive
reading of the model card:
  - `MiMoAudioForCausalLM.forward()` unconditionally slices
    `hidden_states[:, -1:, :]` before the lm_head -- one call only ever
    returns logits for the token *after* whatever prefix you passed in, so
    `_answer_logprob` below reruns a full forward pass once per extra
    (candidate-word) *group*, each time feeding a longer prefix.
  - The model runs at a GROUPED sequence resolution: `_prepare_input_embeds`
    folds every `group_size` (4) raw [audio_channels+1, T] columns into one
    hidden-state position, and `InputSegment.to_input_id` guarantees every
    semantic unit (one text token, or one sosp/eosp/audio-frame group)
    occupies exactly one full group, with the real text-channel id at
    position 0 of its group and `-100` filler after (see
    `InputSegment.insert_between`). So attention_mask/position_ids/
    cache_position must be at length `T // group_size`, and stepping the
    teacher-forced scoring loop one *group* at a time (not one raw token)
    is what the architecture actually supports -- there is no way to read
    a logit at an intra-group position.

Setup (needs its own env -- verified here against the `kimiaudio` conda env,
which already has a compatible torch==2.6.0 + flash-attn build; a fresh env
should follow third_party/MiMo-Audio/requirements.txt: torch==2.6.0,
transformers==4.49.0, flash-attn):
    git clone https://github.com/XiaomiMiMo/MiMo-Audio.git third_party/MiMo-Audio
    ln -s ../../third_party/MiMo-Audio/src/mimo_audio musiclistenbench/backends/mimo_audio
    ln -s ../../third_party/MiMo-Audio/src/mimo_audio_tokenizer musiclistenbench/backends/mimo_audio_tokenizer
    hf download XiaomiMiMo/MiMo-Audio-7B-Instruct --local-dir third_party/MiMo-Audio/pretrained_models/MiMo-Audio-7B-Instruct
    hf download XiaomiMiMo/MiMo-Audio-Tokenizer --local-dir third_party/MiMo-Audio/pretrained_models/MiMo-Audio-Tokenizer
"""

import os

import torch
import torch.nn.functional as F

from musiclistenbench.backends import config

REPO_DIR = os.environ.get("MIMO_AUDIO_REPO", os.path.join(config.PROJECT_ROOT, "third_party", "MiMo-Audio"))
MODEL_PATH = os.environ.get(
    "MIMO_AUDIO_MODEL", os.path.join(REPO_DIR, "pretrained_models", "MiMo-Audio-7B-Instruct")
)
TOKENIZER_PATH = os.environ.get(
    "MIMO_AUDIO_TOKENIZER", os.path.join(REPO_DIR, "pretrained_models", "MiMo-Audio-Tokenizer")
)

_state = {}

SPECIAL_TOKENS = [
    "<|sosp|>", "<|eosp|>", "<|empty|>", "<|Human|>", "<|SpeechLM|>",
    "<|sostm|>", "<|eostm|>", "<|eot|>",
]


def load(device="cuda:0", dtype=torch.bfloat16):
    if _state:
        return _state
    from torchaudio.transforms import MelSpectrogram
    from transformers import AutoTokenizer

    from musiclistenbench.backends.mimo_audio.modeling_mimo_audio import MiMoAudioArguments, MiMoAudioForCausalLM
    from musiclistenbench.backends.mimo_audio_tokenizer import MiMoAudioTokenizer

    text_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    for token in SPECIAL_TOKENS:
        if token not in text_tokenizer.get_vocab():
            text_tokenizer.add_tokens([token], special_tokens=True)

    empty_idx = text_tokenizer.convert_tokens_to_ids("<|empty|>")
    model_args = MiMoAudioArguments(
        model_name_or_path=MODEL_PATH,
        sosp_idx=text_tokenizer.convert_tokens_to_ids("<|sosp|>"),
        eosp_idx=text_tokenizer.convert_tokens_to_ids("<|eosp|>"),
        sostm_idx=text_tokenizer.convert_tokens_to_ids("<|sostm|>"),
        eostm_idx=text_tokenizer.convert_tokens_to_ids("<|eostm|>"),
        eot_idx=text_tokenizer.convert_tokens_to_ids("<|eot|>"),
        empty_idx=empty_idx,
    )
    model = MiMoAudioForCausalLM.from_pretrained(
        MODEL_PATH, args=model_args, torch_dtype=dtype, device_map={"": device}
    )
    model.eval()

    audio_tokenizer = MiMoAudioTokenizer.from_pretrained(TOKENIZER_PATH)
    audio_tokenizer.eval().bfloat16().to(device)

    mel_transform = MelSpectrogram(
        sample_rate=audio_tokenizer.config.sampling_rate,
        n_fft=audio_tokenizer.config.nfft,
        hop_length=audio_tokenizer.config.hop_length,
        win_length=audio_tokenizer.config.window_size,
        f_min=audio_tokenizer.config.fmin,
        f_max=audio_tokenizer.config.fmax,
        n_mels=audio_tokenizer.config.n_mels,
        power=1.0,
        center=True,
    ).to(device)

    _state.update(
        model=model,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        mel_transform=mel_transform,
        device=device,
        group_size=model.config.group_size,
        audio_channels=model.config.audio_channels,
        speech_zeroemb_idx=model.speech_empty_ids,
        empty_idx=empty_idx,
    )
    return _state


def sample_rate():
    return load()["audio_tokenizer"].config.sampling_rate


def _group_by_length(features, lengths, max_length):
    """Mirrors MimoAudio.group_by_length: split into <=max_length chunks
    without cutting a frame's worth of features in half."""
    split_points = []
    current_sum = 0
    for i, seq_len in enumerate(lengths):
        if current_sum + seq_len > max_length and current_sum > 0:
            split_points.append(i)
            current_sum = seq_len.item()
        else:
            current_sum += seq_len.item()
    group_sizes = []
    prev = 0
    for point in split_points:
        group_sizes.append(point - prev)
        prev = point
    if prev < len(lengths):
        group_sizes.append(len(lengths) - prev)
    len_groups = torch.split(lengths, group_sizes)
    feature_sizes = [g.sum().item() for g in len_groups]
    feature_groups = torch.split(features, feature_sizes)
    return feature_groups, len_groups


@torch.no_grad()
def _encode_batch(st, input_features, input_lens, max_length=256000):
    feature_groups, len_groups = _group_by_length(input_features, input_lens, max_length)
    encoded_parts = []
    for features, lengths in zip(feature_groups, len_groups):
        codes, _ = st["audio_tokenizer"].encoder.encode(
            input_features=features.to(st["device"]),
            input_lens=lengths.to(st["device"]),
            return_codes_only=True,
        )
        encoded_parts.append(codes)
    return torch.cat(encoded_parts, dim=-1)


@torch.no_grad()
def _audio_tokenized(st, clip_path):
    """Mirrors MimoAudio.preprocess_input's audio branch: raw wav ->
    log-mel -> RVQ codes -> flat [T * audio_channels] tensor."""
    from musiclistenbench.backends import audio_io

    wav = audio_io.load_mono(clip_path, sr=sample_rate())
    wav = torch.from_numpy(wav).to(st["device"])

    spec = st["mel_transform"](wav[None, :])
    mel = torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1)  # (seq_len, n_mels)

    input_len = mel.size(0)
    segment_size = 6000
    input_len_seg = [segment_size] * (input_len // segment_size)
    if input_len % segment_size > 0:
        input_len_seg.append(input_len % segment_size)

    codes_packed = _encode_batch(st, mel, torch.tensor(input_len_seg))
    codes = codes_packed.transpose(0, 1).detach().cpu()
    audio_codes = codes[:, : st["audio_channels"]]

    num_timesteps = audio_codes.shape[0]
    group_size = st["group_size"]
    if num_timesteps % group_size != 0:
        padding_needed = group_size - (num_timesteps % group_size)
        last_tokens = audio_codes[-1:, :]
        audio_codes = torch.cat([audio_codes, last_tokens.repeat(padding_needed, 1)], dim=0)

    return audio_codes.reshape(-1)


def _text_segment(st, text):
    from musiclistenbench.backends.mimo_audio.process_speechdata import InputSegment

    return InputSegment(
        text=text,
        speech_zeroemb_idx=st["speech_zeroemb_idx"],
        text_zeroemb_idx=st["empty_idx"],
    ).to_input_id(st["text_tokenizer"], st["group_size"], st["audio_channels"])


def _audio_segment(st, audio_tokenized):
    from musiclistenbench.backends.mimo_audio.process_speechdata import InputSegment

    return InputSegment(
        audio=audio_tokenized,
        speech_zeroemb_idx=st["speech_zeroemb_idx"],
        text_zeroemb_idx=st["empty_idx"],
    ).to_input_id(st["text_tokenizer"], st["group_size"], st["audio_channels"])


def _build_input_ids(st, audio_tokenized, question, answer=None):
    """Mirrors MimoAudio.get_audio_understanding_sft_prompt (non-thinking
    variant)."""
    segments = [
        _text_segment(st, "<|im_start|>user\n"),
        _audio_segment(st, audio_tokenized),
        _text_segment(st, question),
        _text_segment(st, "<|im_end|>\n"),
        _text_segment(st, "<|im_start|>assistant\n"),
        _text_segment(st, "<think>\n\n</think>\n"),
    ]
    if answer is not None:
        segments.append(_text_segment(st, answer))
    return torch.cat(segments, dim=1)  # [audio_channels + 1, T]


@torch.no_grad()
def _forward_last_logits(st, input_ids):
    device = st["device"]
    group_size = st["group_size"]
    _channels, t = input_ids.shape
    assert t % group_size == 0, f"sequence length {t} not a multiple of group_size {group_size}"
    t_group = t // group_size
    batch = input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones(1, t_group, device=device)
    position_ids = torch.arange(t_group, device=device).unsqueeze(0)
    cache_position = torch.arange(t_group, device=device)
    out = st["model"].forward(
        input_ids=batch,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        cache_position=cache_position,
    )
    return out.text_logits[0, -1]  # logits predicting the token *after* this prefix


@torch.no_grad()
def _answer_logprob(st, audio_tokenized, question, candidate_word):
    group_size = st["group_size"]
    prompt_ids = _build_input_ids(st, audio_tokenized, question)
    full_ids = _build_input_ids(st, audio_tokenized, question, answer=candidate_word)
    prompt_groups = prompt_ids.shape[1] // group_size
    full_groups = full_ids.shape[1] // group_size
    if full_groups <= prompt_groups:
        raise RuntimeError(f"candidate {candidate_word!r} added no new tokens past the prompt")

    logprob = 0.0
    for g in range(prompt_groups, full_groups):
        t = g * group_size  # always a multiple of group_size
        logits = _forward_last_logits(st, full_ids[:, :t])
        true_tok = full_ids[0, t].item()  # real token sits at position 0 of group g
        lp = F.log_softmax(logits.float(), dim=-1)[true_tok]
        logprob += lp.item()
    return logprob


def score_options(question, clip_paths, options, sr=None):
    """clip_paths: 1 wav path (this benchmark's combined-audio strategy never
    sends more than one). options: (word_a, word_b). Returns {word: logprob}."""
    if len(clip_paths) > 1:
        raise NotImplementedError("mimo-audio backend only supports single-clip scoring")
    st = load()
    audio_tokenized = _audio_tokenized(st, clip_paths[0])
    return {opt: _answer_logprob(st, audio_tokenized, question, opt) for opt in options}
