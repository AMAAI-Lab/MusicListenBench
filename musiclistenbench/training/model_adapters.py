"""Per-model adapters for the GRPO trainers (`train_grpo.py` LoRA and
`train_grpo_fullft.py` full-parameter) and, indirectly, eval.

The GRPO algorithm (rollouts, advantage, KL, DDP, reward, logging) is entirely
model-agnostic. Only these model-specific things live here:

  1. HF classes + repo id                4. compute dtype + target sample rate
  2. chat template + audio placeholder    5. LoRA target regex  (LoRA trainer)
  3. the processor call (kwarg names)      6. full-FT freeze prefixes (full-FT trainer)

`train_grpo.py --model NAME` and `train_grpo_fullft.py --model NAME` look the adapter
up here. The four models of the paper are registered: qwen2-audio, qwen2.5-omni,
audio-flamingo3 and phi4-multimodal. Each adapter follows that model's evaluation
backend (musiclistenbench/backends/*_backend.py), so training and evaluation use the
same chat template and audio front end.

Adding a model: copy an adapter, then check the fields tagged `# VERIFY` (LoRA regex,
freeze prefixes, processor kwargs, prompt tags, dtype) with a two-step run:

    CUDA_VISIBLE_DEVICES=0 python -m musiclistenbench.training.train_grpo \
        --model NAME --steps 2 --prompts-per-step 1 --save-every 2 --out-dir /tmp/grpo_smoke

Full fine-tuning's set_trainable() raises if a freeze prefix matches zero parameters, so a
wrong `freeze_prefixes` fails loudly; list the names with
`[n for n, _ in model.named_parameters()]`. `--trainable all` (unfreeze everything) needs
no prefixes.
"""

import torch

# LoRA target heuristic for the non-Qwen2-Audio models: the LM's attention/MLP
# projections, excluding the (frozen) audio/vision encoder. Names vary -> VERIFY.
_LM_PROJ = r"(?!.*(audio_tower|audio_encoder|audio_model|vision|visual|speech_encoder))" \
           r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


class ModelAdapter:
    def __init__(self, name, repo_id, dtype, target_sr, lora_target_regex, freeze_prefixes,
                 _proc, _model, _prompt, audio_kw="audio", pass_sampling_rate=True,
                 audio_as_tuple=False, freeze_excludes=(), logits_to_keep_kwarg=None):
        self.name = name
        self.repo_id = repo_id
        self.dtype = dtype
        self.target_sr = target_sr
        self.lora_target_regex = lora_target_regex          # LoRA trainer
        self.freeze_prefixes = freeze_prefixes               # full-FT trainer: {choice: prefixes|None}
        self.freeze_excludes = freeze_excludes               # full-FT: name-substrings to keep frozen even if a prefix matches (Phi-4 vision LoRA)
        self.logits_to_keep_kwarg = logits_to_keep_kwarg     # forward kwarg to compute logits for only the last K positions (saves the multi-GB full-seq logits); None = full logits
        self._proc = _proc
        self._model = _model
        self._prompt = _prompt
        self.audio_kw = audio_kw
        self.pass_sampling_rate = pass_sampling_rate
        self.audio_as_tuple = audio_as_tuple                 # Phi-4: audios=[(clip, sr)] tuples

    def load_processor(self):
        return self._proc(self.repo_id)

    def load_model(self, src=None, dtype=None):
        """Load model weights from `src` (a repo id OR a save_pretrained dir, e.g.
        full-FT --init-from); defaults to the base repo id and the adapter's dtype."""
        return self._model(src or self.repo_id, dtype or self.dtype)

    def load(self):
        """-> (processor, base_model). Convenience for the LoRA trainer / eval."""
        return self.load_processor(), self.load_model()

    def build_prompt_text(self, processor, question):
        return self._prompt(processor, question)

    def processor_inputs(self, processor, prompt_text, clip):
        # Phi-4's processor wants audios=[(array, sr)] tuples and no sampling_rate kwarg
        # (mirrors musiclistenbench/backends/phi4_multimodal_backend.py); other models take a bare [clip] + sampling_rate.
        audio_val = [(clip, self.target_sr)] if self.audio_as_tuple else [clip]
        kw = {"text": prompt_text, self.audio_kw: audio_val, "return_tensors": "pt"}
        if self.pass_sampling_rate and not self.audio_as_tuple:
            kw["sampling_rate"] = self.target_sr
        return processor(**kw)


# ---- Qwen2-Audio (VERIFIED) --------------------------------------------------
def _proc_qwen2_audio(src):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(src)


def _model_qwen2_audio(src, dtype):
    from transformers import Qwen2AudioForConditionalGeneration
    return Qwen2AudioForConditionalGeneration.from_pretrained(src, dtype=dtype)


def _prompt_qwen2_audio(processor, question):
    span = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{span}{question}<|im_end|>\n"
            "<|im_start|>assistant\n")


# ---- Qwen2.5-Omni Thinker (GROUNDED in musiclistenbench/backends/qwen_omni_backend.py) -------------
def _proc_omni(src):
    from transformers import Qwen2_5OmniProcessor
    return Qwen2_5OmniProcessor.from_pretrained(src)


def _model_omni(src, dtype):
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(src, dtype=dtype)


# ---- Audio-Flamingo-3 (GROUNDED in musiclistenbench/backends/audio_flamingo3_backend.py) -----------
def _proc_af3(src):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(src)


def _model_af3(src, dtype):
    from transformers import AudioFlamingo3ForConditionalGeneration
    return AudioFlamingo3ForConditionalGeneration.from_pretrained(src, dtype=dtype)


# ---- Phi-4-multimodal (LEAST verified) ---------------------------------------
def _proc_phi4(src):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(src, trust_remote_code=True)


def _model_phi4(src, dtype):
    from transformers import AutoModelForCausalLM
    # Phi-4 runs in its pinned transformers==4.48.2 env, which uses torch_dtype=, not the
    # 5.x dtype= alias (the other adapters run in the 5.x env and keep dtype=).
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, trust_remote_code=True)
    # Phi-4's bundled SigLIP vision + conformer audio encoders (and the decoder) run in train
    # mode with gradient_checkpointing=True but WITHOUT _gradient_checkpointing_func set (only
    # gradient_checkpointing_enable() sets it, which the LoRA trainer never calls) -> the
    # `if self.gradient_checkpointing and self.training` branch AttributeErrors. Force it off on
    # every submodule; the full-FT trainer re-enables it properly when --grad-checkpointing is set.
    for m in model.modules():
        if getattr(m, "gradient_checkpointing", False):
            m.gradient_checkpointing = False
    return model


def _prompt_phi4(processor, question):
    return f"<|user|><|audio_1|>{question}<|end|><|assistant|>"   # VERIFY: Phi-4 chat tags


def _chat_template_prompt(processor, question, audio_field="audio_url"):
    content = [{"type": "audio", audio_field: "placeholder"}, {"type": "text", "text": question}]
    return processor.apply_chat_template([{"role": "user", "content": content}],
                                         add_generation_prompt=True, tokenize=False)


ADAPTERS = {
    "qwen2-audio": ModelAdapter(
        "qwen2-audio", "Qwen/Qwen2-Audio-7B-Instruct",
        dtype=torch.bfloat16, target_sr=16000,
        lora_target_regex=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        freeze_prefixes={                                    # VERIFIED (transformers 5.x layout)
            "language_model": ("model.language_model.", "lm_head."),
            "lm+projector": ("model.language_model.", "lm_head.", "model.multi_modal_projector."),
            "all": None},
        _proc=_proc_qwen2_audio, _model=_model_qwen2_audio, _prompt=_prompt_qwen2_audio,
        audio_kw="audio", pass_sampling_rate=True),

    "qwen2.5-omni": ModelAdapter(
        "qwen2.5-omni", "Qwen/Qwen2.5-Omni-7B",
        dtype=torch.bfloat16, target_sr=16000,               # VERIFY sr
        lora_target_regex=_LM_PROJ,                           # VERIFY
        freeze_prefixes={                                     # VERIFY prefixes (Thinker submodule names)
            "language_model": ("model.", "lm_head."),
            "lm+projector": ("model.", "lm_head."),
            "all": None},
        _proc=_proc_omni, _model=_model_omni,
        _prompt=lambda p, q: _chat_template_prompt(p, q, "audio_url"),
        audio_kw="audio", pass_sampling_rate=True),

    "audio-flamingo3": ModelAdapter(
        "audio-flamingo3", "nvidia/audio-flamingo-3-hf",
        dtype=torch.float32, target_sr=16000,                # fp32: AF3 bf16 embed_positions bug; VERIFY sr
        lora_target_regex=_LM_PROJ,                           # VERIFY
        freeze_prefixes={   # VERIFIED against RUNTIME named_parameters() (transformers 5.x standardizes AF3 to model.language_model / model.audio_tower / model.multi_modal_projector / top-level lm_head -- IDENTICAL layout to qwen2-audio; the on-disk safetensors keys differ and must NOT be used here)
            "language_model": ("model.language_model.", "lm_head."),                                # decoder 7069M + lm_head 544M = 7.61B; audio_tower(637M)+projector(17M) frozen -> encoder-frozen, matches qwen2-audio
            "lm+projector": ("model.language_model.", "lm_head.", "model.multi_modal_projector."),  # + the 17.4M audio->LM projector
            "all": None},
        _proc=_proc_af3, _model=_model_af3,
        _prompt=lambda p, q: _chat_template_prompt(p, q, "path"),
        audio_kw="audio", pass_sampling_rate=False,
        logits_to_keep_kwarg="logits_to_keep"),   # fp32 AF3: compute only the needed tail logits so the policy fits 80GB (forward verified to accept logits_to_keep)

    "phi4-multimodal": ModelAdapter(
        "phi4-multimodal", "microsoft/Phi-4-multimodal-instruct",
        dtype=torch.bfloat16, target_sr=16000,               # VERIFY sr
        lora_target_regex=r"(?!.*(audio|vision|image|speech)).*\.(qkv_proj|o_proj|gate_up_proj|down_proj)$",  # VERIFY
        freeze_prefixes={                                     # base LM + embed + lm_head (speech LoRA rides the audio path; vision LoRA excluded below)
            "language_model": ("model.layers.", "model.embed_tokens.", "lm_head."),
            "lm+projector": ("model.layers.", "model.embed_tokens.", "lm_head."),
            "all": None},
        # Phi-4-mm bakes per-layer speech AND vision LoRA under model.layers.*. Audio-only
        # batches never run the vision LoRA, so leaving it trainable trips DDP's
        # find_unused_parameters assertion at step 2. Freeze it by name -- base LM + the
        # speech LoRA stay trainable (audio + language retraining), and the dead vision
        # LoRA is also kept out of the optimizer. See train_grpo_fullft.set_trainable.
        freeze_excludes=("vision",),
        _proc=_proc_phi4, _model=_model_phi4, _prompt=_prompt_phi4,
        audio_kw="audios", pass_sampling_rate=True, audio_as_tuple=True),
}

TRAINABLE = list(ADAPTERS)   # the four models GRPO can train


def get_adapter(name):
    if name not in ADAPTERS:
        raise ValueError(f"unknown --model {name!r}; choices: {TRAINABLE}")
    return ADAPTERS[name]
