"""Caption MusicCaps clips with Qwen2-Audio / Qwen2.5-Omni (Thinker) — base
weights vs GRPO checkpoints (LoRA and full fine-tune) — for Probe B (easy
version: "did GRPO hurt captioning?").

Every (model, variant) combination sees the SAME frozen manifest of clips with
the SAME prompt and greedy decoding, so base-vs-trained differences are paired.

Model loading mirrors the proven paths in this repo:
  * qwen2-audio : musiclistenbench/eval/run_probe_mm.py     (manual chat template, 16 kHz)
  * qwen2.5-omni: musiclistenbench/backends/qwen_omni_backend.py       (processor chat template)
Full-FT checkpoints are standalone saves (model + tokenizer + processor config);
LoRA adapters are merged into the base model with peft, exactly like
run_probe.py / run_probe_mm.py do.

Usage (one config per GPU; see run_all_captioning.sh):
    python run_captioning.py --model qwen2-audio  --variant base   --device cuda:2
    python run_captioning.py --model qwen2-audio  --variant lora   --device cuda:3
    python run_captioning.py --model qwen2-audio  --variant fullft --device cuda:4
    python run_captioning.py --model qwen2.5-omni --variant base   --device cuda:5
    ...
"""

import argparse
import json
import os
import time
from pathlib import Path

from musiclistenbench import paths

DATA = Path(paths.REPO_ROOT) / "musiccaps"                 # downloaded clips + manifest (not redistributed)
RESULTS = Path(paths.RESULTS_DIR) / "captioning"
CKPT = Path(paths.CHECKPOINT_DIR)

QWEN2_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
LORA_DIRS = {
    "qwen2-audio": CKPT / "grpo_qwen2_audio_lora",
    "qwen2.5-omni": CKPT / "grpo_omni_lora",
}
FULLFT_DIRS = {
    "qwen2-audio": CKPT / "grpo_qwen2_audio_fullft",
    "qwen2.5-omni": CKPT / "grpo_omni_fullft",
}

PROMPT = ("Describe this music in a few sentences. Mention the genre, the "
          "instruments, the mood, and the tempo.")
MAX_NEW_TOKENS = 256
# Both models occasionally keep generating a fake multi-turn transcript after
# answering ("Human: ... "). Stop at the first fake turn — applied identically
# to every (model, variant), so the paired comparison stays fair.
STOP_STRINGS = ["Human:", "\nUser:", "\n\nUser"]


def cut_at_stop(text: str) -> str:
    for s in STOP_STRINGS:
        i = text.find(s)
        if i != -1:
            text = text[:i]
    return text.strip()


def latest_checkpoint(base_dir: Path, ckpt_step=None) -> Path:
    if ckpt_step is not None:
        p = base_dir / f"checkpoint-{ckpt_step}"
        if not p.is_dir():
            raise SystemExit(f"no such checkpoint: {p}")
        return p
    best, best_n = None, -1
    for c in base_dir.glob("checkpoint-*"):
        n = c.name.split("-")[-1]
        if n.isdigit() and int(n) > best_n:
            best, best_n = c, int(n)
    if best is None:
        raise SystemExit(f"no checkpoint-* under {base_dir}")
    return best


def load_audio(path: Path, target_sr: int):
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)


def sanitize_generation(model, max_new_tokens: int) -> None:
    """Greedy decoding, no sampling — set on the model-level config so shipped
    generation_config.json defaults (which often turn sampling on) can't leak in."""
    gc = model.generation_config
    gc.do_sample = False
    gc.num_beams = 1
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    gc.repetition_penalty = 1.0
    gc.max_new_tokens = max_new_tokens


class Qwen2AudioRunner:
    """Exact prompt/processor path of musiclistenbench/eval/run_probe_mm.py."""

    model_id = QWEN2_MODEL_ID

    def __init__(self, variant, ckpt, device):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.device = device
        # processor ALWAYS from the base repo (proven path; full-FT ckpts don't
        # carry the feature-extractor config)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        src = ckpt if variant == "fullft" else self.model_id
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            src, dtype=torch.bfloat16, device_map=device)
        if variant == "lora":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ckpt)
            model = model.merge_and_unload()
        model.eval()
        self.model = model
        self.torch = torch

    def generate(self, clip) -> str:
        audio_span = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{audio_span}{PROMPT}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = self.processor(text=text, audio=[clip], sampling_rate=16000,
                                return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # cast floating inputs (input_features, ...) to model dtype, as run_probe.py does
        inputs = {k: (v.to(self.model.dtype) if self.torch.is_floating_point(v) else v)
                  for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=False, num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
                stop_strings=STOP_STRINGS,
                tokenizer=self.processor.tokenizer)
        new_ids = out[0][input_len:]
        text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        return cut_at_stop(text), int(len(new_ids))


class OmniRunner:
    """Thinker-only, processor chat template — the path musiclistenbench/backends/qwen_omni_backend.py uses."""

    model_id = OMNI_MODEL_ID

    def __init__(self, variant, ckpt, device):
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        self.device = device
        # full-FT ckpts carry processor config + tokenizer (proven by step3 evals)
        src = ckpt if variant == "fullft" else self.model_id
        self.processor = Qwen2_5OmniProcessor.from_pretrained(src)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            src, torch_dtype=torch.bfloat16, device_map=device)
        if variant == "lora":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ckpt)
            model = model.merge_and_unload()
        model.eval()
        self.model = model
        self.torch = torch
        self.sr = self.processor.feature_extractor.sampling_rate

    def generate(self, clip) -> str:
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": PROMPT}]},
        ]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt_text, audio=[clip],
                                return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=False, num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
                stop_strings=STOP_STRINGS,
                tokenizer=self.processor.tokenizer)
        new_ids = out[0][input_len:]
        text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        return cut_at_stop(text), int(len(new_ids))


RUNNERS = {"qwen2-audio": Qwen2AudioRunner, "qwen2.5-omni": OmniRunner}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--variant", required=True, choices=["base", "lora", "fullft"])
    ap.add_argument("--ckpt-step", type=int, default=None,
                    help="explicit GRPO checkpoint step (default: latest in the run dir)")
    ap.add_argument("--manifest", default=str(DATA / "manifest.jsonl"))
    ap.add_argument("--audio-root", default=str(DATA))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace(".", "_").replace("-", "_")
    if args.out is None:
        args.out = str(RESULTS / f"preds_{safe}_{args.variant}.jsonl")

    ckpt = "base"
    if args.variant != "base":
        base_dir = LORA_DIRS[args.model] if args.variant == "lora" else FULLFT_DIRS[args.model]
        ckpt = str(latest_checkpoint(base_dir, args.ckpt_step))

    with open(args.manifest) as f:
        items = sorted((json.loads(l) for l in f if l.strip()), key=lambda r: r["ytid"])
    if args.limit is not None:
        items = items[: args.limit]
    if args.num_shards > 1:
        items = items[args.shard:: args.num_shards]

    # resume: skip ytids already written by a previous (crashed) run of this config
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["ytid"])
                except Exception:
                    pass
        print(f"resuming: {len(done)} items already in {args.out}")
    todo = [it for it in items if it["ytid"] not in done]
    print(f"model={args.model} variant={args.variant} ckpt={ckpt} "
          f"clips={len(todo)} (of {len(items)}) device={args.device}")

    runner = RUNNERS[args.model](args.variant, ckpt, args.device)
    sanitize_generation(runner.model, MAX_NEW_TOKENS)

    # audio loading rate: qwen2-audio wants 16 kHz; omni uses its feature extractor's rate
    target_sr = 16000 if args.model == "qwen2-audio" else runner.sr

    t_start = time.time()
    with open(args.out, "a") as out_f:
        for i, it in enumerate(todo):
            wav = Path(args.audio_root) / it["wav"]
            t0 = time.time()
            clip = load_audio(wav, target_sr)
            gen, n_new = runner.generate(clip)
            dt = time.time() - t0
            out_f.write(json.dumps({
                "ytid": it["ytid"], "wav": it["wav"], "ref": it["caption"],
                "gen": gen, "model": args.model, "variant": args.variant,
                "ckpt": ckpt, "prompt": PROMPT,
                "n_words": len(gen.split()),
                "n_new_tokens": n_new,
                "hit_max_new_tokens": bool(n_new >= MAX_NEW_TOKENS),
                "wall_s": round(dt, 2),
            }) + "\n")
            out_f.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t_start
                print(f"[{i + 1}/{len(todo)}] {el:.0f}s elapsed, "
                      f"{el / (i + 1):.1f}s/item", flush=True)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
