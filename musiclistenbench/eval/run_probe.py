"""Runs a model against the single-combined-audio probe built by
`build_probe.py`. Reports two independent readings per item: real free-text
generation (does "one audio in, one letter out" actually elicit a clean
letter -- the thing SonicBench-style prompting is being tested for), and
forced-choice log P('A'|prompt) vs log P('B'|prompt) (a parse-free graded
signal).

This is the Qwen2-Audio-only runner. It was used for the leave-one-task-out
evaluations of Qwen2-Audio; every other result in the paper comes from
`run_probe_mm.py`, which reads the same forced-choice log-probabilities.

Run: python -m musiclistenbench.eval.run_probe --device cuda:0
"""

import argparse
import json
import os
import re

import numpy as np
import soundfile as sf

from musiclistenbench import paths

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
TARGET_SR = 16000
LETTERS = ["A", "B"]


def load_clip(path, target_sr):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)


def parse_letter(text):
    match = re.search(r"[AaBb]", text)
    return match.group(0).upper() if match else None


def score_letter_logprobs(model, tokenizer, prefix_ids, mm_kwargs, labels, device):
    """Length-normalised log P(label | prefix) for each candidate letter, one
    forced-decoding forward pass per label. `prefix_ids` is the (1, seq_len)
    prompt (audio features + question text); `mm_kwargs` carries the audio
    forward()-kwargs, identical across labels since only the trailing text
    token(s) change."""
    import torch

    out = {}
    for label in labels:
        label_ids = tokenizer(label, add_special_tokens=False).input_ids
        label_ids_t = torch.tensor(label_ids, device=device, dtype=torch.long).unsqueeze(0)
        full_ids = torch.cat([prefix_ids, label_ids_t], dim=1)
        attention_mask = torch.ones_like(full_ids)
        with torch.no_grad():
            logits = model(input_ids=full_ids, attention_mask=attention_mask, **mm_kwargs).logits
        start = prefix_ids.shape[1] - 1
        step_logprobs = torch.log_softmax(logits[0, start:start + len(label_ids), :].float(), dim=-1)
        token_logprobs = step_logprobs[torch.arange(len(label_ids)), label_ids_t[0]]
        out[label] = float(token_logprobs.mean())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", default=paths.EVAL_JSON)
    parser.add_argument("--audio-root", default=paths.AUDIO_DIR)
    parser.add_argument("--out", default=None,
                         help="defaults to results_qwen2_audio.jsonl, or "
                              "results_qwen2_audio_grpo.jsonl when --lora-checkpoint is set")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="cap total items, for a quick smoke run")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1,
                         help="split the eval set across N parallel processes (e.g. one per GPU)")
    parser.add_argument("--shard", type=int, default=0,
                         help="which shard this process handles, 0-indexed; use with --num-shards")
    parser.add_argument("--lora-checkpoint", default=None,
                         help="path to a train_grpo.py checkpoint dir (e.g. "
                              "checkpoints/grpo_checkpoints/checkpoint-500) to evaluate the "
                              "trained adapter instead of the base model")
    parser.add_argument("--full-checkpoint", default=None,
                         help="path to a train_grpo_fullft.py checkpoint dir (a standalone full "
                              "model, not a LoRA adapter) to evaluate instead of the base model; "
                              "mutually exclusive with --lora-checkpoint")
    args = parser.parse_args()
    if args.lora_checkpoint and args.full_checkpoint:
        parser.error("pass at most one of --lora-checkpoint / --full-checkpoint")
    if args.out is None:
        trained = args.lora_checkpoint or args.full_checkpoint
        default_name = "results_qwen2_audio_grpo.jsonl" if trained else "results_qwen2_audio.jsonl"
        args.out = os.path.join(paths.RESULTS_DIR, default_name)

    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # full-FT checkpoints are complete models -- load directly from the ckpt dir;
    # otherwise load the base model (and optionally bake in a LoRA adapter).
    model_src = args.full_checkpoint or MODEL_ID
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_src, dtype=torch.bfloat16, device_map=args.device,
    )
    if args.full_checkpoint:
        print(f"loaded full-FT checkpoint -> {args.full_checkpoint}")
    if args.lora_checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_checkpoint)
        model = model.merge_and_unload()   # bake the adapter in, so this is a plain Qwen2Audio again
        print(f"loaded LoRA checkpoint -> {args.lora_checkpoint}")
    model.eval()

    with open(args.eval_json) as f:
        items = json.load(f)
    if args.limit is not None:
        items = items[: args.limit]
    if args.num_shards > 1:
        items = items[args.shard::args.num_shards]

    audio_span = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    stats = {}
    n_unparsed = 0

    with open(args.out, "w") as out_f:
        for i, item in enumerate(items):
            voice_relpath = item["voice"][0]
            task = voice_relpath.split("/")[0]
            question = item["conversations"][0]["value"].replace("<audio>", "").strip()
            gold = item["conversations"][1]["value"].strip().upper()

            clip = load_clip(os.path.join(args.audio_root, voice_relpath), TARGET_SR)

            text = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{audio_span}{question}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            inputs = processor(text=text, audio=[clip], sampling_rate=TARGET_SR, return_tensors="pt")
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            prefix_ids = inputs["input_ids"]
            mm_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
            mm_kwargs = {k: (v.to(model.dtype) if torch.is_floating_point(v) else v)
                         for k, v in mm_kwargs.items()}
            logprobs = score_letter_logprobs(model, processor.tokenizer, prefix_ids, mm_kwargs,
                                              LETTERS, args.device)
            logprob_predicted = max(logprobs, key=logprobs.get)
            ranked = sorted(logprobs.values(), reverse=True)
            logprob_margin = ranked[0] - ranked[1] if len(ranked) > 1 else None
            logprob_correct = logprob_predicted == gold

            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            new_tokens = generated[:, input_len:]
            raw_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            predicted = parse_letter(raw_text)
            if predicted is None:
                n_unparsed += 1

            correct = predicted == gold
            task_stats = stats.setdefault(task, {
                "n": 0, "correct": 0, "unparsed": 0, "logprob_correct": 0,
            })
            task_stats["n"] += 1
            task_stats["correct"] += int(correct)
            task_stats["unparsed"] += int(predicted is None)
            task_stats["logprob_correct"] += int(logprob_correct)

            out_f.write(json.dumps({
                "voice": voice_relpath, "task": task, "question": question,
                "gold": gold, "raw_text": raw_text, "predicted": predicted, "correct": correct,
                "logprob_A": logprobs["A"], "logprob_B": logprobs["B"],
                "logprob_margin": logprob_margin,
                "logprob_predicted": logprob_predicted, "logprob_correct": logprob_correct,
            }) + "\n")

            if (i + 1) % 25 == 0:
                print(f"[shard {args.shard}] [{i + 1}/{len(items)}] running...")

    shard_note = f" (shard {args.shard}/{args.num_shards})" if args.num_shards > 1 else ""
    print()
    print(f"results{shard_note}:")
    print(f"{'task':<12} {'n':>5} {'gen_acc':>10} {'unparsed':>10} {'logprob_acc':>12}")
    total_n = total_correct = total_logprob_correct = 0
    for task, s in sorted(stats.items()):
        acc = s["correct"] / s["n"] if s["n"] else float("nan")
        lp_acc = s["logprob_correct"] / s["n"] if s["n"] else float("nan")
        print(f"{task:<12} {s['n']:>5} {acc:>10.3f} {s['unparsed']:>10} {lp_acc:>12.3f}")
        total_n += s["n"]
        total_correct += s["correct"]
        total_logprob_correct += s["logprob_correct"]
    overall_acc = total_correct / total_n if total_n else float("nan")
    overall_lp_acc = total_logprob_correct / total_n if total_n else float("nan")
    print(f"{'overall':<12} {total_n:>5} {overall_acc:>10.3f} {n_unparsed:>10} {overall_lp_acc:>12.3f}")
    print(f"\nper-item results -> {args.out}")


if __name__ == "__main__":
    main()
