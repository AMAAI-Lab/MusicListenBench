"""GRPO training on the base-only train set (data/train.json --
Q1_melody/Q2_harmony/Q3_timbre/Q4_rhythm, `expected_role == "base"`, no
invariance/equivariance re-renders). Since there are no re-renders or content
edits in this data, the reward is parse (0.2) + correctness (1.0) only
(`training/reward.py`), plus the small anti-collapse letter nudge described in
the paper's Appendix D. Both clips of a trial are presented as one audio file.

Algorithm (minimal GRPO, one gradient update per on-policy batch -- no PPO
clipping, since old==new policy at generation time within a batch):
  for each of `--prompts-per-step` prompts (per GPU):
    - sample `--rollouts-per-prompt` (G) completions from the current policy
    - reward each rollout (reward.py); if every rollout in the group scored
      identically (zero-advantage, no gradient signal), resample fresh
      rollouts up to `--max-resample-tries` times before giving up
      (DAPO-style "dynamic sampling" -- see `sample_group`). Every prompt
      still gets a forward+backward call regardless of outcome -- `train_
      model` is DDP-wrapped, and it must be called exactly
      `--prompts-per-step` times on every rank every step, or DDP's
      per-forward buffer broadcast falls out of lockstep across ranks and
      hangs until the NCCL watchdog times out. A group that's still
      zero-advantage after every resample attempt just contributes ~zero
      policy gradient on its own, same as the pre-resampling code's
      fallback.
    - normalise rewards into a per-rollout advantage within the group:
      (r - mean) / (std + eps)
    - recompute per-token log-probs of the sampled tokens under the current
      policy (with grad) and under the frozen reference (LoRA adapter
      disabled, no grad)
    - loss = -advantage * policy_logprob + kl_coef * k3_kl(policy, ref),
      masked to real (non-pad) generated tokens, averaged over tokens
    - accumulate gradients across the batch, then one optimizer step

LoRA only touches `model.language_model.*` (the decision path), never
`model.audio_tower.*`: the audio encoder stays frozen in the LoRA runs.

Multi-GPU is real data parallelism via `accelerate`, not independent shards:
every GPU holds a full model replica and processes its own
`--prompts-per-step` prompts each step, gradients are all-reduced across
every GPU before the one optimizer step, so all replicas stay identical.

Run (single GPU):  python -m musiclistenbench.training.train_grpo
Run (all 8 GPUs):  accelerate launch --multi_gpu --num_processes 8 -m musiclistenbench.training.train_grpo
"""

import argparse
import json
import os
import random
from contextlib import nullcontext

import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator

from musiclistenbench import paths
from musiclistenbench.training.dataset import EpochSampler, load_train_set
from musiclistenbench.training.model_adapters import TRAINABLE, get_adapter
from musiclistenbench.training.reward import balance_adjustment, compute_reward

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
TARGET_SR = 16000
MODEL_DTYPE = torch.bfloat16
AUDIO_SPAN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def load_clip(path, target_sr):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)


def build_prompt_text(question):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{AUDIO_SPAN}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def to_model_dtype(mm_kwargs, dtype):
    # Phi-4's processor returns None for the unused modality's keys (image tensors when
    # only audio is passed); is_floating_point() errors on None, so gate on is_tensor first.
    return {k: (v.to(dtype) if (torch.is_tensor(v) and torch.is_floating_point(v)) else v)
            for k, v in mm_kwargs.items()}


def _frac_pred_b(preds):
    letters = [p for p in preds if p is not None]
    return float(np.mean([p == "B" for p in letters])) if letters else float("nan")


class PredLetterBalance:
    """Per-task running estimate of which letter the policy currently
    over-predicts, feeding `balance_adjustment`. Tracks a plain EMA of
    "fraction of parsed rollouts that said B" per task -- symmetric by
    construction, since it is defined purely from the policy's own recent
    outputs, not from any assumption about which letter/answer is the
    'hard' one for a given task. Every task (Q1-Q4, same/different or
    first/second) is tracked the same way with no per-task special-casing.
    """

    def __init__(self, decay=0.95):
        self.decay = decay
        self.frac_b = {}  # task -> EMA fraction of parsed rollouts predicting 'B'

    def skew(self, task):
        return self.frac_b.get(task, 0.5) - 0.5

    def update(self, task, preds):
        letters = [p for p in preds if p is not None]
        if not letters:
            return
        obs = sum(1 for p in letters if p == "B") / len(letters)
        prev = self.frac_b.get(task, 0.5)
        self.frac_b[task] = self.decay * prev + (1 - self.decay) * obs


def sample_group(gen_model, processor, example, args, device, pad_id, balance_tracker, adapter=None):
    """No-grad phase: build the prompt, sample G rollouts, reward them, and
    -- if the whole group scores identically (zero-advantage, no gradient
    signal) -- resample fresh rollouts up to `--max-resample-tries` times
    before giving up. This is the DAPO-style "dynamic sampling" fix for
    GRPO's known failure mode where small G + binary reward makes
    zero-advantage groups common, and the groups that DO survive
    increasingly reflect whichever direction the policy already leans
    rather than genuine signal.

    Cheap (no_grad only, no backward graph held) relative to the forward-
    with-grad pass in `compute_group_loss` that immediately follows it per
    prompt in the main loop.

    IMPORTANT: this always returns usable rollouts, even if every attempt
    stayed zero-advantage (`stats["gave_up_zero_advantage"] = True` in that
    case) -- the caller must NOT skip `compute_group_loss`/`.backward()` for
    such a prompt. `train_model` is DDP-wrapped, and DDP broadcasts buffers
    on every `forward()` call regardless of `no_sync()`; if different ranks
    called it a different number of times in the same step (e.g. because
    each rank independently decided to drop a different number of prompts),
    the ranks' collectives fall out of lockstep and whichever rank is left
    waiting on a peer that already moved on hangs until the NCCL watchdog
    times out. A zero-advantage group still contributes ~zero policy
    gradient on its own (advantages are ~0 when std~0) -- exactly like the
    original no-resampling code -- so proceeding is harmless, just a wasted
    forward/backward in the (rare, post-resampling) worst case.

    Returns (inputs, prefix_len, generated, new_tokens, rewards_t, stats).
    """
    # adapter=None keeps the original Qwen2-Audio path exactly (so the fullft/sft
    # trainers that import sample_group are unaffected); an adapter routes prompt
    # building + the processor call + sample rate through model_adapters.py.
    sr = adapter.target_sr if adapter is not None else TARGET_SR
    clip = load_clip(example["audio_path"], sr)
    if adapter is not None:
        text = adapter.build_prompt_text(processor, example["question"])
        inputs = adapter.processor_inputs(processor, text, clip)
    else:
        text = build_prompt_text(example["question"])
        inputs = processor(text=text, audio=[clip], sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    prefix_len = inputs["input_ids"].shape[1]
    G = args.rollouts_per_prompt
    # Read the tracker's current skew ONCE, before any rollouts for this
    # prompt are generated -- rewards for this whole group are judged
    # against the policy's skew coming INTO this prompt, not updated
    # mid-resample, so the nudge never reacts to its own outcome.
    skew = balance_tracker.skew(example["task"])

    # Rollouts must decode in eval mode. transformers' GradientCheckpointingLayer
    # only checkpoints when `self.training` is True, and in that state it silently
    # strips `past_key_values=None` -- so with --grad-checkpointing (full-FT trainer)
    # an incremental decode runs with a half-built KV cache and SDPA dies with
    # "Key and Value must have the same sequence length" (hit on qwen2.5-omni,
    # whose decoder layer goes through that wrapper). Every dropout in these
    # models' configs is 0.0, so eval mode is numerically identical to train mode
    # here; we restore the original mode so the loss forward below still gets
    # checkpointed. No-op for the LoRA trainers, which never enable checkpointing.
    was_training = gen_model.training
    for attempt in range(1, args.max_resample_tries + 2):  # first try + retries
        with torch.no_grad():
            gen_model.eval()
            try:
                generated = gen_model.generate(
                    **inputs, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    num_return_sequences=G, max_new_tokens=args.max_new_tokens, pad_token_id=pad_id,
                )
            finally:
                if was_training:
                    gen_model.train()
        new_tokens = generated[:, prefix_len:]
        texts = processor.batch_decode(new_tokens, skip_special_tokens=True)

        rewards, parsed, corrects, preds = [], [], [], []
        for t in texts:
            r, p, c = compute_reward(t, example["gold"])
            r += balance_adjustment(p, skew, args.balance_coef)
            rewards.append(r)
            parsed.append(p is not None)
            corrects.append(c)
            preds.append(p)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        std = rewards_t.std(unbiased=False).item()

        stats = {
            "task": example["task"],
            "gold": example["gold"],
            "mean_reward": float(rewards_t.mean().item()),
            "parse_rate": float(np.mean(parsed)),
            "accuracy": float(np.mean(corrects)),
            "frac_pred_B": _frac_pred_b(preds),
            "resample_tries": attempt,
            "balance_skew": skew,
        }
        if std >= 1e-6:
            stats["gave_up_zero_advantage"] = False
            balance_tracker.update(example["task"], preds)
            return inputs, prefix_len, generated, new_tokens, rewards_t, stats

    # every attempt was zero-advantage -- proceed anyway with the last
    # attempt's rollouts (advantages will come out ~0, contributing ~no
    # policy gradient, same as the pre-resampling code's fallback) rather
    # than skip the forward/backward call; see the DDP lockstep note above.
    stats["gave_up_zero_advantage"] = True
    balance_tracker.update(example["task"], preds)
    return inputs, prefix_len, generated, new_tokens, rewards_t, stats


def compute_group_loss(train_model, gen_model, inputs, prefix_len, generated, new_tokens, rewards_t,
                        args, device, pad_id, dtype=None):
    """Expensive forward-with-grad phase, called for every prompt in the
    batch regardless of whether `sample_group` found a nonzero-advantage
    group (see that function's docstring for why this must never be
    skipped). Called immediately followed by `.backward()` per prompt in
    the main loop, so only one prompt's
    activation graph is held in memory at a time."""
    G = generated.shape[0]
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std(unbiased=False) + 1e-6)

    mm_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    mm_kwargs = to_model_dtype(mm_kwargs, dtype if dtype is not None else MODEL_DTYPE)
    mm_rep = {k: (v.repeat(G, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
              for k, v in mm_kwargs.items()}

    attn_prefix = torch.ones((G, prefix_len), dtype=torch.long, device=device)
    attn_gen = (new_tokens != pad_id).long()
    full_attn = torch.cat([attn_prefix, attn_gen], dim=1)

    logits_policy = train_model(input_ids=generated, attention_mask=full_attn, **mm_rep).logits

    T = new_tokens.shape[1]
    sl = slice(prefix_len - 1, prefix_len - 1 + T)
    lp_policy = torch.log_softmax(logits_policy[:, sl, :].float(), dim=-1) \
        .gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)
    mask = attn_gen.float()

    # The KL reference is a SECOND full forward over the G rollouts (adapter
    # disabled). It's only needed when the KL term is actually used, so --kl-coef 0
    # skips it entirely (DAPO-style KL-free RL) -- nearly halving the per-prompt
    # forward cost. When --kl-coef > 0 this path is byte-for-byte the original.
    if args.kl_coef > 0:
        with torch.no_grad(), gen_model.disable_adapter():
            logits_ref = gen_model(input_ids=generated, attention_mask=full_attn, **mm_rep).logits
        lp_ref = torch.log_softmax(logits_ref[:, sl, :].float(), dim=-1) \
            .gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)
        diff = lp_ref - lp_policy
        kl_tok = torch.exp(diff) - diff - 1   # k3 estimator, unbiased, >= 0
    else:
        kl_tok = torch.zeros_like(lp_policy)

    per_token_loss = -(advantages.unsqueeze(-1) * lp_policy) + args.kl_coef * kl_tok
    denom = mask.sum().clamp(min=1)
    loss = (per_token_loss * mask).sum() / denom
    mean_kl = (kl_tok * mask).sum().item() / denom.item()
    return loss, mean_kl


def gather_mean(accelerator, value):
    """All-reduce a python scalar (already averaged over this process's own
    batch) into the true mean over every GPU's batch this step."""
    t = torch.tensor([value], dtype=torch.float32, device=accelerator.device)
    return accelerator.gather(t).mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", default=paths.TRAIN_JSON)
    parser.add_argument("--audio-root", default=paths.AUDIO_DIR)
    parser.add_argument("--out-dir", default=os.path.join(paths.CHECKPOINT_DIR, "grpo_checkpoints"))
    parser.add_argument("--model", default="qwen2-audio", choices=TRAINABLE,
                         help="which audio-LLM to GRPO-train (see musiclistenbench/training/model_adapters.py). "
                              "the paper trains qwen2-audio, qwen2.5-omni, audio-flamingo3 and phi4-multimodal.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--prompts-per-step", type=int, default=4,
                         help="B: prompts per gradient step, PER GPU -- effective global batch is B * num_processes")
    parser.add_argument("--rollouts-per-prompt", type=int, default=8, help="G: rollouts per prompt")
    parser.add_argument("--max-resample-tries", type=int, default=4,
                         help="extra generation attempts for a zero-advantage group (all G rollouts "
                              "scored identically) before giving up and dropping the prompt's "
                              "gradient contribution for this step")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.10)
    parser.add_argument("--balance-coef", type=float, default=0.10,
                         help="anti-collapse reward nudge weight, applied per rollout via "
                              "reward.balance_adjustment -- discourages the policy from settling "
                              "on one letter regardless of gold, symmetric across A/B and across "
                              "all tasks (same/different, first/second, ...); 0 disables it")
    parser.add_argument("--balance-ema-decay", type=float, default=0.95,
                         help="decay for PredLetterBalance's per-task running fraction of "
                              "rollouts predicting 'B', used to compute the skew fed into "
                              "--balance-coef")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    # TF32: run fp32 matmuls on the A100 tensor cores. Storage stays fp32, so the
    # AF3 embed_positions bf16 bug is untouched -- this is pure matmul throughput
    # (~2-4x on fp32), with no measurable effect on the trained model. Safe for all.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    log_path = args.log_file or os.path.join(args.out_dir, "train_log.jsonl")

    from peft import LoraConfig, get_peft_model

    adapter = get_adapter(args.model)          # qwen2-audio == the original hardcoded path
    processor, base_model = adapter.load()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=adapter.lora_target_regex, task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.train()
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # accelerator.prepare places the model on this process's GPU and, when
    # launched with --num_processes > 1, wraps it in DistributedDataParallel
    # so every .backward() call all-reduces gradients across every GPU.
    model, optimizer = accelerator.prepare(model, optimizer)
    gen_model = accelerator.unwrap_model(model)   # for .generate() / .disable_adapter()

    examples = load_train_set(args.train_json, args.audio_root)
    if accelerator.is_main_process:
        print(f"model: {args.model} ({adapter.repo_id}), dtype={adapter.dtype}")
        print(f"loaded {len(examples)} base-only train examples from {args.train_json}")
        print(f"world_size={accelerator.num_processes}, effective batch = "
              f"{args.prompts_per_step} prompts/GPU x {accelerator.num_processes} GPUs "
              f"x {args.rollouts_per_prompt} rollouts = "
              f"{args.prompts_per_step * accelerator.num_processes * args.rollouts_per_prompt} rollouts/step")
    # distinct shuffle per process so 8 GPUs don't all train on the same prompts each step
    rng = random.Random(args.seed + accelerator.process_index)
    sampler = EpochSampler(examples, rng)
    # Per-process tracker is fine here -- it only steers exploration/reward
    # for that process's own rollouts, it doesn't need to match bit-for-bit
    # across ranks the way DDP call counts do.
    balance_tracker = PredLetterBalance(decay=args.balance_ema_decay)

    log_f = open(log_path, "w") if accelerator.is_main_process else None

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        batch = sampler.sample(args.prompts_per_step)

        # Sample + forward-with-grad + immediate backward, one prompt at a
        # time, so only one prompt's no-grad rollout tensors (audio input
        # features, generated token ids) AND activation graph are held in
        # memory at once, rather than accumulating a whole batch's worth of
        # no-grad tensors before any of them are consumed. `optimizer.step()`
        # only runs once at the end of the step, so the policy weights are
        # identical for every prompt regardless of order -- interleaving
        # sampling and backward like this changes nothing about the training
        # math, only peak memory. Every prompt in the batch gets a
        # forward+backward call here, even ones that gave up zero-advantage
        # after resampling -- `train_model` is DDP-wrapped, and it MUST be
        # called exactly `prompts_per_step` times on every rank every step
        # (see the note on `sample_group`); skipping calls per-rank desyncs
        # DDP's buffer broadcast and hangs until the NCCL watchdog times out.
        # Only all-reduce gradients on the last micro-batch of the step.
        step_stats = []
        for i, ex in enumerate(batch):
            inputs, prefix_len, generated, new_tokens, rewards_t, stats = sample_group(
                gen_model, processor, ex, args, device, pad_id, balance_tracker, adapter)
            step_stats.append(stats)

            is_last = i == len(batch) - 1
            sync_ctx = nullcontext() if is_last else accelerator.no_sync(model)
            with sync_ctx:
                loss, mean_kl = compute_group_loss(
                    model, gen_model, inputs, prefix_len, generated, new_tokens, rewards_t,
                    args, device, pad_id, adapter.dtype)
                stats["mean_kl"] = mean_kl
                accelerator.backward(loss / args.prompts_per_step)

        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        pred_b_vals = [s["frac_pred_B"] for s in step_stats if not np.isnan(s["frac_pred_B"])]
        local_agg = {
            "mean_reward": float(np.mean([s["mean_reward"] for s in step_stats])),
            "parse_rate": float(np.mean([s["parse_rate"] for s in step_stats])),
            "accuracy": float(np.mean([s["accuracy"] for s in step_stats])),
            "mean_kl": float(np.mean([s["mean_kl"] for s in step_stats])),
            "zero_advantage_rate": float(np.mean([s["gave_up_zero_advantage"] for s in step_stats])),
            "mean_resample_tries": float(np.mean([s["resample_tries"] for s in step_stats])),
            # frac of parsed rollouts predicting 'B', and the true rate of gold=='B' in this
            # batch -- the gap between these two is the real-time reward-hacking alarm from
            # the checkpoint-100-vs-500 pendulum: a growing gap means the policy is drifting
            # toward a letter prior instead of tracking the actual answer distribution.
            "frac_pred_B": float(np.mean(pred_b_vals)) if pred_b_vals else 0.0,
            "gold_B_rate": float(np.mean([1.0 if s["gold"] == "B" else 0.0 for s in step_stats])),
        }
        # gather_mean averages each metric over every GPU's batch this step,
        # not just the main process's own slice.
        global_agg = {k: gather_mean(accelerator, v) for k, v in local_agg.items()}

        if accelerator.is_main_process:
            by_task = {}
            by_task_gold = {}
            for s in step_stats:
                by_task.setdefault(s["task"], []).append(s["accuracy"])
                by_task_gold.setdefault(f"{s['task']}:{s['gold']}", []).append(s["accuracy"])
            per_task = {t: float(np.mean(v)) for t, v in by_task.items()}   # main process's own batch only
            per_task_gold = {k: float(np.mean(v)) for k, v in by_task_gold.items()}

            record = {
                "step": step, **global_agg,
                "per_task_main_process_only": per_task,
                "per_task_gold_main_process_only": per_task_gold,
                # main process's own PredLetterBalance state -- per-task skew
                # (frac predicting 'B' minus 0.5) feeding --balance-coef's
                # reward nudge; not cross-GPU aggregated, same scope cut as
                # the other per_task_*_main_process_only fields above.
                "balance_skew_main_process_only": {t: balance_tracker.skew(t) for t in balance_tracker.frac_b},
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            if step % args.log_every == 0:
                print(f"[step {step}/{args.steps}] reward={global_agg['mean_reward']:.3f} "
                      f"acc={global_agg['accuracy']:.3f} parse={global_agg['parse_rate']:.3f} "
                      f"kl={global_agg['mean_kl']:.4f} zero_adv={global_agg['zero_advantage_rate']:.2f} "
                      f"frac_B={global_agg['frac_pred_B']:.2f} gold_B_rate={global_agg['gold_B_rate']:.2f}")

        accelerator.wait_for_everyone()
        if step % args.save_every == 0 or step == args.steps:
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(args.out_dir, f"checkpoint-{step}")
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                print(f"saved LoRA adapter -> {ckpt_dir}")
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        log_f.close()
        print(f"\ntraining log -> {log_path}")


if __name__ == "__main__":
    main()
