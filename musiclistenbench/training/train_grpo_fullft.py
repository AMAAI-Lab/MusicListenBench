"""Full-parameter (no-LoRA) variant of `train_grpo.py`.

Same GRPO algorithm, same reward, same anti-collapse machinery, same
DDP-lockstep discipline as `train_grpo.py` -- the ONLY differences are:

  1. No PEFT/LoRA. Instead of wrapping the model in a LoRA adapter and
     training the low-rank deltas, we unfreeze real weights and train them
     directly. Which weights is chosen by `--trainable`:
       language_model  (default) -- `model.language_model.*` + `lm_head`
                        = the full decision path (7.75B params), the direct
                        no-LoRA analog of the original (which LoRA'd only the
                        attn/MLP projections *inside* language_model). Audio
                        encoder + projector stay frozen.
       lm+projector    -- adds `model.multi_modal_projector` (+5.25M): lets a
                        little perception adapt without touching the encoder.
       all             -- everything incl. `model.audio_tower` (8.4B):
                        "encoder surgery". Only this option can change what
                        the model actually *hears*; the other two can only
                        reroute the frozen encoder's existing features).

  2. The KL reference is a SEPARATE frozen copy of the base model, not
     `gen_model.disable_adapter()` (there is no adapter to disable now).
     `--kl-coef 0` skips loading it entirely (saves ~16.8 GB; DAPO-style
     KL-free RL). See the memory note below before choosing.

  3. Optimizer defaults to Adafactor, not AdamW -- full-param AdamW on 7.75B
     needs ~93 GB of state (params+grads+fp32 m,v) and does NOT fit one 80 GB
     card. Adafactor's factored second moment brings that to ~33 GB, which
     fits a single A100-80GB (and comfortably on the 2x A100 here). Pass
     `--optimizer adamw` only if you shard with FSDP (see commands file);
     bitsandbytes 8-bit Adam is not installed on this box.

MEMORY REALITY (this machine: 2x A100-80GB [cuda:0,1] + 6x A40-46GB
[cuda:2-7]). The LoRA run already OOM'd on the 46 GB A40s at step 47. Full
parameter is far heavier, so:
  * The A40s (46 GB) cannot hold a full-param 7B replica. Run this on the
    80 GB cards only: `CUDA_VISIBLE_DEVICES=0,1`.
  * Recommended fit-without-sharding config: `--optimizer adafactor` on 1-2
    of the 80 GB cards. ~33 GB optimizer/param/grad + ~16.8 GB reference
    (if `--kl-coef>0`) + activations. Drop the reference (`--kl-coef 0`) if
    it's tight.
  * `--optimizer adamw` or `--trainable all` on all 8 GPUs needs FSDP
    (accelerate config in the commands file). Note: `.generate()` and the
    frozen reference model under FSDP are fiddly (params are sharded and must
    be gathered) -- the plain non-FSDP path above is the tested-logic one.

Every full checkpoint is a ~15 GB model dir (no tiny adapter to save), so
`--save-every` defaults to 250 here, not 50 -- watch disk.

Run (recommended, 2x 80GB, no sharding):
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
    -m musiclistenbench.training.train_grpo_fullft --out-dir checkpoints/grpo_ckpt_fullft
"""

import argparse
import json
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
from accelerate import Accelerator

from musiclistenbench import paths
from musiclistenbench.training.dataset import EpochSampler, load_train_set
# Reuse everything that isn't LoRA-specific verbatim, so this variant can
# never silently drift from the LoRA script's reward / sampling / logging
# behavior -- only the model-parameterization and KL-reference bits differ.
from musiclistenbench.training.model_adapters import TRAINABLE, get_adapter
from musiclistenbench.training.train_grpo import (
    PredLetterBalance,
    gather_mean,
    sample_group,
    to_model_dtype,
)

# --trainable choices are universal keys; each model maps them to its own submodule
# prefixes via adapter.freeze_prefixes (musiclistenbench/training/model_adapters.py). For qwen2-audio that
# map is the original one (transformers 5.x: model.language_model=7.12B, lm_head=639M,
# model.audio_tower=637M, model.multi_modal_projector=5.25M).
TRAINABLE_CHOICES = ("language_model", "lm+projector", "all")


def set_trainable(model, prefixes, excludes=()):
    """Freeze everything, then unfreeze params whose name starts with any prefix in
    `prefixes` (None = unfreeze all) AND contains none of the substrings in `excludes`.
    Returns (trainable_count, total_count). Must be called BEFORE accelerator.prepare so
    FSDP/DDP see the final requires_grad flags. Raises if a non-None prefix set matches
    ZERO params -- a wrong per-model prefix would otherwise silently train nothing (print
    model.named_parameters() and fix the adapter).

    `excludes` carves modality-specific params back out of a too-coarse prefix. Phi-4-mm
    bakes per-layer speech AND vision LoRA adapters under `model.layers.*`, so the
    `language_model` prefix also unfreezes the vision LoRA -- which gets no gradient on
    audio-only batches and trips DDP's find_unused_parameters assertion at step 2. Excluding
    the vision LoRA by name keeps the trainable set == the used set, so no DDP flag is needed
    and the dead params stay out of the optimizer."""
    trainable, total = 0, 0
    for name, p in model.named_parameters():
        total += p.numel()
        on = (prefixes is None) or any(name.startswith(pre) for pre in prefixes)
        on = on and not any(ex in name for ex in excludes)
        p.requires_grad_(on)
        if on:
            trainable += p.numel()
    if prefixes is not None and trainable == 0:
        raise SystemExit(f"--trainable matched 0 params for prefixes {prefixes} "
                         f"(excludes={excludes}); fix this model's freeze_prefixes in "
                         f"musiclistenbench/training/model_adapters.py")
    return trainable, total


def last_n_layer_prefixes(model, n):
    """Param-name prefixes for the top-`n` language-model decoder layers plus lm_head
    and the multimodal projector -- a PARTIAL full-FT set (real weights, not LoRA).
    All lower LM layers and the audio encoder stay frozen. Backprop runs only through
    these layers, so both gradient and activation memory shrink with n."""
    import re
    ids = sorted({int(x) for name, _ in model.named_parameters()
                  for x in re.findall(r"language_model\.layers\.(\d+)\.", name)})
    if not ids:
        raise SystemExit("--train-last-n: found no language_model.layers.* params to train")
    n = min(n, len(ids))
    keep = ids[-n:]
    sample = next(name for name, _ in model.named_parameters() if "language_model.layers." in name)
    base = sample[:sample.index("language_model.layers.")]      # e.g. "model." (or "")
    prefixes = tuple(f"{base}language_model.layers.{i}." for i in keep)
    prefixes += (f"{base}multi_modal_projector.", "lm_head.")
    return prefixes


def build_optimizer(kind, params, lr):
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if kind == "adafactor":
        from transformers.optimization import Adafactor
        # explicit lr (relative_step=False) so --lr means what it says and the
        # schedule matches the LoRA script's fixed-lr AdamW behavior; factored
        # second moment is the whole point (memory), scale_parameter off so the
        # given lr isn't silently rescaled per-tensor.
        return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False,
                         warmup_init=False, weight_decay=0.0)
    raise ValueError(f"unknown optimizer {kind!r}")


def compute_group_loss(train_model, ref_model, inputs, prefix_len, generated, new_tokens,
                       rewards_t, args, device, pad_id, dtype=None, ref_device=None, logits_kwarg=None):
    """Identical math to train_grpo.compute_group_loss, except the KL
    reference log-probs come from a SEPARATE frozen `ref_model` (or are
    skipped entirely when `ref_model is None`, i.e. --kl-coef 0). Called for
    every prompt in the batch regardless of advantage (DDP lockstep -- see
    train_grpo.sample_group's docstring); one prompt's graph in memory at a
    time."""
    G = generated.shape[0]
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std(unbiased=False) + 1e-6)

    mm_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    mm_kwargs = to_model_dtype(mm_kwargs, dtype if dtype is not None else torch.bfloat16)
    mm_rep = {k: (v.repeat(G, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
              for k, v in mm_kwargs.items()}

    attn_prefix = torch.ones((G, prefix_len), dtype=torch.long, device=device)
    attn_gen = (new_tokens != pad_id).long()
    full_attn = torch.cat([attn_prefix, attn_gen], dim=1)

    # Only the T generated positions enter the loss, so when the model supports it
    # (adapter.logits_to_keep_kwarg) compute logits for just the last T+1 positions instead of
    # the whole audio+text sequence. The full [G, seq, vocab] fp32 logits are the multi-GB
    # tensor that pushes fp32 AF3 over 80GB; this is identical math -- the last T+1 positions
    # are exactly [prefix_len-1 : prefix_len+T), so slice(0, T) here == the full-seq slice
    # below. Unset (Qwen/Phi-4) -> full logits, original behavior.
    T = new_tokens.shape[1]
    if logits_kwarg:
        lk_kwargs = {logits_kwarg: T + 1}
        sl = slice(0, T)                        # model returned only the last T+1 positions
    else:
        lk_kwargs = {}
        sl = slice(prefix_len - 1, prefix_len - 1 + T)

    logits_policy = train_model(input_ids=generated, attention_mask=full_attn,
                                **lk_kwargs, **mm_rep).logits
    lp_policy = torch.log_softmax(logits_policy[:, sl, :].float(), dim=-1) \
        .gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)

    mask = attn_gen.float()
    if ref_model is not None:
        # The KL reference may live on a DIFFERENT GPU than the policy (--ref-device-offset;
        # e.g. fp32 AF3 policy on the A100s + reference on a spare A40, so both fit). Ship the
        # ref forward's inputs to that GPU and bring the per-token ref log-probs back for the
        # KL math. When ref_device == policy device this is a no-op (original behavior).
        rdev = ref_device if ref_device is not None else device
        with torch.no_grad():
            if rdev != device:
                gen_r = generated.to(rdev)
                attn_r = full_attn.to(rdev)
                mm_r = {k: (v.to(rdev) if torch.is_tensor(v) else v) for k, v in mm_rep.items()}
            else:
                gen_r, attn_r, mm_r = generated, full_attn, mm_rep
            logits_ref = ref_model(input_ids=gen_r, attention_mask=attn_r, **lk_kwargs, **mm_r).logits
        lp_ref = torch.log_softmax(logits_ref[:, sl, :].float(), dim=-1) \
            .gather(-1, new_tokens.to(logits_ref.device).unsqueeze(-1)).squeeze(-1)
        diff = lp_ref.to(lp_policy.device) - lp_policy
        kl_tok = torch.exp(diff) - diff - 1   # k3 estimator, unbiased, >= 0
    else:
        kl_tok = torch.zeros_like(lp_policy)

    per_token_loss = -(advantages.unsqueeze(-1) * lp_policy) + args.kl_coef * kl_tok
    denom = mask.sum().clamp(min=1)
    loss = (per_token_loss * mask).sum() / denom
    mean_kl = (kl_tok * mask).sum().item() / denom.item()
    return loss, mean_kl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", default=paths.TRAIN_JSON)
    parser.add_argument("--audio-root", default=paths.AUDIO_DIR)
    parser.add_argument("--out-dir", default=os.path.join(paths.CHECKPOINT_DIR, "grpo_ckpt_fullft"))
    parser.add_argument("--model", default="qwen2-audio", choices=TRAINABLE,
                        help="which audio-LLM to full-FT (see musiclistenbench/training/model_adapters.py). "
                             "the paper trains qwen2-audio, qwen2.5-omni, audio-flamingo3 and phi4-multimodal.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--init-from", default=None,
                        help="warm-start POLICY weights from this save_pretrained dir "
                             "(e.g. .../checkpoint-500) instead of the base MODEL_ID. "
                             "WARM START, NOT a bit-exact resume: save_pretrained stores "
                             "only model weights, so Adafactor optimizer state and the "
                             "data-sampler position are NOT restored. The KL reference "
                             "stays the ORIGINAL base model (faithful continuation of the "
                             "first run's objective).")
    parser.add_argument("--step-offset", type=int, default=0,
                        help="added to every step number for checkpoint naming, the "
                             "train_log 'step' field, and progress prints. Set to the last "
                             "completed step when continuing (e.g. 500) so new checkpoints "
                             "are checkpoint-750.. and never clobber old ones; the log is "
                             "APPENDED (not overwritten) whenever this is > 0.")
    parser.add_argument("--prompts-per-step", type=int, default=4,
                        help="B: prompts per gradient step, PER GPU")
    parser.add_argument("--rollouts-per-prompt", type=int, default=8, help="G: rollouts per prompt")
    parser.add_argument("--max-resample-tries", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # full-param defaults: much smaller LR than LoRA's 1e-5, and Adafactor to
    # fit without sharding. See the module docstring's MEMORY REALITY note.
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    parser.add_argument("--trainable", choices=TRAINABLE_CHOICES, default="language_model",
                        help="which real weights to unfreeze (no LoRA), mapped per-model via "
                             "adapter.freeze_prefixes. 'language_model' = decision path (LM+lm_head), "
                             "encoder frozen; 'lm+projector' adds the multimodal projector; 'all' "
                             "unfreezes everything incl. the audio encoder (prefix-free, safe for any model).")
    parser.add_argument("--train-last-n", type=int, default=0,
                        help="PARTIAL full-FT: train only the top-N language-model decoder layers "
                             "(+ lm_head + projector); freeze all lower layers and the audio encoder. "
                             "Real full-weight updates on those layers (not LoRA). Shrinks BOTH grads and "
                             "backward activations, so it fits plain DDP on both 80GB cards (no FSDP) and "
                             "runs faster. 0 = off (use --trainable); overrides --trainable when >0.")
    parser.add_argument("--kl-coef", type=float, default=0.10,
                        help="0 disables the KL term AND skips loading the reference model "
                             "(saves ~16.8 GB; DAPO-style KL-free RL)")
    parser.add_argument("--ref-device-offset", type=int, default=0,
                        help="put the KL reference model on cuda:{local_rank + OFFSET} instead of "
                             "sharing the policy's GPU (0 = same GPU, default). Offloads a heavy fp32 "
                             "reference onto spare cards so policy+reference fit: e.g. policy on the "
                             "2x A100 (cuda:0,1) + reference on 2 spare A40s via --ref-device-offset 2 "
                             "with CUDA_VISIBLE_DEVICES=0,1,2,3. Only used when --kl-coef>0.")
    parser.add_argument("--balance-coef", type=float, default=0.10)
    parser.add_argument("--balance-ema-decay", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="recompute activations in the backward pass instead of storing "
                             "them: pure memory-for-compute (~20-30%% slower/step), NO effect on "
                             "the trained model. Frees ~activation memory so full-param fp32 AF3 "
                             "fits an 80GB card. Generation keeps its KV cache (fast).")
    parser.add_argument("--save-every", type=int, default=250,
                        help="full checkpoints are ~15 GB each -- higher than the LoRA script's 50")
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

    adapter = get_adapter(args.model)
    processor = adapter.load_processor()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    init_src = args.init_from or adapter.repo_id
    model = adapter.load_model(init_src)
    if accelerator.is_main_process:
        print(f"[full-FT] model={args.model} ({adapter.repo_id}), dtype={adapter.dtype}")
        if args.init_from:
            print(f"[full-FT] WARM START: policy weights from {init_src} (optimizer/sampler "
                  f"state NOT restored); KL reference stays base {adapter.repo_id}")
    if args.train_last_n > 0:
        trainable_prefixes = last_n_layer_prefixes(model, args.train_last_n)
        trainable_label = f"last-{args.train_last_n}-LM-layers+lm_head+projector"
    else:
        trainable_prefixes = adapter.freeze_prefixes[args.trainable]
        trainable_label = args.trainable
    # Modality-specific params to keep frozen even when a coarse prefix matches them
    # (Phi-4-mm's per-layer vision LoRA lives under model.layers.* but is unused on
    # audio-only batches -> DDP find_unused_parameters crash). Empty for every other model.
    trainable_excludes = adapter.freeze_excludes
    if trainable_excludes:
        trainable_label += f" (excl {','.join(trainable_excludes)})"
    n_train, n_total = set_trainable(model, trainable_prefixes, trainable_excludes)
    model.train()
    if accelerator.is_main_process:
        print(f"[full-FT] trainable={trainable_label}: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
              f"params trainable ({100*n_train/n_total:.1f}%), optimizer={args.optimizer}, lr={args.lr}")

    if args.grad_checkpointing:
        # Recompute activations in backward (memory-for-compute; identical math).
        # use_cache must be off for checkpointing to actually save memory on the
        # loss forward, but generation stays KV-cached via generation_config so
        # short-decode rollouts don't slow down. Enable BEFORE accelerator.prepare
        # so DDP wraps the checkpointed graph; use_reentrant=False for DDP safety.
        model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = True
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if accelerator.is_main_process:
            print("[full-FT] gradient checkpointing ON (activations recomputed in backward)")

    # Separate frozen reference for the KL term (there's no adapter to
    # disable). Skipped entirely when --kl-coef 0.
    ref_model = None
    ref_device = device
    if args.kl_coef > 0:
        if args.ref_device_offset != 0:
            ref_idx = accelerator.local_process_index + args.ref_device_offset
            if ref_idx >= torch.cuda.device_count():
                raise SystemExit(f"--ref-device-offset {args.ref_device_offset}: reference GPU "
                                 f"cuda:{ref_idx} not visible (only {torch.cuda.device_count()} GPUs). "
                                 f"Expose more with CUDA_VISIBLE_DEVICES (e.g. 0,1,2,3 = 2 policy + 2 ref).")
            ref_device = torch.device(f"cuda:{ref_idx}")
        ref_model = adapter.load_model()   # base weights, for the KL reference
        ref_model.to(ref_device).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        if accelerator.is_main_process:
            off = f"  [offloaded off policy GPU {device}]" if ref_device != device else ""
            print(f"[full-FT] loaded frozen reference model on {ref_device} for KL (coef={args.kl_coef}){off}")
    elif accelerator.is_main_process:
        print("[full-FT] --kl-coef 0: no reference model, KL term disabled")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(args.optimizer, trainable_params, args.lr)

    model, optimizer = accelerator.prepare(model, optimizer)
    gen_model = accelerator.unwrap_model(model)   # for .generate()

    # Under FSDP, params live sharded (1-D flat shards) outside the fwd/bwd window,
    # so .generate() on the unwrapped model fails ("'weight' must be 2-D"). Gather
    # full params just for the rollout (no_grad, short decode) via summon_full_params,
    # then reshard. Pure plumbing -- identical math and precision. DDP / single-GPU
    # keep the free nullcontext path.
    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    if is_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP

    def rollout_ctx():
        return (_FSDP.summon_full_params(model, recurse=True, writeback=False)
                if is_fsdp else nullcontext())

    examples = load_train_set(args.train_json, args.audio_root)
    if accelerator.is_main_process:
        print(f"loaded {len(examples)} base-only train examples from {args.train_json}")
        print(f"world_size={accelerator.num_processes}, effective batch = "
              f"{args.prompts_per_step} prompts/GPU x {accelerator.num_processes} GPUs "
              f"x {args.rollouts_per_prompt} rollouts = "
              f"{args.prompts_per_step * accelerator.num_processes * args.rollouts_per_prompt} rollouts/step")
    rng = random.Random(args.seed + accelerator.process_index)
    sampler = EpochSampler(examples, rng)
    balance_tracker = PredLetterBalance(decay=args.balance_ema_decay)

    log_mode = "a" if args.step_offset > 0 else "w"
    log_f = open(log_path, log_mode) if accelerator.is_main_process else None

    total_steps = args.step_offset + args.steps
    for local_step in range(1, args.steps + 1):
        step = local_step + args.step_offset
        optimizer.zero_grad()
        batch = sampler.sample(args.prompts_per_step)

        # Same lockstep discipline as train_grpo: every prompt gets exactly one
        # forward+backward on every rank every step (never skip zero-advantage
        # groups), or DDP falls out of sync and hangs on the NCCL watchdog.
        step_stats = []
        for i, ex in enumerate(batch):
            with rollout_ctx():
                inputs, prefix_len, generated, new_tokens, rewards_t, stats = sample_group(
                    gen_model, processor, ex, args, device, pad_id, balance_tracker, adapter)
            step_stats.append(stats)

            is_last = i == len(batch) - 1
            sync_ctx = nullcontext() if is_last else accelerator.no_sync(model)
            with sync_ctx:
                loss, mean_kl = compute_group_loss(
                    model, ref_model, inputs, prefix_len, generated, new_tokens, rewards_t,
                    args, device, pad_id, adapter.dtype, ref_device=ref_device,
                    logits_kwarg=adapter.logits_to_keep_kwarg)
                stats["mean_kl"] = mean_kl
                accelerator.backward(loss / args.prompts_per_step)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        pred_b_vals = [s["frac_pred_B"] for s in step_stats if not np.isnan(s["frac_pred_B"])]
        local_agg = {
            "mean_reward": float(np.mean([s["mean_reward"] for s in step_stats])),
            "parse_rate": float(np.mean([s["parse_rate"] for s in step_stats])),
            "accuracy": float(np.mean([s["accuracy"] for s in step_stats])),
            "mean_kl": float(np.mean([s["mean_kl"] for s in step_stats])),
            "zero_advantage_rate": float(np.mean([s["gave_up_zero_advantage"] for s in step_stats])),
            "mean_resample_tries": float(np.mean([s["resample_tries"] for s in step_stats])),
            "frac_pred_B": float(np.mean(pred_b_vals)) if pred_b_vals else 0.0,
            "gold_B_rate": float(np.mean([1.0 if s["gold"] == "B" else 0.0 for s in step_stats])),
        }
        global_agg = {k: gather_mean(accelerator, v) for k, v in local_agg.items()}

        if accelerator.is_main_process:
            by_task, by_task_gold = {}, {}
            for s in step_stats:
                by_task.setdefault(s["task"], []).append(s["accuracy"])
                by_task_gold.setdefault(f"{s['task']}:{s['gold']}", []).append(s["accuracy"])
            record = {
                "step": step, **global_agg,
                "per_task_main_process_only": {t: float(np.mean(v)) for t, v in by_task.items()},
                "per_task_gold_main_process_only": {k: float(np.mean(v)) for k, v in by_task_gold.items()},
                "balance_skew_main_process_only": {t: balance_tracker.skew(t) for t in balance_tracker.frac_b},
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            if step % args.log_every == 0:
                print(f"[step {step}/{total_steps}] reward={global_agg['mean_reward']:.3f} "
                      f"acc={global_agg['accuracy']:.3f} parse={global_agg['parse_rate']:.3f} "
                      f"kl={global_agg['mean_kl']:.4f} zero_adv={global_agg['zero_advantage_rate']:.2f} "
                      f"frac_B={global_agg['frac_pred_B']:.2f} gold_B_rate={global_agg['gold_B_rate']:.2f}")

        accelerator.wait_for_everyone()
        if step % args.save_every == 0 or step == total_steps:
            # get_state_dict gathers sharded/fp32 weights -- call on ALL ranks.
            state = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(args.out_dir, f"checkpoint-{step}")
                accelerator.unwrap_model(model).save_pretrained(
                    ckpt_dir, state_dict=state, safe_serialization=True)
                processor.save_pretrained(ckpt_dir)   # so the ckpt dir loads standalone
                print(f"saved full model -> {ckpt_dir}")
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        log_f.close()
        print(f"\ntraining log -> {log_path}")


if __name__ == "__main__":
    main()
