"""Builds a controlled flip/stay probe by SELECTING pre-rendered second-clips from
the real test set (`audio_v2/`), in the identical `eval.json` schema so
`run_probe.py` consumes it unchanged.

Every real base test item -- a pair (first sound A, second sound B) with a known
answer -- yields TWO probe items, so the set is exactly balanced flip/stay:

  * FLIP  -> keep the first sound, swap the SECOND sound for the item's
            content-edited clip `eq_B` (the `equivariance` render: melody notes /
            chord quality / instrument / pulse-rate changed). This is exactly
            where the answer CHANGES, so the new gold is the flipped answer.
  * STAY  -> keep the first sound, swap the SECOND sound for a nuisance-perturbed
            clip `inv_<f>_B` (an `invariance` render: room / eq / hiss /
            transposition). The queried feature is untouched, so the answer STAYS
            the base answer. The nuisance factor is assigned ROUND-ROBIN across
            items, so every available perturbation is used evenly across the set
            (no random single-factor pick).

Nothing is synthesised here -- every clip is a real test-set render. The first
sound is always the item's real, untouched `base_A` (for Q1/Q2/Q3 the content edit
leaves A byte-identical anyway; every stay pairs the clean `base_A` with a
perturbed `inv_<f>_B`). Q4_rhythm's "which is faster?" is comparative and is the
ONE exception: its equivariance render swaps the two clips (`eq_B` is byte-
identical to `base_A`), so a Q4 flip must use the native `eq` pair -- first sound
`eq_A != base_A`, noted in the meta as first_sound="eq_A". Q4 stays still keep
`base_A`.

Because every A/B variant of an item renders to the same task window, flip and
stay combined clips are length-identical within a task, so duration never reveals
flip vs stay (asserted per task).

Balance: N base items/task -> N flip + N stay = 2N items/task; across the four
tasks 2000 items total (1000 flip / 1000 stay), every base case used once per
condition.

Metrics (score_perturb.py): false_flip_rate = error rate on STAY items (the model
"flipped" its answer when the edit shouldn't change it); miss_rate = error rate on
FLIP items (the model failed to flip when the edit does change the answer).

Test-only: reads `audio_v2/`'s test split; writes only audio_perturb/,
eval_perturb.json, eval_perturb.meta.jsonl. Never touches train.json or the base
eval.json / audio/.

Run: python -m musiclistenbench.benchmark.build_perturb_probe
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf

from musiclistenbench import paths

from musiclistenbench.benchmark.build_probe import build_combined_clip, make_entry

TASKS = ["Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"]


def load_test_items(manifest_path, clip_root=None):
    """item_id -> {'base': row, 'flip': eq_row, 'stay': {factor: inv_row}}."""
    items = defaultdict(lambda: {"base": None, "flip": None, "stay": {}})
    clip_root = clip_root or os.path.dirname(os.path.abspath(manifest_path))
    with paths.open_text(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not (isinstance(r, dict) and "expected_role" in r):
                continue
            if r["split"] != "test" or r["is_catch"] or r["task"] not in TASKS:
                continue
            r["clip_paths"] = [paths.resolve_clip_path(p, clip_root) for p in r["clip_paths"]]
            slot = items[r["item_id"]]
            if r["expected_role"] == "base":
                slot["base"] = r
            elif r["expected_role"] == "equivariance":
                slot["flip"] = r
            elif r["expected_role"] == "invariance":
                slot["stay"][r["factor_changed"]] = r
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=os.path.join(paths.GENERATED_DIR, "manifest.jsonl"),
                        help="item manifest written by the generator (plain or .gz)")
    parser.add_argument("--clip-root", default=None,
                        help="folder that relative clip paths in the manifest are relative to "
                             "(default: the manifest's own folder)")
    parser.add_argument("--out-dir", default=paths.DATA_DIR)
    parser.add_argument("--audio-subdir", default="audio_perturb")
    parser.add_argument("--json-name", default="eval_perturb.json")
    parser.add_argument("--silence-seconds", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--limit-per-task", type=int, default=None,
                        help="cap base items per task (for a quick run)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-existing-audio", action="store_true")
    args = parser.parse_args()

    clip_root = args.clip_root or os.path.dirname(os.path.abspath(args.manifest))
    items = load_test_items(args.manifest, args.clip_root)
    by_task = defaultdict(list)
    for iid, slot in items.items():
        if slot["base"] and slot["flip"] and slot["stay"]:
            by_task[slot["base"]["task"]].append((iid, slot))
    for t in TASKS:
        by_task[t].sort(key=lambda kv: kv[0])

    audio_root = os.path.join(args.out_dir, args.audio_subdir)
    if os.path.isdir(audio_root) and not args.keep_existing_audio:
        for t in TASKS:
            td = os.path.join(audio_root, t)
            if os.path.isdir(td):
                for fn in os.listdir(td):
                    if fn.endswith(".wav"):
                        os.remove(os.path.join(td, fn))
    os.makedirs(audio_root, exist_ok=True)

    rng = random.Random(args.seed)
    entries, meta_rows = [], []
    task_lengths = defaultdict(set)
    factor_counts = defaultdict(lambda: defaultdict(int))
    for task in TASKS:
        os.makedirs(os.path.join(audio_root, task), exist_ok=True)
        rows = by_task[task]
        if args.limit_per_task is not None:
            rows = rows[: args.limit_per_task]

        def emit(iid, condition, factor, clip_a, clip_b, first_sound, gold_row, base_answer):
            combined = build_combined_clip([clip_a, clip_b], args.silence_seconds, args.sample_rate)
            task_lengths[task].add(len(combined))
            name = f"{iid}__{condition}__{factor}.wav"
            sf.write(os.path.join(audio_root, task, name), combined, args.sample_rate, subtype="PCM_16")
            voice = f"{task}/{name}"
            entry = make_entry(task, gold_row, rng, voice)
            entries.append(entry)
            meta_rows.append({
                "voice": voice, "task": task, "item_id": iid,
                "condition": condition, "factor_changed": factor, "first_sound": first_sound,
                "base_answer": base_answer, "gold": entry["conversations"][1]["value"],
                "clip_a": os.path.relpath(clip_a, clip_root), "clip_b": os.path.relpath(clip_b, clip_root),
            })
            factor_counts[task][(condition, factor)] += 1

        for i, (iid, slot) in enumerate(rows):
            base, flip_row = slot["base"], slot["flip"]

            # ---- FLIP: content-edit the second sound -> answer changes ----
            flip_factor = flip_row["factor_changed"]
            if task == "Q4_rhythm":                          # comparative: only the native swap flips "faster"
                clip_a, first_sound = flip_row["clip_paths"][0], "eq_A"
            else:                                            # eq_A == base_A: first sound is the untouched base_A
                clip_a, first_sound = base["clip_paths"][0], "base_A"
            emit(iid, "flip", flip_factor, clip_a, flip_row["clip_paths"][1],
                 first_sound, flip_row, base["answer"])

            # ---- STAY: nuisance-perturb the second sound -> answer holds ----
            stay_factors = sorted(slot["stay"])              # available nuisances, deterministic order
            stay_factor = stay_factors[i % len(stay_factors)]  # round-robin: every nuisance used evenly
            inv_row = slot["stay"][stay_factor]
            emit(iid, "stay", stay_factor, base["clip_paths"][0], inv_row["clip_paths"][1],
                 "base_A", base, base["answer"])

        n = len(rows)
        # anti-hack: every flip and stay clip in a task must be length-identical
        assert len(task_lengths[task]) == 1, (task, task_lengths[task])
        dist = ", ".join(f"{fac}:{factor_counts[task][('stay', fac)]}"
                         for fac in sorted(slot["stay"]))
        print(f"{task}: {n} base items -> {n} flip / {n} stay, "
              f"combined length {next(iter(task_lengths[task]))} samples; "
              f"stay factors [{dist}]")

    json_path = os.path.join(args.out_dir, args.json_name)
    with open(json_path, "w") as f:
        json.dump(entries, f, indent=2)
    meta_path = os.path.splitext(json_path)[0] + ".meta.jsonl"
    with open(meta_path, "w") as f:
        for r in meta_rows:
            f.write(json.dumps(r) + "\n")

    n_flip = sum(r["condition"] == "flip" for r in meta_rows)
    print(f"\nwrote {len(entries)} items ({n_flip} flip / {len(entries) - n_flip} stay) "
          f"-> {json_path}")
    print(f"per-item condition metadata -> {meta_path}")
    print("every base test case -> 1 flip (content edit) + 1 stay (nuisance, round-robin); "
          "first sound is the real base_A (Q4_rhythm flip uses eq_A -- comparative swap); "
          "second sound is the real eq_B (flip) or inv_<f>_B (stay)")


if __name__ == "__main__":
    main()
