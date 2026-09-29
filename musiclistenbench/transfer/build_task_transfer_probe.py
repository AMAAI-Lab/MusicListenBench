"""Builds a leave-one-task-out GRPO training set from data/train.json: every
item of one held-out task is dropped, the other three tasks are kept. It also
writes the mirror-image eval slice, only the held-out task's 250 clean test
items, for a fast evaluation of the transfer number.

`--held-out rhythm` gives the melody + harmony + timbre -> rhythm experiment of
the paper (Section 5.4, Figure 3); any of the four tasks can be held out.

Run from the repository root:
    for t in melody harmony timbre rhythm; do
      python -m musiclistenbench.transfer.build_task_transfer_probe --held-out $t
    done
"""

import argparse
import json
import os
from collections import Counter

from musiclistenbench import paths

TASKS = ["Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"]


def resolve_task(name):
    if name in TASKS:
        return name
    matches = [t for t in TASKS if t.split("_", 1)[1].lower() == name.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"--held-out {name!r} doesn't match any task; choices: "
                     f"{TASKS} (or their short forms, e.g. 'rhythm')")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held-out", required=True,
                        help="task to exclude from training and hold out for the transfer "
                             "test, e.g. 'rhythm' or 'Q4_rhythm'")
    parser.add_argument("--train-json", default=paths.TRAIN_JSON, help="the benchmark's train.json (read only)")
    parser.add_argument("--eval-json", default=paths.EVAL_JSON, help="the benchmark's eval.json (read only)")
    parser.add_argument("--out-dir", default=os.path.join(paths.DATA_DIR, "transfer"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    held_out = resolve_task(args.held_out)

    with open(args.train_json) as f:
        train_items = json.load(f)

    kept = [item for item in train_items if item["voice"][0].split("/")[0] != held_out]
    dropped_count = len(train_items) - len(kept)
    kept_counts = Counter(item["voice"][0].split("/")[0] for item in kept)

    assert held_out not in kept_counts, (
        f"{held_out} still present in the filtered train set -- filter logic is broken")
    assert dropped_count > 0, (
        f"no items had task == {held_out!r} in {args.train_json} -- wrong task name or file")

    train_out_path = os.path.join(args.out_dir, f"train_held_out_{held_out}.json")
    with open(train_out_path, "w") as f:
        json.dump(kept, f, indent=2)

    print(f"held out: {held_out}  (dropped {dropped_count} items from training)")
    for task in TASKS:
        print(f"  {task}: {kept_counts.get(task, 0)}")
    print(f"wrote {len(kept)} train items -> {train_out_path}")

    with open(args.eval_json) as f:
        eval_items = json.load(f)

    held_out_only = [item for item in eval_items if item["voice"][0].split("/")[0] == held_out]
    assert held_out_only, (
        f"no items had task == {held_out!r} in {args.eval_json} -- wrong task name or file")

    eval_out_path = os.path.join(args.out_dir, f"eval_held_out_only_{held_out}.json")
    with open(eval_out_path, "w") as f:
        json.dump(held_out_only, f, indent=2)
    print(f"wrote {len(held_out_only)} eval items (task {held_out} only) -> {eval_out_path}")


if __name__ == "__main__":
    main()
