"""Lines up a leave-one-task-out checkpoint's per-task clean accuracy against the
matching untrained model. It only reads two per-item result files and reports the
change per task, flagging the held-out task.

A gain on the held-out task, in the direction of the gains on the trained tasks,
is the transfer signal (Figure 3 of the paper).

    python -m musiclistenbench.transfer.compare_transfer --held-out rhythm \
        --trained-results results/transfer/results_qwen2_5_omni_held_out_Q4_rhythm.jsonl
"""

import argparse
import json
import os
import re

from musiclistenbench import paths
from musiclistenbench.transfer.build_task_transfer_probe import TASKS, resolve_task


def load_task_stats(path):
    stats = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            s = stats.setdefault(r["task"], {"n": 0, "correct": 0, "logprob_correct": 0})
            s["n"] += 1
            s["correct"] += int(r["correct"])
            s["logprob_correct"] += int(r.get("logprob_correct", False))
    return stats


def default_baseline_path(trained_results_path):
    name = os.path.basename(trained_results_path)
    match = re.match(r"results_(.+)_held_out_Q\d_\w+\.jsonl$", name)
    if not match:
        raise SystemExit(
            f"can't infer --baseline-results from {name!r} (expected "
            f"results_<model>_held_out_<task>.jsonl); pass --baseline-results explicitly")
    return os.path.join(paths.RESULTS_DIR, "per_item", f"{match.group(1)}.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held-out", required=True)
    parser.add_argument("--trained-results", required=True,
                        help="results_<model>_held_out_<task>.jsonl scoring the trained checkpoint")
    parser.add_argument("--baseline-results", default=None,
                        help="untrained model's clean-test results; default: inferred from "
                             "--trained-results' file name -> results/per_item/<model>.jsonl")
    args = parser.parse_args()

    held_out = resolve_task(args.held_out)
    baseline_path = args.baseline_results or default_baseline_path(args.trained_results)

    base_stats = load_task_stats(baseline_path)
    trained_stats = load_task_stats(args.trained_results)

    tasks_to_report = [t for t in TASKS if t in trained_stats]
    if not tasks_to_report:
        raise SystemExit(f"--trained-results {args.trained_results!r} has none of the known tasks {TASKS}")
    if held_out not in tasks_to_report:
        raise SystemExit(f"--held-out {held_out!r} doesn't appear in --trained-results ({tasks_to_report} do)")
    missing_in_baseline = [t for t in tasks_to_report if t not in base_stats]
    if missing_in_baseline:
        raise SystemExit(f"--baseline-results is missing task(s) {missing_in_baseline}")
    mismatched_n = {t: (base_stats[t]["n"], trained_stats[t]["n"]) for t in tasks_to_report
                    if base_stats[t]["n"] != trained_stats[t]["n"]}
    if mismatched_n:
        raise SystemExit(f"item-count mismatch between baseline and trained results (base, trained): "
                         f"{mismatched_n}")

    print(f"baseline:  {baseline_path}")
    print(f"trained:   {args.trained_results}  (held out: {held_out})\n")
    print(f"{'task':<12} {'n':>5} {'base_acc':>9} {'trained_acc':>12} {'delta':>8}")
    for task in tasks_to_report:
        b, t = base_stats[task], trained_stats[task]
        b_acc, t_acc = b["correct"] / b["n"], t["correct"] / t["n"]
        flag = "  <- HELD OUT (transfer test)" if task == held_out else ""
        print(f"{task:<12} {b['n']:>5} {b_acc:>9.3f} {t_acc:>12.3f} {t_acc - b_acc:>+8.3f}{flag}")


if __name__ == "__main__":
    main()
