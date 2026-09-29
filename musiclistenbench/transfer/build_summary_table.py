"""Consolidates the leave-one-task-out results (one line per model and held-out
task) into results/transfer/summary_table.csv, the numbers behind Figure 3.

    python -m musiclistenbench.transfer.build_summary_table
"""

import csv
import os

from musiclistenbench import paths
from musiclistenbench.transfer.build_task_transfer_probe import TASKS
from musiclistenbench.transfer.compare_transfer import default_baseline_path, load_task_stats

MODELS = ["qwen2_audio", "qwen2_5_omni"]
TRANSFER_DIR = os.path.join(paths.RESULTS_DIR, "transfer")


def main():
    rows = []
    for model_safe in MODELS:
        for held_out in TASKS:
            trained_path = os.path.join(TRANSFER_DIR, f"results_{model_safe}_held_out_{held_out}.jsonl")
            if not os.path.exists(trained_path):
                print(f"skip (not evaluated yet): {trained_path}")
                continue
            base_stats = load_task_stats(default_baseline_path(trained_path))[held_out]
            trained_stats = load_task_stats(trained_path)[held_out]
            assert base_stats["n"] == trained_stats["n"], (
                f"item-count mismatch for {model_safe}/{held_out}: {base_stats['n']} vs {trained_stats['n']}")
            b_acc, t_acc = base_stats["correct"] / base_stats["n"], trained_stats["correct"] / trained_stats["n"]
            rows.append({
                "model": model_safe, "held_out_task": held_out, "n": base_stats["n"],
                "base_acc": round(b_acc, 4), "trained_acc": round(t_acc, 4), "delta": round(t_acc - b_acc, 4),
            })

    if not rows:
        raise SystemExit("no results files found yet -- nothing to summarize")

    out_path = os.path.join(TRANSFER_DIR, "summary_table.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows -> {out_path}\n")
    print(f"{'model':<14} {'held_out':<10} {'n':>5} {'base_acc':>9} {'trained_acc':>12} {'delta':>8}")
    for r in rows:
        print(f"{r['model']:<14} {r['held_out_task']:<10} {r['n']:>5} {r['base_acc']:>9.3f} "
              f"{r['trained_acc']:>12.3f} {r['delta']:>+8.3f}")


if __name__ == "__main__":
    main()
