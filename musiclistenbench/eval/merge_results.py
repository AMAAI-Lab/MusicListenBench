"""Merges per-shard `run_probe_mm.py --num-shards N --shard i` output files into
one combined jsonl plus a global per-task/overall summary table.

Run: python -m musiclistenbench.eval.merge_results results/results_qwen2_audio.shard*.jsonl
"""

import argparse
import glob
import json
import os

from musiclistenbench import paths as mlb_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", help="shard jsonl files or globs")
    parser.add_argument("--out", default=os.path.join(mlb_paths.RESULTS_DIR, "results_qwen2_audio.jsonl"))
    args = parser.parse_args()

    paths = sorted({p for pattern in args.shards for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"no files matched: {args.shards}")

    stats = {}
    n_unparsed = 0
    total = 0
    with open(args.out, "w") as out_f:
        for path in paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    out_f.write(line + "\n")
                    task_stats = stats.setdefault(r["task"], {
                        "n": 0, "correct": 0, "unparsed": 0, "logprob_correct": 0,
                    })
                    task_stats["n"] += 1
                    task_stats["correct"] += int(r["correct"])
                    task_stats["unparsed"] += int(r["predicted"] is None)
                    task_stats["logprob_correct"] += int(r.get("logprob_correct", False))
                    n_unparsed += int(r["predicted"] is None)
                    total += 1

    print(f"merged {total} items from {len(paths)} shard files -> {args.out}\n")
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


if __name__ == "__main__":
    main()
