"""Scores a leaderboard submission (Appendix E of the paper).

A submission is one JSON Lines file. The first line holds the metadata, every
following line is one answer:

    {"meta": {"model": "Qwen2.5-Omni-7B", "version": "ae9e169",
              "readout": "logprob",              # "logprob" (letter probabilities) or "generated" (text)
              "used_training_split": "none",     # "none", or how the released train split was used
              "date": "2026-09-28",
              "code_url": "https://..."}}        # optional
    {"item_id": "Q1_melody__test__100__000000__base", "answer": "A"}
    ...

`item_id` is the name of the audio file without ".wav" (see data/eval.json and
data/eval_perturb.json); the file must have one record per test item, 3,000 in
all (1,000 clean, 1,000 flip, 1,000 stay). Missing or invalid answers count as
wrong. Training on test items is not allowed.

Usage:
    python -m musiclistenbench.scoring.score_submission my_submission.jsonl
    python -m musiclistenbench.scoring.score_submission my_submission.jsonl --out score.json
"""

import argparse
import json
import sys

from musiclistenbench.scoring import metrics

REQUIRED_META = ("model", "version", "readout", "used_training_split", "date")
READOUTS = ("logprob", "generated")


def read_submission(path):
    """Returns (meta, answers, problems). `problems` lists everything that will
    cost the submission points or that the maintainers must look at."""
    meta, answers, problems = {}, {}, []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"line {line_no}: not valid JSON")
                continue
            if "meta" in rec:
                meta = rec["meta"]
                continue
            item_id, answer = rec.get("item_id"), rec.get("answer")
            if item_id is None:
                problems.append(f"line {line_no}: no item_id")
                continue
            if item_id in answers:
                problems.append(f"line {line_no}: duplicate item_id {item_id} (the first answer is kept)")
                continue
            answers[item_id] = str(answer).strip().upper() if answer is not None else None
    return meta, answers, problems


def validate(meta, answers, reference):
    problems = []
    for key in REQUIRED_META:
        if key not in meta:
            problems.append(f"meta: missing '{key}'")
    if meta.get("readout") not in READOUTS:
        problems.append(f"meta.readout must be one of {READOUTS}")
    unknown = [i for i in answers if i not in reference]
    if unknown:
        problems.append(f"{len(unknown)} unknown item ids (first: {unknown[0]})")
    bad = [i for i, a in answers.items() if a not in metrics.LETTERS]
    if bad:
        problems.append(f"{len(bad)} answers are not 'A' or 'B' (first: {bad[0]}); they count as wrong")
    missing = [i for i in reference if i not in answers]
    if missing:
        problems.append(f"{len(missing)} test items have no answer; they count as wrong")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission")
    ap.add_argument("--out", default=None, help="write the full score as JSON")
    ap.add_argument("--eval-json", default=None)
    ap.add_argument("--eval-perturb-json", default=None)
    args = ap.parse_args()

    reference = metrics.load_reference(args.eval_json, args.eval_perturb_json)
    meta, answers, problems = read_submission(args.submission)
    problems += validate(meta, answers, reference)
    result = metrics.score(answers, reference)
    result["meta"] = meta
    result["problems"] = problems

    c, fs = result["clean"], result["flip_stay"]["All"]
    print(f"model: {meta.get('model', '?')} ({meta.get('version', '?')}), readout: {meta.get('readout', '?')}, "
          f"training split: {meta.get('used_training_split', '?')}")
    print(f"answers: {result['n_valid_answers']} valid of {result['n_items']} test items")
    print()
    print(f"{'task':<10}{'clean acc':>10}{'A rate':>9}{'Acc_FS':>9}{'miss':>8}{'false flip':>12}")
    for t in metrics.TASKS + ["All"]:
        f = result["flip_stay"][t]
        print(f"{metrics.TASK_SHORT.get(t, t):<10}{100 * c[t]['accuracy']:>10.1f}{100 * c[t]['a_rate']:>9.1f}"
              f"{100 * f['acc_fs']:>9.1f}{100 * f['miss_rate']:>8.1f}{100 * f['false_flip_rate']:>12.1f}")
    lo, hi = metrics.chance_interval(2000)
    print(f"\nFLIP/STAY accuracy {100 * fs['acc_fs']:.1f}% (chance range on 2,000 items: {lo:.1f}% to {hi:.1f}%)")
    if problems:
        print("\nproblems:")
        for p in problems:
            print("  -", p)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nfull score -> {args.out}")
    return 1 if any(p.startswith(("meta", "line")) for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
