"""Turns a per-item result file written by `musiclistenbench.eval.run_probe_mm`
(clean and flip/stay files) into one leaderboard submission (see score_submission.py).

    python -m musiclistenbench.scoring.convert_results \
        --clean results/open_models/qwen2_5_omni.jsonl \
        --flip-stay results/open_models/qwen2_5_omni_perturb.jsonl \
        --model Qwen2.5-Omni-7B --version ae9e169 --readout logprob \
        --used-training-split none --date 2026-09-28 --out submission.jsonl
"""

import argparse
import json

from musiclistenbench.scoring.metrics import item_id_from_voice


def read_answers(path):
    answers = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                # `predicted` is the answer letter each runner records (for the local
                # models it is the higher of log P('A') and log P('B'); for API models
                # the generated letter).
                answers[item_id_from_voice(r["voice"])] = r.get("predicted")
    return answers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--flip-stay", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--readout", required=True, choices=["logprob", "generated"])
    ap.add_argument("--used-training-split", default="none")
    ap.add_argument("--date", required=True)
    ap.add_argument("--code-url", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    answers = {**read_answers(args.clean), **read_answers(args.flip_stay)}
    meta = {"model": args.model, "version": args.version, "readout": args.readout,
            "used_training_split": args.used_training_split, "date": args.date}
    if args.code_url:
        meta["code_url"] = args.code_url
    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for item_id in sorted(answers):
            f.write(json.dumps({"item_id": item_id, "answer": answers[item_id]}) + "\n")
    print(f"wrote {len(answers)} answers -> {args.out}")


if __name__ == "__main__":
    main()
