#!/usr/bin/env python3
"""Validate leaderboard.csv. Run locally or in CI: python scripts/validate.py leaderboard.csv"""
import csv, sys, re
from datetime import date

COLUMNS = ["type","model","organization","model_url","access","params_b","method",
           "melody","rhythm","timbre","harmony","overall","acc_fs","miss_rate","false_flip_rate","a_rate",
           "readout","verified","submitted_by","date","source_url","benchmark_version","notes"]
ASPECTS = ["melody","rhythm","timbre","harmony"]
ENUMS = {"type": {"model","baseline"}, "verified": {"yes","no"}}
MODEL_ACCESS = {"open","closed"}
MODEL_READOUT = {"logprob","generated"}
KNOWN_VERSIONS = {"1.0"}
# The test split has 250 clean items per task, so "overall" (clean accuracy on all 1,000 items)
# equals the mean of the four task scores.
# acc_fs = 100 - (miss_rate + false_flip_rate) / 2 (FLIP/STAY accuracy, paper Section 3.3).
# Scores are rounded to one decimal, so allow a small tolerance.
TOLERANCE = 0.1

errors = []
def err(line, msg):
    errors.append(f"::error file=leaderboard.csv,line={line}::{msg}")

def number(row, col, line, required=True):
    """Return the value of a percent column, or None (and report) if it is missing or out of range."""
    if row[col] == "" and not required:
        return None
    try:
        v = float(row[col])
        if not 0 <= v <= 100:
            raise ValueError
        return v
    except ValueError:
        err(line, f"{col} must be a number between 0 and 100 (percent)")
        return None

def main(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            err(1, f"Header must be exactly: {','.join(COLUMNS)}")
            return
        seen = set()
        for i, row in enumerate(reader, start=2):
            for col, allowed in ENUMS.items():
                if row[col] not in allowed:
                    err(i, f"{col} must be one of {sorted(allowed)}, got '{row[col]}'")
            if not row["model"].strip():
                err(i, "model is required")
            scores = {a: number(row, a, i) for a in ASPECTS + ["overall", "acc_fs", "miss_rate", "false_flip_rate"]}
            number(row, "a_rate", i, required=False)   # share of 'A' answers on the clean items; empty if unknown
            if None not in (scores[a] for a in ASPECTS + ["overall"]):
                mean = sum(scores[a] for a in ASPECTS) / 4
                if abs(mean - scores["overall"]) > TOLERANCE:
                    err(i, f"overall ({scores['overall']}) should equal the mean of the four tasks ({mean:.2f})")
            if None not in (scores["acc_fs"], scores["miss_rate"], scores["false_flip_rate"]):
                fs = 100 - (scores["miss_rate"] + scores["false_flip_rate"]) / 2
                if abs(fs - scores["acc_fs"]) > TOLERANCE:
                    err(i, f"acc_fs ({scores['acc_fs']}) should equal 100 - (miss_rate + false_flip_rate) / 2 = {fs:.2f}")
            if row["type"] == "model":
                if row["access"] not in MODEL_ACCESS:
                    err(i, "access must be 'open' or 'closed'")
                if row["readout"] not in MODEL_READOUT:
                    err(i, "readout must be 'logprob' (letter probabilities) or 'generated' (generated text)")
                if not row["method"].strip():
                    err(i, "method is required: 'zero-shot', or how the released training split was used (e.g. GRPO (LoRA))")
                if row["params_b"] and not re.fullmatch(r"\d+(\.\d+)?", row["params_b"]):
                    err(i, "params_b must be a number in billions, or empty if unknown")
                if not row["source_url"] and row["verified"] == "no":
                    err(i, "self-reported results need a source_url (paper, code, or eval logs)")
            try:
                date.fromisoformat(row["date"])
            except ValueError:
                err(i, "date must be YYYY-MM-DD")
            if row["benchmark_version"] not in KNOWN_VERSIONS:
                err(i, f"benchmark_version must be one of {sorted(KNOWN_VERSIONS)}")
            key = (row["model"], row["method"], row["benchmark_version"])
            if key in seen:
                err(i, f"duplicate entry for {key}")
            seen.add(key)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "leaderboard.csv")
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("leaderboard.csv is valid")
