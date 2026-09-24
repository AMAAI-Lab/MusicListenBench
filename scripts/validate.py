#!/usr/bin/env python3
"""Validate leaderboard.csv. Run locally or in CI: python scripts/validate.py leaderboard.csv"""
import csv, sys, re
from datetime import date

COLUMNS = ["type","model","organization","model_url","access","params_b","method",
           "melody","rhythm","timbre","harmony","overall","verified","submitted_by",
           "date","source_url","benchmark_version","notes"]
ASPECTS = ["melody","rhythm","timbre","harmony"]
ENUMS = {"type": {"model","baseline"}, "verified": {"yes","no"}}
MODEL_ACCESS = {"open","closed"}
KNOWN_VERSIONS = {"1.0"}
# TODO(maintainers): confirm "overall" is the macro-average of the four aspects.
# Set to False if overall is scored as its own task.
OVERALL_IS_MEAN = True
TOLERANCE = 0.1

errors = []
def err(line, msg):
    errors.append(f"::error file=leaderboard.csv,line={line}::{msg}")

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
            scores = {}
            for a in ASPECTS + ["overall"]:
                try:
                    v = float(row[a])
                    if not 0 <= v <= 100:
                        raise ValueError
                    scores[a] = v
                except ValueError:
                    err(i, f"{a} must be a number between 0 and 100 (percent accuracy)")
            if OVERALL_IS_MEAN and len(scores) == 5:
                mean = sum(scores[a] for a in ASPECTS) / 4
                if abs(mean - scores["overall"]) > TOLERANCE:
                    err(i, f"overall ({scores['overall']}) should equal the mean of the four aspects ({mean:.2f})")
            if row["type"] == "model":
                if row["access"] not in MODEL_ACCESS:
                    err(i, "access must be 'open' or 'closed'")
                if not row["method"].strip():
                    err(i, "method is required (e.g. zero-shot, SFT, GRPO)")
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
