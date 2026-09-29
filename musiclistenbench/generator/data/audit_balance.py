"""Step 3a: class-balance / decorrelation audit.

Hard rules, enforced here as hard failures (not warnings, not skippable by a
flag; they guard against confounds such as melody identity vs. pitch class):
  - base answers are class-balanced per row
  - every factor not being asked about is sampled independently of the label
    (near-zero correlation between each non-target factor and the label)

Usage:
    python -m musiclistenbench.generator.data.audit_balance --manifest audio_v2/manifest.jsonl
"""

import argparse
import collections
import sys

import numpy as np

from musiclistenbench.generator.data import manifest as manifest_mod

BALANCE_MAX_WORST_FRACTION = 0.65   # [P]
CORRELATION_MAX_ABS = 0.2           # [P]


def check_balance(base_trials):
    ok = True
    by_task = collections.defaultdict(list)
    for t in base_trials:
        by_task[t["task"]].append(t["answer"])
    for task, answers in sorted(by_task.items()):
        counts = collections.Counter(answers)
        total = sum(counts.values())
        worst = max(counts.values()) / total if total else 0
        status = "OK" if worst <= BALANCE_MAX_WORST_FRACTION else "FAIL"
        print(f"[balance:{status}] {task}: {dict(counts)} (n={total}, worst={worst:.2f})")
        if worst > BALANCE_MAX_WORST_FRACTION:
            ok = False
    return ok


def check_decorrelation(base_trials):
    ok = True
    by_task = collections.defaultdict(list)
    for t in base_trials:
        by_task[t["task"]].append(t)
    labels_seen = sorted({v for trials in by_task.values() for v in {t["answer"] for t in trials}})

    for task, trials in sorted(by_task.items()):
        labels = sorted({t["answer"] for t in trials})
        if len(labels) != 2:
            continue
        y = np.array([0 if t["answer"] == labels[0] else 1 for t in trials], dtype=np.float64)

        factor_keys = set()
        for t in trials:
            sp = t.get("synth_params") or {}
            for k, v in sp.items():
                if isinstance(v, (int, float)) and k != "home_soundfont":
                    factor_keys.add(k)

        for key in sorted(factor_keys):
            x = np.array([float((t.get("synth_params") or {}).get(key, np.nan)) for t in trials])
            valid = ~np.isnan(x)
            if valid.sum() < 10 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
                continue
            corr = float(np.corrcoef(x[valid], y[valid])[0, 1])
            status = "OK" if abs(corr) <= CORRELATION_MAX_ABS else "FAIL"
            print(f"[decorr:{status}] {task}.{key} vs answer: r={corr:.3f}")
            if abs(corr) > CORRELATION_MAX_ABS:
                ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    _, trials = manifest_mod.read_manifest(args.manifest)
    base_trials = [t for t in trials if t["expected_role"] == "base" and not t["is_catch"]]

    ok = check_balance(base_trials)
    ok &= check_decorrelation(base_trials)

    print()
    if ok:
        print("BALANCE AUDIT PASSED")
    else:
        print("BALANCE AUDIT FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
