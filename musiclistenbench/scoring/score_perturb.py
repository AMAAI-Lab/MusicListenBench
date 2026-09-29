"""Scores a `run_probe.py` results file for the controlled flip/stay probe
(`build_perturb_probe.py`), adding the two requested rates on top of the usual
per-task accuracy table.

Each item edits the second sound of a real test pair either as a FLIP (content
edit -> the answer should change) or a STAY (nuisance -> the answer should hold).
The condition and the specific factor are encoded in the voice filename, e.g.
  Q1_melody/Q1_melody__test__100__000002__flip__note_identity.wav
  Q1_melody/Q1_melody__test__200__000010__stay__hiss.wav

Two rates, each in [0, 1], per task and overall:

  false_flip_rate = error rate on STAY items
      -- how often the model changed its answer when the edit should NOT change it
         ("flipped when it shouldn't"). = fraction of stay items answered wrong.

  miss_rate       = error rate on FLIP items
      -- how often the model kept its answer when the edit DOES change it
         ("didn't flip when it should"). = fraction of flip items answered wrong.

Both are reported for the two result fields the runners write: `correct` (`_gen`) and
`logprob_correct` (`_lp`). Files written by `run_probe_mm` (every result of the paper)
hold the same forced-choice reading in both, so the two columns agree. A per-factor
breakdown shows which specific edit drives the errors (which sound change causes false
flips, which content edit is missed). For the tables of the paper use `paper_tables.py`.

Run: python -m musiclistenbench.scoring.score_perturb results_qwen2_audio_perturb.jsonl
"""

import argparse
import json
import os
from collections import defaultdict

from musiclistenbench import paths


def load_results(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_voice(voice):
    """(condition, factor) from '.../<item_id>__<flip|stay>__<factor>.wav'."""
    base = str(voice).split("/")[-1].replace(".wav", "")
    parts = base.split("__")
    if len(parts) >= 2 and parts[-2] in ("flip", "stay"):
        return parts[-2], parts[-1]
    return None, None


def rate(num, den):
    return (num / den) if den else float("nan")


def _fmt(x, w=9, p=3):
    return f"{x:>{w}.{p}f}" if x == x else f"{'n/a':>{w}}"   # NaN -> n/a


def score(rows):
    tasks = defaultdict(lambda: {"n": 0, "gen_correct": 0, "lp_correct": 0, "unparsed": 0,
                                 "n_flip": 0, "n_stay": 0,
                                 "flip_gen_err": 0, "flip_lp_err": 0,
                                 "stay_gen_err": 0, "stay_lp_err": 0})
    factors = defaultdict(lambda: {"n": 0, "gen_err": 0, "lp_err": 0, "condition": None})
    for r in rows:
        cond, factor = parse_voice(r.get("voice", ""))
        gen_wrong = int(not r.get("correct", False))
        lp_wrong = int(not r.get("logprob_correct", False))
        s = tasks[r["task"]]
        s["n"] += 1
        s["gen_correct"] += int(r.get("correct", False))
        s["lp_correct"] += int(r.get("logprob_correct", False))
        s["unparsed"] += int(r.get("predicted") is None)
        if cond == "flip":
            s["n_flip"] += 1; s["flip_gen_err"] += gen_wrong; s["flip_lp_err"] += lp_wrong
        elif cond == "stay":
            s["n_stay"] += 1; s["stay_gen_err"] += gen_wrong; s["stay_lp_err"] += lp_wrong
        if factor:
            fk = factors[(r["task"], factor)]
            fk["condition"] = cond
            fk["n"] += 1; fk["gen_err"] += gen_wrong; fk["lp_err"] += lp_wrong
    return tasks, factors


def print_task_table(tasks):
    order = ["Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"]
    keys = [k for k in order if k in tasks] + [k for k in tasks if k not in order]
    tot = defaultdict(int)
    print(f"{'task':<12}{'n':>5}{'acc':>8}{'unparsed':>9}{'flip':>6}{'stay':>6}"
          f"{'miss_gen':>9}{'ff_gen':>8}{'miss_lp':>9}{'ff_lp':>8}")
    for task in keys:
        s = tasks[task]
        for k, v in s.items():
            if isinstance(v, int):
                tot[k] += v
        print(f"{task:<12}{s['n']:>5}{_fmt(rate(s['gen_correct'], s['n']), 8)}{s['unparsed']:>9}"
              f"{s['n_flip']:>6}{s['n_stay']:>6}"
              f"{_fmt(rate(s['flip_gen_err'], s['n_flip']))}{_fmt(rate(s['stay_gen_err'], s['n_stay']), 8)}"
              f"{_fmt(rate(s['flip_lp_err'], s['n_flip']))}{_fmt(rate(s['stay_lp_err'], s['n_stay']), 8)}")
    print(f"{'overall':<12}{tot['n']:>5}{_fmt(rate(tot['gen_correct'], tot['n']), 8)}{tot['unparsed']:>9}"
          f"{tot['n_flip']:>6}{tot['n_stay']:>6}"
          f"{_fmt(rate(tot['flip_gen_err'], tot['n_flip']))}{_fmt(rate(tot['stay_gen_err'], tot['n_stay']), 8)}"
          f"{_fmt(rate(tot['flip_lp_err'], tot['n_flip']))}{_fmt(rate(tot['stay_lp_err'], tot['n_stay']), 8)}")
    print("\nmiss = miss_rate = error rate on FLIP items (didn't flip when it should)")
    print("ff   = false_flip_rate = error rate on STAY items (flipped when it shouldn't)")
    print("_gen = generated-letter channel, _lp = forced-choice logprob channel")
    return tot


def print_factor_table(factors):
    print("\nper-factor (flip factors = content edits; stay factors = nuisances):")
    print(f"{'task':<12}{'factor':<16}{'kind':<6}{'n':>5}{'err_gen':>9}{'err_lp':>9}")
    def sort_key(k):
        (task, factor) = k
        return (["Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"].index(task)
                if task in ["Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"] else 9,
                factors[k]["condition"] or "", factor)
    for k in sorted(factors, key=sort_key):
        f = factors[k]
        print(f"{k[0]:<12}{k[1]:<16}{(f['condition'] or '?'):<6}{f['n']:>5}"
              f"{_fmt(rate(f['gen_err'], f['n']))}{_fmt(rate(f['lp_err'], f['n']))}")


def summarize(tasks, factors):
    order = ["Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"]
    tot = defaultdict(int)
    per_task = {}
    for task, s in tasks.items():
        for k, v in s.items():
            if isinstance(v, int):
                tot[k] += v
        per_task[task] = {
            "n": s["n"], "acc": rate(s["gen_correct"], s["n"]),
            "n_flip": s["n_flip"], "n_stay": s["n_stay"],
            "miss_rate_gen": rate(s["flip_gen_err"], s["n_flip"]),
            "false_flip_rate_gen": rate(s["stay_gen_err"], s["n_stay"]),
            "miss_rate_logprob": rate(s["flip_lp_err"], s["n_flip"]),
            "false_flip_rate_logprob": rate(s["stay_lp_err"], s["n_stay"]),
        }
    out = {"per_task": {k: per_task[k] for k in
                        ([t for t in order if t in per_task] +
                         [t for t in per_task if t not in order])},
           "overall": {
               "n": tot["n"], "acc": rate(tot["gen_correct"], tot["n"]),
               "n_flip": tot["n_flip"], "n_stay": tot["n_stay"],
               "miss_rate_gen": rate(tot["flip_gen_err"], tot["n_flip"]),
               "false_flip_rate_gen": rate(tot["stay_gen_err"], tot["n_stay"]),
               "miss_rate_logprob": rate(tot["flip_lp_err"], tot["n_flip"]),
               "false_flip_rate_logprob": rate(tot["stay_lp_err"], tot["n_stay"]),
           },
           "per_factor": {f"{t}/{f}": {"condition": factors[(t, f)]["condition"],
                                       "n": factors[(t, f)]["n"],
                                       "error_rate_gen": rate(factors[(t, f)]["gen_err"], factors[(t, f)]["n"]),
                                       "error_rate_logprob": rate(factors[(t, f)]["lp_err"], factors[(t, f)]["n"])}
                          for (t, f) in factors}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?",
                        default=os.path.join(paths.RESULTS_DIR, "results_qwen2_audio_perturb.jsonl"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load_results(args.results)
    if not rows:
        raise SystemExit(f"no rows in {args.results}")
    if not any(parse_voice(r.get("voice", ""))[0] for r in rows):
        print("WARNING: results don't look like a flip/stay perturb set "
              "(no __flip__/__stay__ in voice paths); rates will be n/a.\n")

    tasks, factors = score(rows)
    print(f"scored {len(rows)} items from {args.results}\n")
    print_task_table(tasks)
    print_factor_table(factors)

    out_path = args.out or (os.path.splitext(args.results)[0] + "_perturb_summary.json")
    with open(out_path, "w") as f:
        json.dump(summarize(tasks, factors), f, indent=2)
    print(f"\nsummary JSON -> {out_path}")


if __name__ == "__main__":
    main()
