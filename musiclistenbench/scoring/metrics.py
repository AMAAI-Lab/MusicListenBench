"""Metrics of MusicListenBench (Sections 3.3 and 3.5 of the paper).

One prediction is one answer letter, 'A' or 'B', for one test item. Test items
come in three kinds, identified by the name of their audio file:

    <item>__base.wav               clean item (1,000)
    <item>__flip__<factor>.wav     FLIP variant: the queried attribute changes (1,000)
    <item>__stay__<factor>.wav     STAY variant: only the sound changes (1,000)

Definitions
-----------
clean accuracy   share of clean items answered correctly.
miss rate (MR)   share of FLIP items answered wrongly (the model keeps the old
                 answer although the music changed).
false-flip rate  share of STAY items answered wrongly (the model changes its
(FFR)            answer although only the sound changed).
Acc_FS           1 - (MR + FFR) / 2, the FLIP/STAY accuracy. A model that ignores
                 the audio scores 50% on average, whatever letter it prefers.

Missing or invalid answers count as wrong.
"""

import json
import math
import os
from collections import defaultdict

from musiclistenbench import paths

TASKS = ["Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"]
TASK_SHORT = {"Q1_melody": "Mel.", "Q2_harmony": "Har.", "Q3_timbre": "Tim.", "Q4_rhythm": "Rhy."}
LETTERS = ("A", "B")


def item_id_from_voice(voice):
    """'Q1_melody/Q1_melody__test__100__000000__base.wav' -> 'Q1_melody__test__100__000000__base'."""
    if isinstance(voice, list):
        voice = voice[0]
    return os.path.basename(voice)[: -len(".wav")]


def parse_item_id(item_id):
    """(base item, kind, factor) with kind in {'clean', 'flip', 'stay'}."""
    parts = item_id.split("__")
    if parts[-1] == "base":
        return "__".join(parts[:-1]), "clean", None
    if len(parts) >= 2 and parts[-2] in ("flip", "stay"):
        return "__".join(parts[:-2]), parts[-2], parts[-1]
    raise ValueError(f"not a MusicListenBench item id: {item_id}")


def load_reference(eval_json=None, eval_perturb_json=None):
    """item_id -> {'task', 'kind', 'factor', 'gold'} for all 3,000 test items."""
    ref = {}
    for path in (eval_json or paths.EVAL_JSON, eval_perturb_json or paths.EVAL_PERTURB_JSON):
        with open(path) as f:
            entries = json.load(f)
        for e in entries:
            voice = e["voice"][0] if isinstance(e["voice"], list) else e["voice"]
            item_id = item_id_from_voice(voice)
            _, kind, factor = parse_item_id(item_id)
            ref[item_id] = {
                "task": voice.split("/")[0],
                "kind": kind,
                "factor": factor,
                "gold": e["conversations"][1]["value"].strip().upper(),
            }
    return ref


def rate(num, den):
    return num / den if den else float("nan")


def chance_interval(n, z=1.96):
    """Range (in %) of scores that are not significantly different from 50%
    (two-sided, p > 0.05, normal approximation) for n items."""
    half = 100 * z * math.sqrt(0.25 / n)
    return 50 - half, 50 + half


def score(predictions, reference):
    """predictions: item_id -> 'A' | 'B' (anything else, or absent, counts as wrong).
    Returns a nested dict of every number the paper reports for one model."""
    def correct(item_id):
        return predictions.get(item_id) == reference[item_id]["gold"]

    def is_a(item_id):
        return predictions.get(item_id) == "A"

    cnt = defaultdict(int)          # flat counters keyed by tuples
    for item_id, r in reference.items():
        task, kind, factor = r["task"], r["kind"], r["factor"]
        ok = correct(item_id)
        for t in (task, "All"):
            cnt[("n", kind, t)] += 1
            cnt[("ok", kind, t)] += ok
            cnt[("A", kind, t)] += is_a(item_id)
            cnt[("goldA", kind, t)] += r["gold"] == "A"
            cnt[("okA", kind, t)] += ok and r["gold"] == "A"
            cnt[("okB", kind, t)] += ok and r["gold"] == "B"
        if kind in ("flip", "stay"):
            cnt[("fn", kind, task, factor)] += 1
            cnt[("fok", kind, task, factor)] += ok
            cnt[("fn", kind, "All", factor)] += 1
            cnt[("fok", kind, "All", factor)] += ok

    n_answered = sum(1 for i in reference if predictions.get(i) in LETTERS)
    out = {"n_items": len(reference), "n_valid_answers": n_answered,
           "n_missing_or_invalid": len(reference) - n_answered}

    clean = {}
    flipstay = {}
    for t in TASKS + ["All"]:
        n = cnt[("n", "clean", t)]
        nA = cnt[("goldA", "clean", t)]
        nB = n - nA
        acc_a = rate(cnt[("okA", "clean", t)], nA)
        acc_b = rate(cnt[("okB", "clean", t)], nB)
        clean[t] = {
            "n": n,
            "accuracy": rate(cnt[("ok", "clean", t)], n),
            "a_rate": rate(cnt[("A", "clean", t)], n),
            "acc_given_A": acc_a, "acc_given_B": acc_b,
            "balanced_accuracy": (acc_a + acc_b) / 2, "letter_gap": abs(acc_a - acc_b),
        }
        nf, ns = cnt[("n", "flip", t)], cnt[("n", "stay", t)]
        mr = 1 - rate(cnt[("ok", "flip", t)], nf)
        ffr = 1 - rate(cnt[("ok", "stay", t)], ns)
        flipstay[t] = {
            "n_flip": nf, "n_stay": ns,
            "miss_rate": mr, "false_flip_rate": ffr, "acc_fs": 1 - (mr + ffr) / 2,
            "a_rate": rate(cnt[("A", "flip", t)] + cnt[("A", "stay", t)], nf + ns),
        }
    out["clean"] = clean
    out["flip_stay"] = flipstay

    # per-factor error rates: false flips per sound change, misses per content edit
    per_factor = {}
    for key, n in list(cnt.items()):
        if key[0] != "fn":
            continue
        _, kind, t, factor = key
        per_factor[f"{t}/{kind}/{factor}"] = {
            "n": n, "error_rate": 1 - rate(cnt[("fok", kind, t, factor)], n)}
    out["per_factor_error"] = per_factor
    return out


def always_answer(letter, reference):
    """The 'always A' (or 'always B') baseline predictions."""
    return {i: letter for i in reference}


# ------------------------------------------------------------------ statistics


def two_rate_drop_test(mr0, mr1, ffr0, ffr1, n=1000):
    """Did the miss rate fall more than the false-flip rate? (Table 16.)
    z-test on (drop in MR) - (drop in FFR), treating the four rates as independent
    binomial samples of n items each (this overestimates the variance when the
    before/after answers are positively correlated). Returns (z, two-sided p)."""
    diff = (mr0 - mr1) - (ffr0 - ffr1)
    var = sum(p * (1 - p) / n for p in (mr0, mr1, ffr0, ffr1))
    if var == 0:
        return float("nan"), float("nan")
    z = diff / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p
