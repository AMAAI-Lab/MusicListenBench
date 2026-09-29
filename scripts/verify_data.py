"""Checks the released item files (and, if present, the audio) against the numbers
stated in the paper. Exits with a non-zero status if any check fails.

    python scripts/verify_data.py                  # item files only
    python scripts/verify_data.py --audio          # also open every WAV (needs data/audio*/)
    python scripts/verify_data.py --audio --test-only   # only the 3,000 test files

Checks (paper section in brackets):
  * 10,000 training items, 2,500 per task; 49.6% of training answers are 'A';
    the letter meaning is swapped in 51.4% of the same/different training items [3.4, A]
  * 1,000 clean test items, 250 per task, 528 of them with answer 'A' [3.4, A]
  * 2,000 flip/stay items: 250 flip and 250 stay per task, stay changes as in Table 6 [A]
  * 34 question wordings (8 + 8 + 8 + 10) [A, Table 7]
  * flip changes the gold letter on all melody/harmony/timbre items and on 130 of the
    250 rhythm items; flip and stay share the letter on 131 rhythm items [3.3, A]
  * train and test items come from disjoint generated items [3.4]
  * every clip of a task has the same number of samples; 44.1 kHz, mono, 16-bit [A]
"""

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musiclistenbench import paths  # noqa: E402
from musiclistenbench.scoring import metrics as M  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def wording(prompt):
    return re.sub(r" Only answer '.'.*$", "", prompt.replace(" <audio>", "")).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", action="store_true", help="also check every WAV file")
    ap.add_argument("--test-only", action="store_true", help="with --audio: skip the training files")
    args = ap.parse_args()

    train = json.load(open(paths.TRAIN_JSON))
    clean = json.load(open(paths.EVAL_JSON))
    fs = json.load(open(paths.EVAL_PERTURB_JSON))
    ref = M.load_reference()

    task_of = lambda e: e["voice"][0].split("/")[0]
    gold_of = lambda e: e["conversations"][1]["value"]

    # ---- training split
    n_task = collections.Counter(task_of(e) for e in train)
    check("train: 10,000 items", len(train) == 10000, str(len(train)))
    check("train: 2,500 per task", all(n_task[t] == 2500 for t in M.TASKS), str(dict(n_task)))
    a_share = sum(gold_of(e) == "A" for e in train) / len(train)
    check("train: 49.6% of answers are 'A'", abs(a_share - 0.496) < 0.0006, f"{100 * a_share:.2f}%")
    sd = [e for e in train if task_of(e) != "Q4_rhythm"]
    swapped = sum("Only answer 'B' for the same" in e["conversations"][0]["value"] for e in sd) / len(sd)
    check("train: letter meaning swapped in 51.4% of same/different items", abs(swapped - 0.514) < 0.0006,
          f"{100 * swapped:.2f}%")

    # ---- clean test
    n_task = collections.Counter(task_of(e) for e in clean)
    check("clean test: 1,000 items, 250 per task", len(clean) == 1000 and all(n_task[t] == 250 for t in M.TASKS))
    n_a = sum(gold_of(e) == "A" for e in clean)
    check("clean test: 528 answers are 'A'", n_a == 528, str(n_a))

    # ---- flip/stay test
    comp = collections.Counter((r["task"], r["kind"], r["factor"]) for r in ref.values() if r["kind"] != "clean")
    check("flip/stay: 2,000 items", len(fs) == 2000)
    check("flip/stay: 250 flip and 250 stay per task",
          all(sum(v for (t, k, _), v in comp.items() if t == task and k == kind) == 250
              for task in M.TASKS for kind in ("flip", "stay")))
    expected = {"eq": (63, 63, 84, 84), "hiss": (63, 63, 83, 83), "room": (62, 62, 83, 83),
                "transposition": (62, 62, 0, 0)}
    ok = all(comp.get((task, "stay", f), 0) == expected[f][i] for f in expected for i, task in enumerate(M.TASKS))
    check("flip/stay: stay changes per task as in Table 6", ok)
    totals = {f: sum(v for (t, k, ff), v in comp.items() if k == "stay" and ff == f) for f in expected}
    check("flip/stay: 294 EQ, 292 hiss, 290 room, 124 transposition",
          totals == {"eq": 294, "hiss": 292, "room": 290, "transposition": 124}, str(totals))

    # ---- wordings
    ws = collections.defaultdict(set)
    for e in train + clean + fs:
        ws[task_of(e)].add(wording(e["conversations"][0]["value"]))
    counts = {t: len(ws[t]) for t in M.TASKS}
    check("34 question wordings (8 + 8 + 8 + 10)", counts == {"Q1_melody": 8, "Q2_harmony": 8, "Q3_timbre": 8,
                                                              "Q4_rhythm": 10}, str(counts))

    # ---- letters of flip/stay against the clean item
    by_item = collections.defaultdict(dict)
    for item_id, r in ref.items():
        base, kind, _ = M.parse_item_id(item_id)
        by_item[base][kind] = r
    for t in ("Q1_melody", "Q2_harmony", "Q3_timbre"):
        items = [v for b, v in by_item.items() if v["clean"]["task"] == t]
        check(f"{t}: flip changes the letter, stay keeps it (250 items)",
              all(v["flip"]["gold"] != v["clean"]["gold"] and v["stay"]["gold"] == v["clean"]["gold"] for v in items)
              and len(items) == 250)
    rh = [v for v in by_item.values() if v["clean"]["task"] == "Q4_rhythm"]
    check("Q4_rhythm: flip changes the letter on 130 of 250 items",
          sum(v["flip"]["gold"] != v["clean"]["gold"] for v in rh) == 130)
    check("Q4_rhythm: flip and stay share the letter on 131 of 250 items",
          sum(v["flip"]["gold"] == v["stay"]["gold"] for v in rh) == 131)

    # ---- disjoint train/test items
    train_items = {os.path.basename(e["voice"][0]).split("__base")[0] for e in train}
    test_items = {b for b in by_item}
    check("train and test items are disjoint", not (train_items & test_items))

    # ---- audio
    if args.audio:
        import soundfile as sf
        lengths = collections.defaultdict(set)
        missing, bad_format = 0, 0
        sets = ((clean, paths.AUDIO_DIR), (fs, paths.AUDIO_PERTURB_DIR)) if args.test_only else (
            (train + clean, paths.AUDIO_DIR), (fs, paths.AUDIO_PERTURB_DIR))
        for jf, root in sets:
            for e in jf:
                path = os.path.join(root, e["voice"][0])
                if not os.path.exists(path):
                    missing += 1
                    continue
                info = sf.info(path)
                if info.samplerate != 44100 or info.channels != 1 or info.subtype != "PCM_16":
                    bad_format += 1
                if "__train__" not in path:
                    lengths[task_of(e)].add(info.frames)
        check("audio: no missing files", missing == 0, f"{missing} missing")
        check("audio: 44.1 kHz, mono, 16-bit PCM", bad_format == 0, f"{bad_format} deviating")
        check("audio: one length per task in the test files (6.0/5.0/6.0/9.0 s)",
              {t: sorted(v) for t, v in lengths.items()} ==
              {"Q1_melody": [264600], "Q2_harmony": [220500], "Q3_timbre": [264600], "Q4_rhythm": [396900]},
              str({t: sorted(v) for t, v in lengths.items()}))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
