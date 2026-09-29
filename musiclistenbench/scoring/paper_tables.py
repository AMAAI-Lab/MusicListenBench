"""Recomputes the results tables of the paper from per-item answers (written by scripts/eval_all_models.sh).

    python -m musiclistenbench.scoring.paper_tables            # every table, as Markdown
    python -m musiclistenbench.scoring.paper_tables --table 2 3
    python -m musiclistenbench.scoring.paper_tables --csv-dir results/tables

Every number comes from results/per_item/<model>.jsonl (1,000 clean items) and
results/per_item/<model>_perturb.jsonl (2,000 flip/stay items), scored with
musiclistenbench.scoring.metrics. Tables reproduced:

    2   results without training                   14  letter bias on the clean test
    3   results after GRPO                         15  per-task flip/stay accuracy, no training
    12  per-task miss / false-flip rate, untrained 16  drop in miss rate vs. false-flip rate
    13  per-task miss / false-flip rate, after GRPO 19  false-flip rate by kind of sound change
    21  errors by correct answer (Gemini 2.5 Pro, GPT-Audio-mini)   20  ... by task
"""

import argparse
import csv
from decimal import ROUND_HALF_UP, Decimal
import json
import os

from musiclistenbench import paths
from musiclistenbench.scoring import metrics as M

PER_ITEM = os.path.join(paths.RESULTS_DIR, "per_item")

OPEN = [  # (display name, file slug)
    ("Qwen2-Audio", "qwen2_audio"), ("Qwen2.5-Omni", "qwen2_5_omni"), ("Audio Flamingo 3", "audio_flamingo3"),
    ("Phi-4-multimodal", "phi4_multimodal"), ("Kimi-Audio", "kimi_audio"), ("MiMo-Audio", "mimo_audio"),
    ("Fun-Audio-Chat", "fun_audio_chat"), ("Step-Audio 2", "step_audio2"),
]
COMMERCIAL = [
    ("GPT-Audio-mini", "gpt_audio_mini"), ("GPT-Audio-1.5", "gpt_audio_1_5"),
    ("Gemini 2.5 Pro", "gemini_2_5_pro"), ("Gemini 3.8 Flash", "gemini_3_8_flash"),
]
TRAINED = [  # (display name, slug of the untrained model)
    ("Qwen2-Audio", "qwen2_audio"), ("Qwen2.5-Omni", "qwen2_5_omni"),
    ("Audio Flamingo 3", "audio_flamingo3"), ("Phi-4-multimodal", "phi4_multimodal"),
]
GRPO_VARIANTS = [("LoRA", "_grpo_lora"), ("full", "_grpo_full")]
FACTORS = [("EQ", "eq"), ("Room", "room"), ("Hiss", "hiss"), ("Transp.", "transposition")]

_cache = {}


def load(slug):
    """Score of one model (clean + flip/stay files), cached."""
    if slug not in _cache:
        answers = {}
        for suffix in ("", "_perturb"):
            path = os.path.join(PER_ITEM, f"{slug}{suffix}.jsonl")
            with open(path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        answers[M.item_id_from_voice(r["voice"])] = r.get("predicted")
        _cache[slug] = M.score(answers, REF)
    return _cache[slug]


def pct(x, nd=1):
    """Percentage with `nd` decimals; exact ties round up (61.15 -> 61.2)."""
    if x != x:
        return "n/a"
    exact = Decimal(repr(round(100 * x, 6)))
    return str(exact.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


class Table:
    def __init__(self, number, title, header):
        self.number, self.title, self.header, self.rows = number, title, header, []

    def add(self, *cells):
        self.rows.append([str(c) for c in cells])

    def markdown(self):
        out = [f"### Table {self.number}: {self.title}", "",
               "| " + " | ".join(self.header) + " |", "|" + "|".join("---" for _ in self.header) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in self.rows]
        return "\n".join(out) + "\n"

    def write_csv(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"table_{self.number}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            w.writerows(self.rows)


def clean_cols(s):
    return [pct(s["clean"][t]["accuracy"]) for t in M.TASKS + ["All"]]


def fs_cols(s):
    f = s["flip_stay"]["All"]
    return [pct(f["acc_fs"]), pct(f["miss_rate"]), pct(f["false_flip_rate"])]


def table2():
    t = Table(2, "Results without training (%)",
              ["Model", "Mel.", "Har.", "Tim.", "Rhy.", "All (clean)", "Acc_FS", "MR", "FFR"])
    for name, slug in OPEN + COMMERCIAL:
        t.add(name, *clean_cols(load(slug)), *fs_cols(load(slug)))
    a = M.score(M.always_answer("A", REF), REF)
    t.add("Always 'A'", "", "", "", "", pct(a["clean"]["All"]["accuracy"]), *fs_cols(a))
    return t


def table3():
    t = Table(3, "GRPO results (%). none: no training; LoRA/full: GRPO with LoRA / full fine-tuning",
              ["Model", "GRPO", "Mel.", "Har.", "Tim.", "Rhy.", "All (clean)", "Acc_FS", "MR", "FFR"])
    for name, slug in TRAINED:
        t.add(name, "none", *clean_cols(load(slug)), *fs_cols(load(slug)))
        for label, suffix in GRPO_VARIANTS:
            s = load(slug + suffix)
            t.add("", label, *clean_cols(s), *fs_cols(s))
    return t


def _mr_ffr_table(number, title, entries):
    t = Table(number, title, ["Model", "GRPO", "MR Mel.", "MR Har.", "MR Tim.", "MR Rhy.",
                              "FFR Mel.", "FFR Har.", "FFR Tim.", "FFR Rhy."])
    for name, grpo, slug in entries:
        f = load(slug)["flip_stay"]
        t.add(name, grpo, *[pct(f[x]["miss_rate"]) for x in M.TASKS], *[pct(f[x]["false_flip_rate"]) for x in M.TASKS])
    return t


def table12():
    names = OPEN + [c for c in COMMERCIAL if c[1] in ("gpt_audio_mini", "gemini_2_5_pro")]
    return _mr_ffr_table(12, "Per-task miss rate (MR) and false-flip rate (FFR) without training (%)",
                         [(n, "none", s) for n, s in names])


def table13():
    entries = [(n, label, s + suf) for n, s in TRAINED for label, suf in GRPO_VARIANTS]
    return _mr_ffr_table(13, "Per-task miss rate and false-flip rate after GRPO (%)", entries)


def table14():
    t = Table(14, "Letter bias on the clean test (%)",
              ["Model", "GRPO", "A rate", "Acc|A", "Acc|B", "Balanced", "Gap"])
    ordered = []
    for name, slug in TRAINED:
        ordered.append((name, "none", slug))
        ordered += [("", label, slug + suf) for label, suf in GRPO_VARIANTS]
    ordered += [("GPT-Audio-mini", "none", "gpt_audio_mini"), ("Gemini 2.5 Pro", "none", "gemini_2_5_pro")]
    for name, grpo, slug in ordered:
        c = load(slug)["clean"]["All"]
        t.add(name, grpo, pct(c["a_rate"]), pct(c["acc_given_A"]), pct(c["acc_given_B"]),
              pct(c["balanced_accuracy"]), pct(c["letter_gap"]))
    return t


def table15():
    t = Table(15, "Per-task flip/stay accuracy without training (%)", ["Model", "Mel.", "Har.", "Tim.", "Rhy.", "All"])
    for name, slug in OPEN + COMMERCIAL:
        f = load(slug)["flip_stay"]
        t.add(name, *[pct(f[x]["acc_fs"]) for x in M.TASKS + ["All"]])
    return t


def table16():
    t = Table(16, "Drop in miss rate and false-flip rate from no training to GRPO, in points; z-test of "
                  "'miss rate fell more than false-flip rate'",
              ["Model", "GRPO", "dMR", "dFFR", "Ratio", "z", "p"])
    for name, slug in TRAINED:
        before = load(slug)["flip_stay"]["All"]
        for label, suf in GRPO_VARIANTS:
            after = load(slug + suf)["flip_stay"]["All"]
            d_mr = before["miss_rate"] - after["miss_rate"]
            d_ff = before["false_flip_rate"] - after["false_flip_rate"]
            z, p = M.two_rate_drop_test(before["miss_rate"], after["miss_rate"],
                                        before["false_flip_rate"], after["false_flip_rate"])
            ratio = f"{d_mr / d_ff:.1f}" if d_ff > 0 and d_mr > 0 else "n/a"
            t.add(name if label == "LoRA" else "", label + ("*" if slug == "audio_flamingo3" and label == "full" else ""),
                  f"{100 * d_mr:.1f}", f"{100 * d_ff:.1f}", ratio, f"{z:.1f}",
                  "<0.001" if p < 0.001 else f"{p:.4f}" if p < 0.01 else f"{p:.2f}")
    return t


def _factor_rows(entries):
    for name, grpo, slug in entries:
        yield name, grpo, load(slug)


def table19():
    t = Table(19, "False-flip rate (%) for each kind of stay change, pooled over tasks (lower is better)",
              ["Model", "GRPO", "EQ", "Room", "Hiss", "Transp.", "All"])
    entries = []
    for name, slug in TRAINED:
        entries.append((name, "none", slug))
        entries += [("", label, slug + suf) for label, suf in GRPO_VARIANTS]
    entries += [(n, "none", s) for n, s in OPEN[4:]] + [(n, "none", s) for n, s in COMMERCIAL]
    for name, grpo, s in _factor_rows(entries):
        pf = s["per_factor_error"]
        t.add(name, grpo, *[pct(pf[f"All/stay/{key}"]["error_rate"]) for _, key in FACTORS],
              pct(s["flip_stay"]["All"]["false_flip_rate"]))
    return t


def table20():
    header = ["Model", "GRPO"]
    for task in ("Mel.", "Har."):
        header += [f"{task} EQ", f"{task} Room", f"{task} Hiss", f"{task} Tr."]
    for task in ("Tim.", "Rhy."):
        header += [f"{task} EQ", f"{task} Room", f"{task} Hiss"]
    t = Table(20, "False-flip rate (%) by task and stay change (lower is better)", header)
    entries = []
    for name, slug in TRAINED:
        entries.append((name, "none", slug))
        entries += [("", label, slug + suf) for label, suf in GRPO_VARIANTS]
    entries += [(n, "none", s) for n, s in OPEN[4:]] + [(n, "none", s) for n, s in COMMERCIAL]
    for name, grpo, s in _factor_rows(entries):
        pf = s["per_factor_error"]
        cells = []
        for task, facs in (("Q1_melody", ("eq", "room", "hiss", "transposition")),
                           ("Q2_harmony", ("eq", "room", "hiss", "transposition")),
                           ("Q3_timbre", ("eq", "room", "hiss")), ("Q4_rhythm", ("eq", "room", "hiss"))):
            cells += [pct(pf[f"{task}/stay/{fac}"]["error_rate"]) for fac in facs]
        t.add(name, grpo, *cells)
    return t


def table21():
    """Errors on melody, harmony and timbre split by the correct answer (same / different)."""
    t = Table(21, "Errors (%) on melody, harmony and timbre, split by the correct answer",
              ["Items", "n same", "n diff.", "Gemini 2.5 Pro: same", "Gemini 2.5 Pro: diff.",
               "GPT-Audio-mini: same", "GPT-Audio-mini: diff."])
    answers = {}
    for slug in ("gemini_2_5_pro", "gpt_audio_mini"):
        answers[slug] = {}
        for suffix in ("", "_perturb"):
            with open(os.path.join(PER_ITEM, f"{slug}{suffix}.jsonl")) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        answers[slug][M.item_id_from_voice(r["voice"])] = r.get("predicted")
    groups = [("Clean", lambda r: r["kind"] == "clean"), ("flip", lambda r: r["kind"] == "flip"),
              ("stay, EQ", lambda r: r["kind"] == "stay" and r["factor"] == "eq"),
              ("stay, room echo", lambda r: r["kind"] == "stay" and r["factor"] == "room"),
              ("stay, hiss", lambda r: r["kind"] == "stay" and r["factor"] == "hiss"),
              ("stay, transposition", lambda r: r["kind"] == "stay" and r["factor"] == "transposition")]
    # 'same' items are those whose gold letter is A on the same/different tasks (A means same)
    for label, sel in groups:
        ids = [i for i, r in REF.items() if sel(r) and r["task"] != "Q4_rhythm"]
        same = [i for i in ids if REF[i]["gold"] == "A"]
        diff = [i for i in ids if REF[i]["gold"] == "B"]
        cells = []
        for slug in ("gemini_2_5_pro", "gpt_audio_mini"):
            for grp in (same, diff):
                wrong = sum(answers[slug].get(i) != REF[i]["gold"] for i in grp)
                cells.append(pct(wrong / len(grp)))
        t.add(label, len(same), len(diff), *cells)
    return t


TABLES = {2: table2, 3: table3, 12: table12, 13: table13, 14: table14, 15: table15, 16: table16,
          19: table19, 20: table20, 21: table21}


def main():
    global REF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=int, nargs="*", default=sorted(TABLES), choices=sorted(TABLES))
    ap.add_argument("--csv-dir", default=None, help="also write one CSV per table here")
    args = ap.parse_args()
    REF = M.load_reference()
    for n in args.table:
        t = TABLES[n]()
        print(t.markdown())
        if args.csv_dir:
            t.write_csv(args.csv_dir)


REF = None

if __name__ == "__main__":
    main()
