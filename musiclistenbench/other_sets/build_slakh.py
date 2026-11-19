"""Builds the Slakh2100 version of MusicListenBench. Slakh stems are rendered from MIDI with sample-based
virtual instruments, so this is not a set of live recordings.

Same four tasks, clip lengths, file format (clip A + 1.0 s silence + clip B, 44.1 kHz 16-bit mono, -23 LUFS per
clip), question wordings and letter rules as the main benchmark: 2,500 training items per task (Slakh train and
validation tracks) and 250 clean test items per task (Slakh test tracks), each test item with one flip and one stay
version. Labels come from the stem MIDI.

Melody   a real phrase of 4-6 notes (MIDI-verified monophonic) from one stem, 2.5 s. Different: note 2 is
         pitch-shifted by 100/200/400/700 cents (Rubber Band, formants preserved); in same pairs the same region
         passes through the shifter at 0 cents.
Harmony  one isolated note from a piano/guitar/organ/strings stem; the triad is root, third and fifth made from that
         note with the same shifter (0, 300 or 400, 700 cents), 2.0 s. Different: the third moves by 100/200/300
         cents (major down, minor up), as in the generator.
Timbre   piano, acoustic guitar, violin / string ensemble and flute stems, 2.5 s. Both clips are always different
         passages: same = two passages of one stem, different = passages of stems of two families.
Rhythm   a drum stem, 6 s source; a clip is the source time-stretched by 1.0 or by the rate ratio 1.10/1.20/1.35/1.50
         (faster), cut to 4.0 s. Both clips come from the same source window. Flip swaps the clips.
Held out: tracks (the Slakh split) and plugins (no training item uses a plugin that a test item uses).

Stay changes (room, EQ, hiss, and transposition of 0-200 cents for melody and harmony) are made with the generator.

    SLAKH_OUT=generated/slakh python -m musiclistenbench.other_sets.build_slakh   (after scan_slakh)
"""
import json, os, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from pedalboard import time_stretch
from musiclistenbench.generator import config as config_mod
from musiclistenbench.generator.data import render
from musiclistenbench.benchmark.build_probe import make_entry
from musiclistenbench.other_sets import slakh_candidates as C
OUT = os.environ.get("SLAKH_OUT", "generated/slakh"); SR = 44100
WIN = {"Q1_melody": 2.5, "Q2_harmony": 2.0, "Q3_timbre": 2.5, "Q4_rhythm": 4.0}
N_TEST, N_TRAIN = int(os.environ.get("SLAKH_NTEST", 250)), int(os.environ.get("SLAKH_NTRAIN", 2500))
_cfg = None
def cfg():
    global _cfg
    if _cfg is None: _cfg = config_mod.load_config(output_root=f"{OUT}/_unused")
    return _cfg

def read(path, t0, dur):
    i = sf.info(path); a = int(round(t0 * SR)); n = int(round(dur * SR))
    y, _ = sf.read(path, start=a, stop=min(a + n, i.frames), dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    return np.pad(y, (0, n - len(y))).astype(np.float32)

def shift(y, cents):
    return time_stretch(y[None], SR, stretch_factor=1.0, pitch_shift_in_semitones=cents / 100.0)[0][:len(y)].astype(np.float32)

def stretch(y, factor, n):
    out = time_stretch(y[None], SR, stretch_factor=factor)[0]
    return np.pad(out, (0, max(0, n - len(out))))[:n].astype(np.float32)

def finalize(pair, task):
    return render.finalize_clips(list(pair), SR, cfg().target_lufs, round(WIN[task] * SR))[0]

def loud_enough(y): return np.sqrt(np.mean(y ** 2)) > 1e-3

# ------------------------------------------------------------------ raw pairs per task
def melody_raw(c, cents):
    """-> (A, B_same, B_diff): note 2 region [on1 - 20 ms, on2) shifted; 10 ms crossfades."""
    X = read(c["audio"], c["t0"], WIN["Q1_melody"]); on1, on2 = c["notes"][1][0], c["notes"][2][0]
    a, b = max(0, int((on1 - 0.02) * SR)), int(on2 * SR); pad = int(0.05 * SR); f = int(0.01 * SR)
    def edit(cn):
        lo, hi = max(0, a - pad), min(len(X), b + pad); seg = shift(X[lo:hi], cn)[a - lo:a - lo + (b - a)]
        y = X.copy(); w = np.ones(b - a, np.float32); w[:f] = np.linspace(0, 1, f); w[-f:] = np.linspace(1, 0, f)
        y[a:b] = X[a:b] * (1 - w) + seg * w; return y
    return X, edit(0.0), edit(cents)

def harmony_raw(c, quality, level):
    N = read(c["audio"], c["t0"], WIN["Q2_harmony"]); third = 400 if quality == "major" else 300
    moved = third - level if quality == "major" else third + level
    root, fifth = shift(N, 0.0), shift(N, 700.0)
    return root + shift(N, third) + fifth, root + shift(N, moved) + fifth

def rhythm_raw(c, ratio):
    src = read(c["audio"], c["t0"], 6.0); n = round(WIN["Q4_rhythm"] * SR)
    return stretch(src, 1.0, n), stretch(src, ratio, n)

# ------------------------------------------------------------------ item construction
def make_item(spec):
    """spec: dict(task, split, idx, ...). Returns list of (rel_voice, entry, meta) and writes audio."""
    task, split, idx = spec["task"], spec["split"], spec["idx"]; rng = random.Random(spec["seed"])
    gold_base = spec["gold_base"]; item = f"{task}__slakh_{split}__{spec['level']}__{idx:06d}"
    if task == "Q1_melody":
        A, Bs, Bd = melody_raw(spec["c"], spec["sign"] * spec["level"])
        rawB = Bs if gold_base == "same" else Bd; rawF = Bd if gold_base == "same" else Bs; rawA = A
    elif task == "Q2_harmony":
        A, D = harmony_raw(spec["c"], spec["quality"], spec["level"])
        rawA = A; rawB = A if gold_base == "same" else D; rawF = D if gold_base == "same" else A
    elif task == "Q3_timbre":
        s1, s2, t1 = (read(x["audio"], x["t0"], WIN[task]) for x in (spec["c_same1"], spec["c_same2"], spec["c_other"]))
        rawA = s1; rawB = s2 if gold_base == "same" else t1; rawF = t1 if gold_base == "same" else s2
    else:
        ref, fast = rhythm_raw(spec["c"], spec["level"])
        rawA, rawB = (fast, ref) if gold_base == "first" else (ref, fast); rawF = None
    clipA, clipB = finalize((rawA, rawB), task)
    out = []
    def add(kind, a, b, gold_ans, factor=None):
        rel = f"{task}/{item}__{kind}{'__' + factor if factor else ''}.wav"; p = f"{OUT}/audio_{split}/{rel}"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        sf.write(p, np.concatenate([a, np.zeros(SR, np.float32), b]), SR, subtype="PCM_16")
        e = make_entry(task, {"answer": gold_ans}, rng, rel, counterbalance=(split == "train"))
        out.append((e, dict(voice=rel, item=item, task=task, kind=kind, factor=factor, level=spec["level"], answer=gold_ans,
                             gold=e["conversations"][1]["value"], source=spec["source"])))
    add("base", clipA, clipB, gold_base)
    if split == "test":
        if task == "Q4_rhythm":
            other = "second" if gold_base == "first" else "first"; add("flip", clipB, clipA, other, "pulse_rate")
        else:
            other = "different" if gold_base == "same" else "same"
            add("flip", *finalize((rawA, rawF), task), other, {"Q1_melody": "note_identity", "Q2_harmony": "third_quality", "Q3_timbre": "instrument"}[task])
        fam = spec["stay"]
        if fam == "transposition":
            b = finalize((rawA, shift(rawB, spec["transp_cents"])), task)[1]
        else:
            fc = cfg().transforms["families"][fam]
            tr, _ = render.apply_family_to_pair([rawA, rawB], SR, fam, fc, np.random.default_rng(spec["seed"] + 7), seed=spec["seed"] + 7)
            b = finalize(tr, task)[1]
        add("stay", clipA, b, gold_base, fam)
    return out

# ------------------------------------------------------------------ selection
def select(stems, split_names, n, task, rng, banned_plugins, start_idx=0):
    pool = [s for s in stems if s["split"] in split_names and s["plugin"] not in banned_plugins]
    rng.shuffle(pool); specs = []; per_track = Counter(); cap = 3 if "test" in split_names else 15
    stays = ["eq", "hiss", "room", "transposition"] if task in ("Q1_melody", "Q2_harmony") else ["eq", "hiss", "room"]
    levels = {"Q1_melody": [100, 200, 400, 700], "Q2_harmony": [100, 200, 300], "Q4_rhythm": [1.10, 1.20, 1.35, 1.50]}.get(task, ["instrument"])
    golds = ["same", "different"] if task != "Q4_rhythm" else ["first", "second"]
    def base_spec(k, src):
        return dict(task=task, split="test" if "test" in split_names else "train", idx=start_idx + k, seed=rng.randrange(2 ** 31),
                    level=levels[k % len(levels)], gold_base=golds[(k // len(levels)) % 2], stay=stays[k % len(stays)],
                    transp_cents=rng.uniform(0, 200), source=src)
    if task == "Q3_timbre":
        fams = defaultdict(list)
        for s in pool:
            w, fam = C.timbre_windows(s)
            if fam and len(w) >= 2: fams[fam].append((s, w))
        names = sorted(fams); k = 0; tries = 0
        if len(names) < 2: return []
        while k < n and tries < 50 * n:
            tries += 1; f1 = names[(k + tries) % len(names)]  # move on when a family is blocked by the per-track cap
            f2 = rng.choice([f for f in names if f != f1])
            s, w = rng.choice(fams[f1]); t, wt = rng.choice(fams[f2])
            if s["track"] == t["track"] or per_track[s["track"]] >= cap: continue
            per_track[s["track"]] += 1; w1, w2 = rng.sample(w, 2); wt1 = rng.choice(wt)
            sp = base_spec(k, dict(same=[s["track"], s["stem"], s["plugin"], f1], other=[t["track"], t["stem"], t["plugin"], f2]))
            sp.update(c_same1=dict(audio=s["audio"], t0=w1["t0"]), c_same2=dict(audio=s["audio"], t0=w2["t0"]), c_other=dict(audio=t["audio"], t0=wt1["t0"]))
            specs.append(sp); k += 1
        return specs
    finder = {"Q1_melody": C.melody_windows, "Q4_rhythm": C.rhythm_windows, "Q2_harmony": isolated_notes}[task]
    cands = []
    for s in pool:
        for w in finder(s): cands.append((s, w))
    rng.shuffle(cands); k = 0
    for s, w in cands:
        if k >= n: break
        if per_track[s["track"]] >= cap: continue
        sp = base_spec(k, dict(track=s["track"], stem=s["stem"], plugin=s["plugin"], program=s["program"], t0=w["t0"]))
        c = dict(audio=s["audio"], **w)
        if task == "Q1_melody":
            sp["sign"] = rng.choice([1, -1])
            p1 = w["notes"][1][2] + sp["sign"] * sp["level"] / 100
            if not 40 <= p1 <= 90: continue
        if task == "Q2_harmony": sp["quality"] = rng.choice(["major", "minor"])
        sp["c"] = c; per_track[s["track"]] += 1; specs.append(sp); k += 1
    return specs

def isolated_notes(st, win=2.0):
    if st["is_drum"] or st["inst_class"] not in {"Piano", "Guitar", "Organ", "Strings", "Strings (continued)"}: return []
    N = st["notes"]; out = []
    for i, (s0, e0, p, v) in enumerate(N):
        if not 48 <= p <= 67 or e0 - s0 < 0.3: continue
        if i + 1 < len(N) and N[i + 1][0] < s0 + win: continue
        if any(e > s0 - 0.01 for s, e, _, _ in N[max(0, i - 30):i]): continue
        out.append(dict(t0=s0, pitch=p))
    return out

def screen(spec):
    """Drop windows whose stem audio is (near) silent."""
    t = spec["task"]
    if t == "Q3_timbre": xs = [spec["c_same1"], spec["c_same2"], spec["c_other"]]
    else: xs = [spec["c"]]
    return all(loud_enough(read(x["audio"], x["t0"], 6.0 if t == "Q4_rhythm" else WIN[t])) for x in xs)

def run_split(specs, name):
    with ProcessPoolExecutor(48) as ex:
        ok = list(ex.map(screen, specs, chunksize=8))
    return [s for s, o in zip(specs, ok) if o]

if __name__ == "__main__":
    stems = C.load(); rng = random.Random(20261107); os.makedirs(OUT, exist_ok=True)
    test_specs, train_specs = [], []
    for task in ("Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"):
        ts = run_split(select(stems, {"test"}, int(N_TEST * 1.3), task, rng, set()), "test")[:N_TEST]
        assert len(ts) == N_TEST, (task, len(ts))
        test_specs += ts
        banned = set()
        for s in ts:
            src = s["source"]; banned |= {src["plugin"]} if "plugin" in src else {src["same"][2], src["other"][2]}
        tr = run_split(select(stems, {"train", "validation"}, int(N_TRAIN * 1.2), task, rng, banned), "train")[:N_TRAIN]
        holdout = "tracks and plugins"
        if len(tr) < N_TRAIN:   # plugin hold-out leaves too few items: fall back to track hold-out only
            print(task, f"plugin hold-out leaves {len(tr)} training items; using track hold-out only", flush=True)
            tr = run_split(select(stems, {"train", "validation"}, int(N_TRAIN * 1.2), task, rng, set()), "train")[:N_TRAIN]
            holdout = "tracks only"
        assert len(tr) == N_TRAIN, (task, len(tr))
        with open(f"{OUT}/holdout.txt", "a") as f: f.write(f"{task}: held out {holdout}; {len(banned)} test plugins\n")
        train_specs += tr
        print(task, "test", len(ts), "train", len(tr), "banned plugins", len(banned), flush=True)
    with ProcessPoolExecutor(64) as ex:
        res_test = list(ex.map(make_item, test_specs, chunksize=4)); res_train = list(ex.map(make_item, train_specs, chunksize=8))
    def dump(res, kinds, path):
        rows = [(e, m) for r in res for e, m in r if m["kind"] in kinds]
        json.dump([e for e, _ in rows], open(path + ".json", "w"))
        with open(path + ".meta.jsonl", "w") as f:
            for _, m in rows: f.write(json.dumps(m) + "\n")
        return len(rows)
    print("eval", dump(res_test, {"base"}, f"{OUT}/eval"), "eval_perturb", dump(res_test, {"flip", "stay"}, f"{OUT}/eval_perturb"),
          "train", dump(res_train, {"base"}, f"{OUT}/train"))
    json.dump([{k: v for k, v in s.items() if k not in ("c", "c_same1", "c_same2", "c_other")} | {"c": s.get("c"), "timbre": [s.get(k) for k in ("c_same1", "c_same2", "c_other")]}
               for s in test_specs + train_specs], open(f"{OUT}/specs.json", "w"), default=str)
