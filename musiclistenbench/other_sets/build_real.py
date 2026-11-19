"""Builds the real-recordings test set of MusicListenBench (evaluation only, no training split).

Same four tasks, clip lengths, file format (clip A + 1.0 s silence + clip B, 44.1 kHz 16-bit mono, -23 LUFS per clip),
question wordings, letter rule (test: A = same), flip and stay versions and scoring as the rendered benchmark. 150 base
items per task, each with a flip and a stay version (600 clean and 1,200 flip/stay items). Labels come from the
annotations of the datasets.

Melody   GuitarSet solo excerpts, microphone recording. A phrase of 4-6 notes, monophonic according to the string-wise
         note annotations, 2.5 s. Different: note 2 is pitch-shifted by 100/200/400/700 cents (Rubber Band); in same
         pairs the same region passes through the shifter at 0 cents.
Harmony  GuitarSet comping excerpts, hexaphonic pickup (one cleaned track per string), 2.0 s window starting at a strum
         in which every annotated note belongs to one major or minor triad and nothing rings in from before. Different:
         the strings that play the third are pitch-shifted by 100/200/300 cents (major down, minor up), the other
         strings stay untouched; in same pairs those strings pass through the shifter at 0 cents.
Timbre   Medley-solos-DB (clarinet, distorted electric guitar, flute, piano, tenor saxophone, trumpet, violin; the singer
         class is left out), 2.5 s. Every pair uses two different recordings, for same and for different pairs.
Rhythm   MUSDB18-HQ drum stems, 6 s source window with an onset in every second; Rubber Band time-stretch at 1.0 and at
         1.10/1.20/1.35/1.50, cut to 4.0 s. Flip swaps the clips. (fetch_musdb_drums.py downloads the stems.)
Stay changes (EQ, hiss, room, and transposition of 0-200 cents for melody and harmony) are made with the generator.

Also builds a small extra harmony set (eval_take.json): two strums of one chord in the same excerpt, untouched (same) or
with the third of the second strum moved (different).

Input: REAL_DATA_DIR with guitarset/{annotation,mic,hex}, medley_solos/{audio,Medley-solos-DB_metadata.csv} and
musdb18hq/drums/*.wav. Output: REAL_OUT/{eval,eval_perturb,eval_take}.json (+ .meta.jsonl, specs.json) and audio_test/.

    python -m musiclistenbench.other_sets.build_real
"""
import csv, glob, json, os, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf, librosa
from scipy.signal import resample_poly
from musiclistenbench.generator.data import render
from musiclistenbench.benchmark.build_probe import make_entry
from musiclistenbench.other_sets import build_slakh as B   # read, shift, stretch, finalize, melody_raw, rhythm_raw, cfg
from musiclistenbench.other_sets import slakh_candidates as C
RD = os.environ.get("REAL_DATA_DIR", "real_data"); OUT = os.environ.get("REAL_OUT", "generated/real"); SR = 44100
os.environ.setdefault("SLAKH_OUT", OUT)   # the shared helpers only use it for a scratch path
N_TEST = int(os.environ.get("REAL_NTEST", 150))
WIN = B.WIN

# ------------------------------------------------------------------ sources
def gs_notes(jams):
    d = json.load(open(jams)); out = []
    for a in d["annotations"]:
        if a["namespace"] == "note_midi":
            s = int(a["annotation_metadata"]["data_source"])
            out += [(n["time"], n["time"] + n["duration"], int(round(n["value"])), s) for n in a["data"]]
    return sorted(out)

def gs_audio(stem, kind):
    pat = {"mic": f"{RD}/guitarset/mic/**/{stem}_mic.wav", "hex": f"{RD}/guitarset/hex/**/{stem}_hex_cln.wav"}[kind]
    g = glob.glob(pat, recursive=True); return g[0] if g else None

def melody_cands():
    out = []
    for j in sorted(glob.glob(f"{RD}/guitarset/annotation/*_solo.jams")):
        stem = os.path.basename(j)[:-5]; audio = gs_audio(stem, "mic")
        if not audio: continue
        N = gs_notes(j); last = -9
        for w in C.melody_windows(dict(is_drum=False, inst_class="Guitar", notes=[(a, b, p, 1) for a, b, p, _ in N])):
            if w["t0"] - last >= WIN["Q1_melody"]:
                out.append(dict(src=stem, audio=audio, **w)); last = w["t0"]
    return out

def harmony_cands(win=2.0):
    out = []
    for j in sorted(glob.glob(f"{RD}/guitarset/annotation/*_comp.jams")):
        stem = os.path.basename(j)[:-5]; audio = gs_audio(stem, "hex")
        if not audio: continue
        N = gs_notes(j); last = -9
        for t0 in sorted({round(a, 3) for a, _, _, _ in N}):
            if t0 - last < win: continue
            grp = [n for n in N if t0 - 0.005 <= n[0] <= t0 + 0.08]
            if len({n[3] for n in grp}) < 3: continue
            W = [n for n in N if n[0] < t0 + win and n[1] > t0 + 0.03]
            if any(n[0] < t0 - 0.005 for n in W): continue          # nothing rings in from before
            pcs = {n[2] % 12 for n in W}
            if len(pcs) != 3: continue
            q = None
            for r in pcs:
                rel = tuple(sorted((p - r) % 12 for p in pcs))
                if rel == (0, 4, 7): q = ("major", r)
                if rel == (0, 3, 7): q = ("minor", r)
            if not q: continue
            third = (q[1] + (4 if q[0] == "major" else 3)) % 12
            tstr = sorted({n[3] for n in W if n[2] % 12 == third})
            if any(n[2] % 12 != third for n in W if n[3] in tstr): continue
            out.append(dict(src=stem, audio=audio, t0=t0, quality=q[0], root_pc=q[1], third_strings=tstr,
                            pitches=sorted({n[2] for n in W}))); last = t0
    return out

MEDLEY_KEEP = {"clarinet", "distorted electric guitar", "flute", "piano", "tenor saxophone", "trumpet", "violin"}
def timbre_pool():
    files = {os.path.basename(p).rsplit("_", 1)[-1][:-4]: p for p in glob.glob(f"{RD}/medley_solos/audio/**/*.wav", recursive=True) if not os.path.basename(p).startswith("._")}
    by = defaultdict(list)
    for r in csv.DictReader(open(f"{RD}/medley_solos/Medley-solos-DB_metadata.csv")):
        if r["instrument"] in MEDLEY_KEEP and r["uuid4"] in files:
            by[r["instrument"]].append(dict(audio=files[r["uuid4"]], song=r["song_id"], inst=r["instrument"], uuid=r["uuid4"], subset=r["subset"]))
    return by

def rhythm_cands(src=6.0):
    out = []
    for p in sorted(glob.glob(f"{RD}/musdb18hq/drums/*.wav")):
        y, sr = sf.read(p, dtype="float32"); y = y.mean(1) if y.ndim > 1 else y
        if np.sqrt(np.mean(y ** 2)) < 1e-3: continue
        on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False); last = -9
        for t0 in on:
            if t0 - last < src or t0 + src > len(y) / sr: continue
            w = on[(on >= t0) & (on < t0 + src)]
            seg = y[int(t0 * sr):int((t0 + src) * sr)]
            if len(w) >= 12 and all(((w >= t0 + k) & (w < t0 + k + 1)).sum() >= 1 for k in range(int(src))) \
                    and min(np.sqrt(np.mean(seg[k * sr:(k + 1) * sr] ** 2)) for k in range(int(src))) > 3e-3:
                out.append(dict(src=os.path.basename(p)[:-4], audio=p, t0=float(t0))); last = t0
    return out

# ------------------------------------------------------------------ raw audio
def read_medley(path, dur, off):
    y, sr = sf.read(path, dtype="float32"); y = y.mean(1) if y.ndim > 1 else y
    if sr != SR: y = resample_poly(y, SR, sr).astype(np.float32)
    n = int(round(dur * SR)); a = int(round(off * SR)); y = y[a:a + n]
    return np.pad(y, (0, n - len(y))).astype(np.float32)

def harmony_raw(c, level):
    """-> (A, B_same, B_diff) from the hexaphonic recording: only the third's strings are processed."""
    pad = 0.05; n = int(round(WIN["Q2_harmony"] * SR)); a0 = max(0.0, c["t0"] - pad); lead = int(round((c["t0"] - a0) * SR))
    info = sf.info(c["audio"]); s = int(round(a0 * SR)); m = n + lead + int(pad * SR)
    X, _ = sf.read(c["audio"], start=s, stop=min(s + m, info.frames), dtype="float32", always_2d=True)
    X = np.pad(X, ((0, m - len(X)), (0, 0)))
    cents = -level if c["quality"] == "major" else level
    def mix(cn):
        y = X.copy()
        for st in c["third_strings"]:
            y[:, st] = B.shift(np.ascontiguousarray(X[:, st]), cn)
        return y.sum(1)[lead:lead + n].astype(np.float32)
    return X.sum(1)[lead:lead + n].astype(np.float32), mix(0.0), mix(cents)

def raw_triple(spec):
    t = spec["task"]
    if t == "Q1_melody":
        A, Bs, Bd = B.melody_raw(spec["c"], spec["sign"] * spec["level"])
    elif t == "Q2_harmony":
        A, Bs, Bd = harmony_raw(spec["c"], spec["level"])
    elif t == "Q3_timbre":
        A, Bs, Bd = (read_medley(x["audio"], WIN[t], x["off"]) for x in (spec["c_a"], spec["c_same"], spec["c_other"]))
    else:
        ref, fast = B.rhythm_raw(spec["c"], spec["level"]); return ref, fast, None
    return A, Bs, Bd

def make_item(spec):
    task, idx = spec["task"], spec["idx"]; rng = random.Random(spec["seed"]); gold_base = spec["gold_base"]
    item = f"{task}__real_test__{spec['level']}__{idx:06d}"
    A, Bs, Bd = raw_triple(spec)
    if task == "Q4_rhythm":
        rawA, rawB = (Bs, A) if gold_base == "first" else (A, Bs); rawF = None   # Bs = faster clip here
    else:
        rawA = A; rawB = Bs if gold_base == "same" else Bd; rawF = Bd if gold_base == "same" else Bs
    clipA, clipB = B.finalize((rawA, rawB), task); out = []
    def add(kind, a, b, gold_ans, factor=None):
        rel = f"{task}/{item}__{kind}{'__' + factor if factor else ''}.wav"; p = f"{OUT}/audio_test/{rel}"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        sf.write(p, np.concatenate([a, np.zeros(SR, np.float32), b]), SR, subtype="PCM_16")
        e = make_entry(task, {"answer": gold_ans}, rng, rel, counterbalance=False)
        out.append((e, dict(voice=rel, item=item, task=task, kind=kind, factor=factor, level=spec["level"], answer=gold_ans,
                             gold=e["conversations"][1]["value"], source=spec["source"])))
    add("base", clipA, clipB, gold_base)
    if task == "Q4_rhythm":
        add("flip", clipB, clipA, "second" if gold_base == "first" else "first", "pulse_rate")
    else:
        add("flip", *B.finalize((rawA, rawF), task), "different" if gold_base == "same" else "same",
            {"Q1_melody": "note_identity", "Q2_harmony": "third_quality", "Q3_timbre": "instrument"}[task])
    fam = spec["stay"]
    if fam == "transposition":
        b = B.finalize((rawA, B.shift(rawB, spec["transp_cents"])), task)[1]
    else:
        fc = B.cfg().transforms["families"][fam]
        tr, _ = render.apply_family_to_pair([rawA, rawB], SR, fam, fc, np.random.default_rng(spec["seed"] + 7), seed=spec["seed"] + 7)
        b = B.finalize(tr, task)[1]
    add("stay", clipA, b, gold_base, fam)
    return out

# ------------------------------------------------------------------ selection
LEVELS = {"Q1_melody": [100, 200, 400, 700], "Q2_harmony": [100, 200, 300], "Q3_timbre": ["instrument"], "Q4_rhythm": [1.10, 1.20, 1.35, 1.50]}
def base_spec(task, k, rng, source):
    lv = LEVELS[task]; stays = ["eq", "hiss", "room", "transposition"] if task in ("Q1_melody", "Q2_harmony") else ["eq", "hiss", "room"]
    golds = ["first", "second"] if task == "Q4_rhythm" else ["same", "different"]
    return dict(task=task, idx=k, seed=rng.randrange(2 ** 31), level=lv[k % len(lv)], gold_base=golds[(k // len(lv)) % 2],
                stay=stays[(k + k // (2 * len(lv))) % len(stays)], transp_cents=rng.uniform(0, 200), source=source)

def loud(spec):
    try:
        return all(np.sqrt(np.mean(x ** 2)) > 1e-3 for x in raw_triple(spec) if x is not None)
    except Exception as e:
        print("screen error", spec["task"], spec["source"], repr(e), flush=True); return False

def pick(task, cands, rng, cap):
    rng.shuffle(cands); per = Counter(); specs = []; k = 0
    for c in cands:
        if len(specs) >= int(N_TEST * 1.3): break
        if per[c["src"]] >= cap: continue
        sp = base_spec(task, k, rng, {kk: v for kk, v in c.items() if kk not in ("audio", "notes")})
        if task == "Q1_melody":
            sp["sign"] = rng.choice([1, -1]); p1 = c["notes"][1][2] + sp["sign"] * sp["level"] / 100
            if not 40 <= p1 <= 90: continue
        sp["c"] = c; per[c["src"]] += 1; specs.append(sp); k += 1
    return specs

def pick_timbre(pool, rng):
    insts = sorted(pool); specs = []; per_song = Counter()
    for k in range(int(N_TEST * 1.3)):
        i1 = insts[k % len(insts)]; i2 = rng.choice([i for i in insts if i != i1])
        for _ in range(1000):
            a, s = rng.sample(pool[i1], 2); o = rng.choice(pool[i2])
            if len({a["song"], s["song"], o["song"]}) == 3 and per_song[a["song"]] < 6: break
        else:
            continue
        per_song[a["song"]] += 1
        src = dict(a=[i1, a["song"], a["uuid"]], same=[i1, s["song"], s["uuid"]], other=[i2, o["song"], o["uuid"]])
        sp = base_spec("Q3_timbre", k, rng, src)
        sp.update(c_a=dict(audio=a["audio"], off=rng.uniform(0, 0.4)), c_same=dict(audio=s["audio"], off=rng.uniform(0, 0.4)),
                  c_other=dict(audio=o["audio"], off=rng.uniform(0, 0.4)))
        specs.append(sp)
    return specs

# ------------------------------------------------------------------ second takes (harmony, extra set eval_take.json)
def take_pairs(hc, rng):
    """Two strums of the same voicing (same set of MIDI pitches, same strings carry the third) in one excerpt,
    at least 2 s apart. 'same' = take 1 vs take 2, both untouched; 'different' = take 1 vs take 2 with its third moved."""
    by = defaultdict(list)
    for c in hc: by[(c["src"], tuple(c["pitches"]), tuple(c["third_strings"]))].append(c)
    pairs = []
    for occ in by.values():
        occ = sorted(occ, key=lambda c: c["t0"])
        for a, b in zip(occ[::2], occ[1::2]):
            if b["t0"] - a["t0"] >= 2.0: pairs.append((a, b))
    rng.shuffle(pairs); return pairs

def make_take(spec):
    task = "Q2_harmony"; rng = random.Random(spec["seed"]); gold = spec["gold_base"]
    item = f"{task}__realtake_test__{spec['level']}__{spec['idx']:06d}"
    A = harmony_raw(spec["c1"], spec["level"])[0]; _, Bs, Bd = harmony_raw(spec["c2"], spec["level"])
    a, b = B.finalize((A, Bs if gold == "same" else Bd), task)
    rel = f"{task}/{item}__base.wav"; p = f"{OUT}/audio_test/{rel}"; os.makedirs(os.path.dirname(p), exist_ok=True)
    sf.write(p, np.concatenate([a, np.zeros(SR, np.float32), b]), SR, subtype="PCM_16")
    e = make_entry(task, {"answer": gold}, rng, rel, counterbalance=False)
    return e, dict(voice=rel, item=item, task=task, kind="base", factor=None, level=spec["level"], answer=gold,
                   gold=e["conversations"][1]["value"], source=dict(take1=[spec["c1"]["src"], spec["c1"]["t0"]], take2=[spec["c2"]["src"], spec["c2"]["t0"]]))

if __name__ == "__main__":
    rng = random.Random(20261109); os.makedirs(OUT, exist_ok=True); specs = []
    cands = {"Q1_melody": (melody_cands(), 3), "Q2_harmony": (harmony_cands(), 4), "Q4_rhythm": (rhythm_cands(), 2)}
    for task in ("Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"):
        sp = pick_timbre(timbre_pool(), rng) if task == "Q3_timbre" else pick(task, cands[task][0], rng, cands[task][1])
        with ProcessPoolExecutor(48) as ex:
            ok = list(ex.map(loud, sp, chunksize=4))
        sp = [s for s, o in zip(sp, ok) if o][:N_TEST]
        for k, s in enumerate(sp): s["idx"] = k
        print(task, "candidates", len(cands[task][0]) if task in cands else "-", "items", len(sp), flush=True)
        assert len(sp) == N_TEST, (task, len(sp))
        specs += sp
    with ProcessPoolExecutor(64) as ex:
        res = list(ex.map(make_item, specs, chunksize=2))
    def dump(kinds, path):
        rows = [(e, m) for r in res for e, m in r if m["kind"] in kinds]
        json.dump([e for e, _ in rows], open(path + ".json", "w"))
        with open(path + ".meta.jsonl", "w") as f:
            for _, m in rows: f.write(json.dumps(m, default=str) + "\n")
        return len(rows)
    print("eval", dump({"base"}, f"{OUT}/eval"), "eval_perturb", dump({"flip", "stay"}, f"{OUT}/eval_perturb"), flush=True)
    # second-take set
    tp = take_pairs(cands["Q2_harmony"][0], random.Random(20261110)); tspecs = []
    for k, (c1, c2) in enumerate(tp):
        tspecs.append(dict(idx=k, seed=rng.randrange(2 ** 31), level=[100, 200, 300][k % 3], gold_base=["same", "different"][(k // 3) % 2], c1=c1, c2=c2))
    with ProcessPoolExecutor(48) as ex:
        tres = list(ex.map(make_take, tspecs, chunksize=2))
    json.dump([e for e, _ in tres], open(f"{OUT}/eval_take.json", "w"))
    with open(f"{OUT}/eval_take.meta.jsonl", "w") as f:
        for _, m in tres: f.write(json.dumps(m, default=str) + "\n")
    print("eval_take", len(tres), Counter(m["answer"] for _, m in tres), flush=True)
    json.dump(dict(main=specs, take=tspecs), open(f"{OUT}/specs.json", "w"), default=str)
    # label sanity: gold letter balance per task and kind
    m = [json.loads(l) for p in ("eval", "eval_perturb") for l in open(f"{OUT}/{p}.meta.jsonl")]
    print(Counter((x["task"], x["kind"], x["gold"]) for x in m))
