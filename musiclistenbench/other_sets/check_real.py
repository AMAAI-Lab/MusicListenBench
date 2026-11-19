"""Label checks on the built audio (clean items, before any stay change), from the audio alone:
Melody   pYIN pitch of the edited note (note 2) in clip B minus clip A: ~0 cents in 'same' pairs, ~ +-level in 'different'.
Harmony  chroma energy of the third's pitch class (12-bin CQT chroma, whole clip): ratio B/A ~1 in 'same' pairs, lower in
         'different' pairs (the third moved away).
Rhythm   onset-envelope tempo ratio (faster / slower clip) ~ the stretch ratio.
Timbre   none (labels are the dataset's instrument classes).
Writes label_checks.json next to the item files and prints a summary."""
import json, os, sys
from collections import defaultdict
import numpy as np, soundfile as sf, librosa
from concurrent.futures import ProcessPoolExecutor
D = os.environ.get("REAL_OUT", "generated/real"); SR = 44100
WIN = {"Q1_melody": 2.5, "Q2_harmony": 2.0, "Q3_timbre": 2.5, "Q4_rhythm": 4.0}
specs = {f"{s['task']}__real_test__{s['level']}__{s['idx']:06d}": s for s in json.load(open(f"{D}/specs.json"))["main"]}

def clips(voice, task):
    y, _ = sf.read(f"{D}/audio_test/{voice}", dtype="float32"); n = int(round(WIN[task] * SR))
    return y[:n], y[n + SR:n + SR + n]

def f0(y, a, b):
    seg = y[int(a * SR):int(b * SR)]
    if len(seg) < 2048: return np.nan
    f, v, _ = librosa.pyin(seg, fmin=70, fmax=1400, sr=SR, frame_length=2048)
    f = f[v] if v.any() else f[~np.isnan(f)]
    return np.nanmedian(f) if len(f) else np.nan

def check(m):
    task = m["task"]; s = specs[m["item"]]; A, Bc = clips(m["voice"], task)
    if task == "Q1_melody":
        on1, on2 = s["c"]["notes"][1][0], s["c"]["notes"][2][0]
        a, b = on1 + 0.03, on2 - 0.01
        return dict(task=task, answer=m["answer"], level=s["level"] * s.get("sign", 1),
                    cents=1200 * np.log2(f0(Bc, a, b) / f0(A, a, b)))
    if task == "Q2_harmony":
        ca = librosa.feature.chroma_cqt(y=A, sr=SR).mean(1); cb = librosa.feature.chroma_cqt(y=Bc, sr=SR).mean(1)
        third = (s["c"]["root_pc"] + (4 if s["c"]["quality"] == "major" else 3)) % 12
        return dict(task=task, answer=m["answer"], level=s["level"], ratio=float((cb[third] / cb.sum()) / (ca[third] / ca.sum())))
    if task == "Q4_rhythm":
        t = [librosa.feature.tempo(onset_envelope=librosa.onset.onset_strength(y=x, sr=SR), sr=SR, aggregate=None) for x in (A, Bc)]
        ta, tb = (np.median(x) for x in t)
        # autocorrelation tempo can jump octaves; compare onset counts too
        na, nb = (len(librosa.onset.onset_detect(y=x, sr=SR)) for x in (A, Bc))
        fast, slow = (na, nb) if m["answer"] == "first" else (nb, na)
        return dict(task=task, answer=m["answer"], level=s["level"], onset_ratio=fast / max(slow, 1))
    return None

if __name__ == "__main__":
    meta = [json.loads(l) for l in open(f"{D}/eval.meta.jsonl")]
    with ProcessPoolExecutor(48) as ex:
        res = [r for r in ex.map(check, meta, chunksize=2) if r]
    os.makedirs(D, exist_ok=True)
    json.dump(res, open(f"{D}/label_checks.json", "w"), default=float)
    g = defaultdict(list)
    for r in res:
        if r["task"] == "Q1_melody":
            g[("melody", r["answer"], "abs dev from target (cents)")].append(abs(r["cents"] - (0 if r["answer"] == "same" else r["level"])))
        elif r["task"] == "Q2_harmony":
            g[("harmony", r["answer"], f"third-chroma ratio B/A, level {r['level']}")].append(r["ratio"])
        else:
            g[("rhythm", r["level"], "onset-count ratio fast/slow")].append(r["onset_ratio"])
    for k in sorted(g, key=str):
        v = np.array(g[k], float); v = v[~np.isnan(v)]
        print(k, f"n={len(v)} median={np.median(v):.3f} p10={np.percentile(v, 10):.3f} p90={np.percentile(v, 90):.3f}")
