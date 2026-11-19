"""Finds segments in the Slakh2100 stems for the four tasks. Every label comes from the stem MIDI.
Window lengths are those of the benchmark: melody 2.5 s, harmony 2.0 s, timbre 2.5 s, rhythm 4.0 s."""
import os
import pickle
from collections import defaultdict
import numpy as np

OUT_DIR = os.environ.get("SLAKH_OUT", "generated/slakh")
IDX = os.path.join(OUT_DIR, "stems_index.pkl")   # written by scan_slakh.py
MELODIC = {"Piano", "Guitar", "Strings", "Strings (continued)", "Brass", "Reed", "Pipe", "Synth Lead", "Organ", "Chromatic Percussion"}
CHORDAL = {"Piano", "Guitar", "Organ", "Strings", "Strings (continued)", "Synth Pad", "Brass"}
TIMBRE_FAMILY = {0: "piano", 1: "piano", 24: "guitar", 25: "guitar", 40: "violin", 48: "violin", 49: "violin", 73: "flute"}
QUAL = {(0, 4, 7): "major", (0, 3, 7): "minor", (0, 2, 7): "sus2", (0, 5, 7): "sus4"}

def load():
    return pickle.load(open(IDX, "rb"))

def melody_windows(st, win=2.5, max_onset_span=2.0):
    """Monophonic phrases: 4-6 onsets within 2.0 s, no overlap, nothing sounding into the window,
    pitches 48-84, notes >= 0.12 s, note index 1 (the edited note, as in the paper) >= 0.15 s and
    followed by another note (so its region is bounded)."""
    if st["is_drum"] or st["inst_class"] not in MELODIC: return []
    N = st["notes"]; out = []
    for i, (s0, e0, p0, _) in enumerate(N):
        if any(s < s0 - 0.01 and e > s0 + 0.03 for s, e, _, _ in N[max(0, i - 20):i]): continue
        w = [n for n in N[i:i + 12] if n[0] < s0 + win]
        on = [n for n in w if n[0] <= s0 + max_onset_span]
        if not (4 <= len(on) <= 6) or len(w) != len(on): continue
        if any(w[k][1] > w[k + 1][0] + 0.03 for k in range(len(w) - 1)): continue
        if any(n[1] - n[0] < 0.12 or not 48 <= n[2] <= 84 for n in w): continue
        if w[1][1] - w[1][0] < 0.15 or len({n[2] for n in w}) < 2: continue
        out.append(dict(t0=s0, notes=[(n[0] - s0, n[1] - s0, n[2]) for n in w]))
    return out

def chord_windows(st, win=2.0):
    """Isolated root-position triads: all notes start within 50 ms of t0, nothing else starts before t0+win,
    nothing sounds into t0, chord notes last >= 1.0 s; pitch classes relative to the lowest note form
    major / minor / sus2 / sus4."""
    if st["is_drum"] or st["inst_class"] not in CHORDAL: return []
    N = st["notes"]; out = []; starts = np.array([n[0] for n in N]) if N else np.array([])
    i = 0
    while i < len(N):
        t0 = N[i][0]; grp = [n for n in N[i:i + 16] if n[0] <= t0 + 0.05]
        j = i + len(grp)
        nxt = N[j][0] if j < len(N) else 1e9
        if nxt >= t0 + win and len(grp) >= 3 and min(n[1] for n in grp) - t0 >= 1.0 \
                and not any(s < t0 - 0.01 and e > t0 + 0.03 for s, e, _, _ in N[max(0, i - 30):i]):
            bass = min(n[2] for n in grp); pcs = tuple(sorted({(n[2] - bass) % 12 for n in grp}))
            if pcs in QUAL and 36 <= bass <= 72:
                out.append(dict(t0=t0, root_pc=bass % 12, bass=bass, quality=QUAL[pcs], pitches=sorted(n[2] for n in grp)))
        i = j
    return out

def harmony_triples(st):
    """(X1, X2, Y): two occurrences of one chord (same root pitch class + quality) and one chord on the same
    root of the other quality at 100 cents (major<->minor) or 200 cents (major->sus2, minor->sus4)."""
    by = defaultdict(list)
    for c in chord_windows(st): by[(c["root_pc"], c["quality"])].append(c)
    out = []
    pairs = {("major", "minor"): 100, ("minor", "major"): 100, ("major", "sus2"): 200, ("minor", "sus4"): 200}
    for (r, q), occ in by.items():
        occ = [o for k, o in enumerate(occ) if k == 0 or o["t0"] - occ[k - 1]["t0"] >= 2.0]
        if len(occ) < 2: continue
        for (q1, q2), lvl in pairs.items():
            if q1 == q and by.get((r, q2)):
                out.append(dict(level=lvl, X1=occ[0], X2=occ[1], Y=by[(r, q2)][0]))
    return out

def timbre_windows(st, win=2.5):
    fam = TIMBRE_FAMILY.get(st["program"]) if not st["is_drum"] else None
    if fam is None: return [], None
    N = st["notes"]; out = []; last = -1e9
    for s0, e0, p, _ in N:
        if s0 - last < win + 0.5: continue
        w = [n for n in N if s0 <= n[0] < s0 + win]
        if len(w) >= 3 and all(48 <= n[2] <= 96 for n in w) and w[-1][1] - s0 >= 1.5:
            out.append(dict(t0=s0)); last = s0
    return out, fam

def rhythm_windows(st, src=6.0):
    """Drum windows of 6.0 s (4.0 s x the largest ratio 1.5) with onsets in every second."""
    if not st["is_drum"]: return []
    on = np.array(sorted({round(n[0], 3) for n in st["notes"]})); out = []; last = -1e9
    for t0 in on:
        if t0 - last < src: continue
        w = on[(on >= t0) & (on < t0 + src)]
        if len(w) >= 12 and all(((w >= t0 + k) & (w < t0 + k + 1)).sum() >= 1 for k in range(int(src))):
            out.append(dict(t0=float(t0))); last = t0
    return out

if __name__ == "__main__":
    S = load(); cnt = defaultdict(int)
    for st in S:
        sp = "test" if st["split"] == "test" else "train"
        cnt[(sp, "melody")] += len(melody_windows(st))
        cnt[(sp, "harmony_triples")] += len(harmony_triples(st))
        for t in harmony_triples(st): cnt[(sp, f"harmony_{t['level']}")] += 1
        tw, fam = timbre_windows(st)
        if fam: cnt[(sp, f"timbre_{fam}")] += len(tw); cnt[(sp, f"timbre_{fam}_stems2plus")] += len(tw) >= 2
        cnt[(sp, "rhythm")] += len(rhythm_windows(st))
    for k in sorted(cnt): print(k, cnt[k])
