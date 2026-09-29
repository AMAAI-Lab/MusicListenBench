"""Step 3c: surface-feature leak audit.

For each task row, extracts cheap surface features from every rendered clip
and fits a trivial classifier (logistic regression AND a single decision
stump) to predict the label from those features alone. Asserts
AUC <= leak_auc_max (base.yaml, [P] 0.55) for every task; a violation fails
generation. Per-task exemptions let the legitimate
target dimension through; duration and file size are NEVER exempted.

Runs on train AND test splits (both are fatal if they leak).

Usage:
    python -m musiclistenbench.generator.data.audit_leakage --manifest audio_v2/manifest.jsonl
"""

import argparse
import collections
import sys

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None

from musiclistenbench.generator import config as config_mod
from musiclistenbench.generator.data import manifest as manifest_mod

FEATURE_NAMES = ["duration_samples", "file_size_bytes", "integrated_loudness", "rms", "peak",
                  "crest_factor", "zero_crossing_rate", "spectral_centroid", "onset_count", "total_energy"]

EXEMPT = {
    # One exemption only: melody, harmony and timbre have none.
    "Q4_rhythm": {"onset_count"},   # at fixed duration, onsets-per-window IS the rate
}
NEVER_EXEMPT = {"duration_samples", "file_size_bytes"}


def _clip_features(path):
    import os
    y, sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1).astype(np.float64)
    size = os.path.getsize(path)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    peak = float(np.abs(y).max())
    crest = peak / rms if rms > 0 else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(y))) > 0))
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) if len(y) > 1 else np.array([0.0])
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr) if len(y) > 1 else np.array([0.0])
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
    energy = float(np.sum(y ** 2))
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        loudness = loudness if np.isfinite(loudness) else -70.0
    except Exception:
        loudness = -70.0
    frame = max(1, int(0.005 * sr))
    n_frames = len(y) // frame
    onset_count = 0
    if n_frames > 1:
        env = np.array([np.sqrt(np.mean(y[i * frame:(i + 1) * frame] ** 2)) for i in range(n_frames)])
        thresh = 0.3 * env.max() if env.max() > 0 else 0
        above = env > thresh
        onset_count = int(np.sum(above[1:] & ~above[:-1]))
    return {
        "duration_samples": float(len(y)), "file_size_bytes": float(size), "integrated_loudness": float(loudness),
        "rms": rms, "peak": peak, "crest_factor": crest, "zero_crossing_rate": zcr,
        "spectral_centroid": centroid, "onset_count": float(onset_count), "total_energy": energy,
    }


def _feature_vector(trial, feature_cache, exempt_keys):
    feats_per_clip = [feature_cache[p] for p in trial["clip_paths"]]
    keys = [k for k in FEATURE_NAMES if k not in exempt_keys]
    if len(feats_per_clip) == 1:
        return {k: feats_per_clip[0][k] for k in keys}
    a, b = feats_per_clip
    out = {}
    for k in keys:
        out[f"{k}_diff"] = b[k] - a[k]
        out[f"{k}_absdiff"] = abs(b[k] - a[k])
    return out


def _auc_for(task, trials, auc_max):
    exempt = EXEMPT.get(task, set()) - NEVER_EXEMPT
    paths = sorted({p for t in trials for p in t["clip_paths"]})
    feature_cache = {p: _clip_features(p) for p in paths}

    rows = [_feature_vector(t, feature_cache, exempt) for t in trials]
    keys = sorted(rows[0].keys())
    X = np.array([[r[k] for k in keys] for r in rows])
    labels = sorted({t["answer"] for t in trials})
    if len(labels) != 2:
        return None, {}
    y = np.array([0 if t["answer"] == labels[0] else 1 for t in trials])

    if len(y) < 20 or len(set(y.tolist())) < 2:
        return None, {"note": "too few samples for a meaningful leak check"}

    X = np.nan_to_num(X)
    n_splits = min(5, min(collections.Counter(y).values()))
    if n_splits < 2:
        return None, {"note": "too few samples per class for CV"}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    best_auc = 0.0
    importances = {}
    for clf_name, clf in [("logreg", LogisticRegression(max_iter=1000)),
                          ("stump", DecisionTreeClassifier(max_depth=1, random_state=0))]:
        try:
            proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, proba)
        except Exception as e:
            continue
        best_auc = max(best_auc, auc)
        if clf_name == "logreg":
            clf.fit(X, y)
            importances = dict(zip(keys, np.abs(clf.coef_[0]).tolist()))

    return best_auc, importances


def run(manifest_path, auc_max):
    _, trials = manifest_mod.read_manifest(manifest_path)
    non_catch = [t for t in trials if not t.get("is_catch")]

    ok = True
    report = {}
    for split in ("train", "test"):
        by_task = collections.defaultdict(list)
        for t in non_catch:
            if t["split"] == split:
                by_task[t["task"]].append(t)
        for task, task_trials in sorted(by_task.items()):
            auc, importances = _auc_for(task, task_trials, auc_max)
            key = f"{task}:{split}"
            if auc is None:
                print(f"[leak:SKIP] {key} ({importances.get('note', 'n/a')})")
                report[key] = {"skipped": True}
                continue
            status = "OK" if auc <= auc_max else "FAIL"
            top = sorted(importances.items(), key=lambda kv: -kv[1])[:3]
            print(f"[leak:{status}] {key}: AUC={auc:.3f} (max {auc_max}); top features: {top}")
            report[key] = {"auc": auc, "passed": auc <= auc_max, "top_features": top}
            if auc > auc_max:
                ok = False
    return ok, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    cfg = config_mod.load_config()

    ok, _ = run(args.manifest, cfg.leak_auc_max)
    print()
    if ok:
        print("LEAK AUDIT PASSED")
    else:
        print("LEAK AUDIT FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
