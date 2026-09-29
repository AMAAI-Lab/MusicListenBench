"""Step 3b: train/test contamination checker. Frozen artifact:
written once, committed, and run before every training run; its JSON report
+ exit code are attached to every training run's artifacts.

Checks, each a hard failure:
  1. content_key intersection between train and test is empty.
  2. soundfont IDs, IR IDs (room rt60/dim/seed), and codec settings used in
     train and test are disjoint sets.
  3. family sets are disjoint and match transforms.yaml.
  4. audio-level near-duplicate check across the split boundary (file hash,
     plus a cheap decimated-waveform fingerprint).
  5. If splits.yaml's transfer_split is enabled: no Q1_melody row has
     split == "train". The option is disabled in the released configuration,
     so this check passes trivially.

Usage:
    python -m musiclistenbench.generator.data.audit_contamination --manifest audio_v2/manifest.jsonl
"""

import argparse
import hashlib
import json
import sys

import numpy as np
import soundfile as sf

from musiclistenbench.generator import config as config_mod
from musiclistenbench.generator.data import manifest as manifest_mod


def _fingerprint(path, n=2000):
    try:
        y, sr = sf.read(path, always_2d=True)
    except Exception:
        return None
    y = y.mean(axis=1)
    if len(y) == 0:
        return None
    idx = np.linspace(0, len(y) - 1, min(n, len(y))).astype(int)
    decimated = np.round(y[idx] * 1000).astype(int)
    return hashlib.sha256(decimated.tobytes()).hexdigest()


def _file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_content_key(trials):
    train_keys = {t["content_key"] for t in trials if t["split"] == "train"}
    test_keys = {t["content_key"] for t in trials if t["split"] == "test"}
    overlap = train_keys & test_keys
    return len(overlap) == 0, {"overlap_count": len(overlap), "examples": list(overlap)[:10]}


def check_soundfont_ir_codec(trials):
    def ids_for(split):
        soundfonts, irs, codecs = set(), set(), set()
        for t in trials:
            if t["split"] != split:
                continue
            sp = t.get("synth_params") or {}
            if "home_soundfont" in sp:
                soundfonts.add(sp["home_soundfont"])
            tp = t.get("transform_params") or {}
            if t.get("family") == "instrument" and "soundfont" in tp:
                soundfonts.add(tp["soundfont"])
            if t.get("family") == "room":
                irs.add((tp.get("rt60"), tuple(tp.get("room_dim", [])), tp.get("seed")))
            if t.get("family") == "codec":
                codecs.add((tp.get("codec"), tp.get("bitrate_kbps")))
        return soundfonts, irs, codecs

    sf_train, ir_train, codec_train = ids_for("train")
    sf_test, ir_test, codec_test = ids_for("test")
    sf_overlap = sf_train & sf_test
    ir_overlap = ir_train & ir_test
    codec_overlap = codec_train & codec_test
    ok = not (sf_overlap or ir_overlap or codec_overlap)
    return ok, {
        "soundfont_overlap": list(sf_overlap), "ir_overlap": [list(x) for x in ir_overlap],
        "codec_overlap": list(codec_overlap),
    }


def check_families(trials, transforms_cfg):
    """Expected sets are the UNION of the default split plus every row's
    row_overrides entry (e.g. Q3_timbre's music_resample substitute for
    instrument) -- not a static summary elsewhere, which would otherwise
    flag a legitimate per-row override as contamination (this exact bug
    happened 2026-08-18 with the old grid's A4_timbre/pitch_resample, fixed
    by reading transforms.yaml, the authoritative source, directly)."""
    train_families = {t["family"] for t in trials if t["split"] == "train" and t["family"]}
    test_families = {t["family"] for t in trials if t["split"] == "test" and t["family"]}

    expected_train = set(transforms_cfg["split"]["train_families"])
    expected_test = set(transforms_cfg["split"]["test_families"])
    for override in transforms_cfg.get("row_overrides", {}).values():
        expected_train |= set(override.get("train_families", []))
        expected_test |= set(override.get("test_families", []))

    ok = train_families.issubset(expected_train) and test_families.issubset(expected_test) \
        and not (train_families & test_families)
    return ok, {"train_families": sorted(train_families), "test_families": sorted(test_families),
                "expected_train": sorted(expected_train), "expected_test": sorted(expected_test)}


def check_melody_transfer_split(trials, splits_cfg):
    if not splits_cfg.get("transfer_split", False):
        return True, {"transfer_split_enabled": False, "note": "transfer_split disabled; check is a no-op"}
    bad = [t["trial_id"] for t in trials if t["task"] == "Q1_melody" and t["split"] == "train"]
    return len(bad) == 0, {"transfer_split_enabled": True, "offending_trial_ids": bad[:10], "count": len(bad)}


def check_audio_duplicates(trials, max_check=None):
    train_paths = {p for t in trials if t["split"] == "train" for p in t["clip_paths"]}
    test_paths = {p for t in trials if t["split"] == "test" for p in t["clip_paths"]}

    train_hashes, test_hashes = {}, {}
    for i, p in enumerate(sorted(train_paths)):
        if max_check and i >= max_check:
            break
        train_hashes[_file_hash(p)] = p
    for i, p in enumerate(sorted(test_paths)):
        if max_check and i >= max_check:
            break
        test_hashes[_file_hash(p)] = p
    hash_overlap = set(train_hashes) & set(test_hashes)

    train_fp = {_fingerprint(p) for p in list(train_paths)[:max_check or len(train_paths)]}
    test_fp = {_fingerprint(p) for p in list(test_paths)[:max_check or len(test_paths)]}
    train_fp.discard(None)
    test_fp.discard(None)
    fp_overlap = train_fp & test_fp

    ok = len(hash_overlap) == 0 and len(fp_overlap) == 0
    return ok, {"hash_overlap_count": len(hash_overlap), "fingerprint_overlap_count": len(fp_overlap)}


def run(manifest_path, max_audio_checks=None):
    cfg = config_mod.load_config()
    _, trials = manifest_mod.read_manifest(manifest_path)

    checks = {}
    checks["content_key_disjoint"] = check_content_key(trials)
    checks["soundfont_ir_codec_disjoint"] = check_soundfont_ir_codec(trials)
    checks["family_split_matches_frozen"] = check_families(trials, cfg.transforms)
    checks["melody_transfer_split"] = check_melody_transfer_split(trials, cfg.splits)
    checks["audio_duplicates"] = check_audio_duplicates(trials, max_check=max_audio_checks)

    report = {name: {"passed": ok, **detail} for name, (ok, detail) in checks.items()}
    overall_ok = all(v["passed"] for v in report.values())
    report["overall_passed"] = overall_ok
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--max-audio-checks", type=int, default=None)
    args = ap.parse_args()

    report = run(args.manifest, max_audio_checks=args.max_audio_checks)
    print(json.dumps(report, indent=2, default=str))
    if args.report_out:
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2, default=str)

    if not report["overall_passed"]:
        print("CONTAMINATION AUDIT FAILED", file=sys.stderr)
        sys.exit(1)
    print("CONTAMINATION AUDIT PASSED")


if __name__ == "__main__":
    main()
