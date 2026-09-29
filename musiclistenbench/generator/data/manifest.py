"""Manifest schema, read/write, and validation. Every task is a
two-clip trial and every clip is normalised individually.

Pure functions over dicts and file paths -- no model calls, so the module is
unit-testable without audio or GPUs (the audio existence/format checks in
`validate` are the one exception).
"""

import collections
import json
import os

import soundfile as sf

FIELDS = [
    "trial_id", "item_id", "task", "answer", "expected_role",
    "factor_changed", "magnitude", "magnitude_unit", "clip_paths",
    "duration_samples", "onset_times_s", "synth_params",
    "transform_params", "content_key", "family",
    "split", "is_catch", "certified", "seed",
]

EXPECTED_ROLES = {"base", "invariance", "equivariance"}
TASK_IDS = {"Q1_melody", "Q2_harmony", "Q3_timbre", "Q4_rhythm"}


def new_trial(**kwargs):
    """Build one manifest record, defaulting every field not passed."""
    record = {
        "onset_times_s": None,
        "synth_params": None,
        "transform_params": None,
        "family": None,
        "is_catch": False,
        "certified": None,
    }
    record.update(kwargs)
    missing = [f for f in FIELDS if f not in record]
    if missing:
        raise ValueError(f"new_trial missing required fields: {missing}")
    if record["expected_role"] not in EXPECTED_ROLES:
        raise ValueError(f"bad expected_role: {record['expected_role']}")
    if record["task"] not in TASK_IDS:
        raise ValueError(f"bad task: {record['task']} (must be one of {sorted(TASK_IDS)})")
    return {k: record[k] for k in FIELDS}


def make_header(commit_hash, seed, config_hashes, timestamp):
    return {
        "header": True,
        "generator_commit": commit_hash,
        "seed": seed,
        "config_hashes": config_hashes,
        "generated_at": timestamp,
    }


def write_manifest(path, header, trials, mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        if mode == "w":
            f.write(json.dumps(header) + "\n")
        for t in trials:
            f.write(json.dumps(t) + "\n")


def read_manifest(path):
    """Returns (header, trials). `path` may be a single file or a list of
    shard files sharing one logical manifest (only the first file's header
    is returned)."""
    paths = [path] if isinstance(path, str) else list(path)
    header = None
    trials = []
    for p in paths:
        with open(p) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        if not lines:
            continue
        first = lines[0]
        if first.get("header"):
            if header is None:
                header = first
            lines = lines[1:]
        trials.extend(lines)
    return header, trials


class Violation(collections.namedtuple("Violation", ["rule", "item_id", "trial_id", "message"])):
    def __str__(self):
        return f"[{self.rule}] item={self.item_id} trial={self.trial_id}: {self.message}"


def _by_item(trials):
    by_item = collections.defaultdict(list)
    for t in trials:
        by_item[t["item_id"]].append(t)
    return by_item


def _ladder_values(cfg, task, magnitude_unit):
    """None means 'no check' -- the task ID is unknown (a schema violation
    caught separately by rule 7), the row has no declared ladder, or
    `magnitude_unit` doesn't match it, meaning this magnitude belongs to an
    extra invariance axis. Unit alone can't disambiguate rows 1-2's
    transposition from their own content ladder (both are 'cents') -- that
    case is filtered by the caller checking `factor_changed` instead."""
    if task not in cfg.tasks:
        return None
    ladder = cfg.tasks[task].get("ladder")
    if ladder is None or ladder.get("unit") != magnitude_unit:
        return None
    return set(ladder["values"])


def validate(path, cfg=None, check_audio=True, max_audio_checks=None):
    """Returns a list of Violation. Empty list == manifest is clean.

    `cfg` is a musiclistenbench.generator.config.Config; if omitted, rules 6 and 9 (which
    need tasks.yaml/splits.yaml/transforms.yaml) are skipped and a note is
    appended as a single Violation so callers can't silently under-check.
    """
    header, trials = read_manifest(path)
    violations = []
    by_item = _by_item(trials)

    # Rule 7: task is one of the four IDs.
    for t in trials:
        if t["task"] not in TASK_IDS:
            violations.append(Violation("rule7", t["item_id"], t["trial_id"],
                                         f"task {t['task']} is not one of {sorted(TASK_IDS)}"))

    # Rule 1 + 2 + 3: per-item role/content_key/answer invariants.
    for item_id, rows in by_item.items():
        roles = collections.Counter(r["expected_role"] for r in rows)
        if roles.get("base", 0) != 1:
            violations.append(Violation("rule1", item_id, None,
                                         f"expected exactly 1 base row, found {roles.get('base', 0)}"))
        if roles.get("invariance", 0) < 1:
            violations.append(Violation("rule1", item_id, None, "no invariance row"))
        if roles.get("equivariance", 0) < 1:
            violations.append(Violation("rule1", item_id, None, "no equivariance row"))

        base_rows = [r for r in rows if r["expected_role"] == "base"]
        if not base_rows:
            continue
        base = base_rows[0]
        for r in rows:
            if r["expected_role"] == "equivariance":
                if r["content_key"] == base["content_key"]:
                    violations.append(Violation("rule2", item_id, r["trial_id"],
                                                 "equivariance row shares content_key with base"))
                if r["answer"] == base["answer"]:
                    violations.append(Violation("rule3", item_id, r["trial_id"],
                                                 "equivariance row answer did not change from base"))
            else:
                if r["content_key"] != base["content_key"]:
                    violations.append(Violation("rule2", item_id, r["trial_id"],
                                                 f"{r['expected_role']} row content_key differs from base"))
                if r["task"] != base["task"]:
                    violations.append(Violation("rule2", item_id, r["trial_id"], "task differs from base"))
                if r["expected_role"] == "invariance" and r["answer"] != base["answer"]:
                    violations.append(Violation("rule3", item_id, r["trial_id"],
                                                 "invariance row answer moved from base"))
            # Rule 8: transposition rows are always invariance, never equivariance.
            if r["factor_changed"] == "transposition" and r["expected_role"] != "invariance":
                violations.append(Violation("rule8", item_id, r["trial_id"],
                                             f"transposition row has expected_role={r['expected_role']}, "
                                             f"must be 'invariance'"))

    # Rule 4 / 5: audio existence, format, 2-clip trials, duration.
    if check_audio:
        checked = 0
        row_durations = collections.defaultdict(set)
        for t in trials:
            if max_audio_checks is not None and checked >= max_audio_checks:
                break
            checked += 1
            if len(t["clip_paths"]) != 2:
                violations.append(Violation("rule4", t["item_id"], t["trial_id"],
                                             f"expected 2 clip_paths, found {len(t['clip_paths'])}"))
            durations_this_trial = set()
            sizes_this_trial = []
            for p in t["clip_paths"]:
                if not os.path.isfile(p):
                    violations.append(Violation("rule4", t["item_id"], t["trial_id"], f"missing file: {p}"))
                    continue
                info = sf.info(p)
                if info.samplerate != 44100 or info.channels != 1 or info.subtype != "PCM_16":
                    violations.append(Violation("rule4", t["item_id"], t["trial_id"],
                                                 f"{p} is not 44.1kHz mono 16-bit PCM ({info.samplerate}Hz, "
                                                 f"{info.channels}ch, {info.subtype})"))
                if info.frames / info.samplerate >= 10.0:
                    violations.append(Violation("rule4", t["item_id"], t["trial_id"], f"{p} is >= 10s"))
                durations_this_trial.add(info.frames)
                sizes_this_trial.append(os.path.getsize(p))
            if len(durations_this_trial) > 1:
                violations.append(Violation("rule5", t["item_id"], t["trial_id"],
                                             f"clips within trial have mismatched durations: {durations_this_trial}"))
            elif durations_this_trial:
                measured = next(iter(durations_this_trial))
                if measured != t["duration_samples"]:
                    violations.append(Violation("rule5", t["item_id"], t["trial_id"],
                                                 f"manifest duration_samples={t['duration_samples']} but decoded "
                                                 f"file has {measured}"))
                row_durations[t["task"]].add(measured)
            if len(sizes_this_trial) == 2:
                tol = cfg.tolerances["filesize_bytes"] if cfg else 2048
                if abs(sizes_this_trial[0] - sizes_this_trial[1]) > tol:
                    violations.append(Violation("rule5", t["item_id"], t["trial_id"],
                                                 f"clip file sizes differ by more than {tol} bytes"))
        for task, durations in row_durations.items():
            if len(durations) > 1:
                violations.append(Violation("rule5", None, None,
                                             f"task {task} has inconsistent duration_samples across trials: "
                                             f"{durations}"))

    # Rule 6: magnitude on the declared ladder. Transposition rows carry a
    # continuous 0-200c magnitude that is never on the row's own content
    # ladder even when the units happen to match (both "cents" for rows
    # 1-2) -- skip them here; rule 8 checks their role separately.
    if cfg is not None:
        for t in trials:
            if t["magnitude"] is None or t["factor_changed"] == "transposition":
                continue
            values = _ladder_values(cfg, t["task"], t["magnitude_unit"])
            if values is not None and t["magnitude"] not in values:
                violations.append(Violation("rule6", t["item_id"], t["trial_id"],
                                             f"magnitude {t['magnitude']} not on {t['task']}'s declared ladder"))
    else:
        violations.append(Violation("rule6", None, None, "skipped: no cfg passed to validate()"))

    # Rule 9: split/content/family contamination (defers to audit_contamination.py
    # for the full check incl. audio fingerprints; this is the cheap manifest-only pass).
    if cfg is not None:
        train_keys = {t["content_key"] for t in trials if t["split"] == "train"}
        test_keys = {t["content_key"] for t in trials if t["split"] == "test"}
        overlap = train_keys & test_keys
        if overlap:
            violations.append(Violation("rule9", None, None,
                                         f"{len(overlap)} content_key(s) appear in both train and test"))
        test_family_set = set(cfg.transforms["split"]["test_families"])
        for t in trials:
            if t["split"] == "train" and t["family"] in test_family_set:
                override = cfg.transforms.get("row_overrides", {}).get(t["task"])
                allowed = set(override["train_families"]) if override else set(cfg.transforms["split"]["train_families"])
                if t["family"] not in allowed:
                    violations.append(Violation("rule9", t["item_id"], t["trial_id"],
                                                 f"train row uses test-only family {t['family']}"))
    else:
        violations.append(Violation("rule9", None, None, "skipped: no cfg passed to validate()"))

    return violations
