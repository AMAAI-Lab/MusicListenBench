"""Entry point of the generator: emits the audio and manifest.jsonl.

The four tasks are Q1_melody, Q2_harmony, Q3_timbre and Q4_rhythm. The size of each is
stated per task in configs/tasks.yaml (budget.test_total / budget.train_total = 250 / 2,500)
and distributed across the task's difficulty levels (or, for timbre, its instrument-pair
cells, weighted for class balance) by `_distribute` below.

Usage:
    python -m musiclistenbench.generator.data.make_render_sets --output-root generated
    # a small sample for a quick check (one item per cell):
    python -m musiclistenbench.generator.data.make_render_sets --output-root /tmp/gen_sample --items-per-cell-cap 1
"""

import argparse
import datetime
import hashlib
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from musiclistenbench.generator import config as config_mod
from musiclistenbench.generator.data import manifest, render, symbolic

BUILDERS = {
    "Q1_melody": lambda rng, cfg, split, rung: symbolic.build_Q1_melody(rng, cfg, split, rung),
    "Q2_harmony": lambda rng, cfg, split, rung: symbolic.build_Q2_harmony(rng, cfg, split, rung),
    "Q3_timbre": lambda rng, cfg, split, rung: symbolic.build_Q3_timbre(rng, cfg, split, rung),
    "Q4_rhythm": lambda rng, cfg, split, rung: symbolic.build_Q4_rhythm(rng, cfg, rung),
}

CATCH_BUILDERS = {
    "Q1_melody": symbolic.build_Q1_catch,
    "Q2_harmony": symbolic.build_Q2_catch,
    "Q3_timbre": symbolic.build_Q3_catch,
}

TRANSPOSITION_ROWS = {"Q1_melody", "Q2_harmony"}


def _rung_cells(cfg, task):
    """List of (rung_value, rung_tag, weight). `weight` matters only for
    Q3_timbre: `combinations_with_replacement` over 4 programs yields 4
    'same' pairs and 6 'different' pairs, so same-pairs get weight 2 to
    keep the row class-balanced -- this is the exact bug
    audit_balance.py caught in the old 9-row grid's A4_timbre."""
    spec = cfg.tasks[task]
    if task == "Q3_timbre":
        programs = spec["params"]["gm_programs"]
        pairs = list(itertools.combinations_with_replacement(programs, 2))
        return [((a, b), f"p{a}-{b}", 2 if a == b else 1) for a, b in pairs]
    ladder = spec.get("ladder")
    if ladder is None:
        return [(None, "none", 1)]
    return [(v, str(v).replace(".", "p"), 1) for v in ladder["values"]]


def _distribute(total, weights):
    """Largest-remainder proportional split of `total` across cells by
    `weights`, summing exactly to `total`."""
    s = sum(weights)
    raw = [total * w / s for w in weights]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: -(raw[i] - floors[i]))
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def _item_seed(master_seed, task, split, rung_tag, idx):
    payload = f"{master_seed}:{task}:{split}:{rung_tag}:{idx}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _decorrelation_factors(task, item):
    """Non-target factors that audit_balance.py checks for near-zero
    correlation with the base answer."""
    p = item.params
    if task == "Q1_melody":
        return {"motif_note0": p["motif_first"][0]}
    if task == "Q2_harmony":
        return {"root_midi": p["root_midi"], "quality_first": 1 if p["quality_first"] == "major" else 0}
    if task == "Q3_timbre":
        return {"melody_note0": p["melody"][0], "program_first": p["program_pair_base"][0]}
    if task == "Q4_rhythm":
        return {"phase_ref": p["phase_ref"], "phase_fast": p["phase_fast"]}
    return {}


def _clip_paths(audio_dir, item_id, trial_tag):
    d = os.path.join(audio_dir, item_id)
    return [os.path.join(d, f"{trial_tag}_A.wav"), os.path.join(d, f"{trial_tag}_B.wav")]


def _write_all(paths, clips, sr):
    for p, y in zip(paths, clips):
        render.write_clip(y, sr, p)


def _onset_times(task, notes_first, notes_second):
    return [n["start"] for n in notes_first] + [n["start"] for n in notes_second]


def build_render_set(task, split, rung_value, idx, cfg, master_seed, verify_errors):
    spec = cfg.tasks[task]
    labels = spec["labels"]
    content_dial = spec["content_dial"]
    window = spec["window_seconds"]
    sr = cfg.sample_rate
    n_samples = round(window * sr)

    is_catch_item = rung_value == "__catch__"
    tag_source = "catch" if is_catch_item else rung_value
    rung_tag = str(tag_source).replace(" ", "").replace("(", "").replace(")", "").replace(",", "-")
    item_id = f"{task}__{split}__{rung_tag}__{idx:06d}"

    seed = _item_seed(master_seed, task, split, rung_tag, idx)
    rng = np.random.default_rng(seed)

    if is_catch_item:
        item = CATCH_BUILDERS[task](rng, cfg, split)
    else:
        item = BUILDERS[task](rng, cfg, split, rung_value)

    pool = cfg.soundfonts_for(split)
    pool_keys = list(pool.keys())
    home_key = rng.choice(pool_keys)
    home_path = pool[home_key]

    trials = []

    def make_pole_clips(the_item, pole, sf2_path, program=0):
        notes_first, notes_second = render.build_notes(task, the_item, pole, program)
        y_first = render.synth_notes(notes_first, window, sf2_path, sr)
        y_second = render.synth_notes(notes_second, window, sf2_path, sr)
        return (y_first, y_second), notes_first, notes_second

    # --- base ---
    raw_base, notes_first_base, notes_second_base = make_pole_clips(item, "base", home_path)
    clips_base, _ = render.finalize_clips(raw_base, sr, cfg.target_lufs, n_samples)
    paths_base = _clip_paths(cfg.audio_dir, item_id, "base")
    _write_all(paths_base, clips_base, sr)

    trials.append(manifest.new_trial(
        trial_id=f"{item_id}__base", item_id=item_id, task=task,
        answer=item.gold_base, expected_role="base", factor_changed="none",
        magnitude=item.ladder_value, magnitude_unit=item.ladder_unit,
        clip_paths=paths_base, duration_samples=n_samples,
        onset_times_s=_onset_times(task, notes_first_base, notes_second_base),
        content_key=item.content_key, split=split, is_catch=is_catch_item, seed=int(seed),
        synth_params={"home_soundfont": home_key, **_decorrelation_factors(task, item)},
    ))

    # --- production re-renders (one per family in this item's split) ---
    families = cfg.families_for(task, split)
    for fam_idx, family in enumerate(families):
        fam_seed = seed + 1000 * (fam_idx + 1)
        fam_rng = np.random.default_rng(fam_seed)
        fam_cfg = cfg.transforms["families"][family]

        if family == "instrument":
            alt_key = rng.choice([k for k in pool_keys if k != home_key])
            raw, _, _ = make_pole_clips(item, "base", pool[alt_key])
            clips, _ = render.finalize_clips(raw, sr, cfg.target_lufs, n_samples)
            transform_params = {"soundfont": alt_key}
        elif family == "music_resample":
            new_melody = [int(fam_rng.choice(spec["params"]["reference_midi_pool"][split]))
                          for _ in range(spec["params"]["n_notes"])]
            variant = symbolic.SymbolicItem(item.item_id, item.task, item.gold_base, item.gold_content_edit,
                                             item.content_key, item.ladder_value, item.ladder_unit,
                                             {**item.params, "melody": new_melody})
            raw, _, _ = make_pole_clips(variant, "base", home_path)
            clips, _ = render.finalize_clips(raw, sr, cfg.target_lufs, n_samples)
            transform_params = {"resampled_melody": new_melody}
        else:
            transformed, transform_params = render.apply_family_to_pair(
                list(raw_base), sr, family, fam_cfg, fam_rng, seed=int(fam_seed))
            clips, _ = render.finalize_clips(transformed, sr, cfg.target_lufs, n_samples)

        paths = _clip_paths(cfg.audio_dir, item_id, f"inv_{family}")
        _write_all(paths, clips, sr)
        trials.append(manifest.new_trial(
            trial_id=f"{item_id}__inv_{family}", item_id=item_id, task=task,
            answer=item.gold_base, expected_role="invariance", factor_changed=family,
            magnitude=item.ladder_value, magnitude_unit=item.ladder_unit,
            clip_paths=paths, duration_samples=n_samples, onset_times_s=None,
            transform_params=transform_params, content_key=item.content_key, family=family,
            split=split, is_catch=is_catch_item, seed=int(fam_seed),
        ))

    # --- transposition invariance (rows 1-2 only, regardless of split) ---
    if task in TRANSPOSITION_ROWS:
        t_seed = seed + 999999
        raw_t = render.synth_transposed_invariance(task, item, window, home_path, sr)
        clips_t, _ = render.finalize_clips(raw_t, sr, cfg.target_lufs, n_samples)
        paths_t = _clip_paths(cfg.audio_dir, item_id, "inv_transposition")
        _write_all(paths_t, clips_t, sr)
        trials.append(manifest.new_trial(
            trial_id=f"{item_id}__inv_transposition", item_id=item_id, task=task,
            answer=item.gold_base, expected_role="invariance", factor_changed="transposition",
            magnitude=item.params["transposition_cents"], magnitude_unit="cents",
            clip_paths=paths_t, duration_samples=n_samples, onset_times_s=None,
            content_key=item.content_key, family=None, split=split,
            is_catch=False,   # never bit-identical by construction -- see render.py docstring
            seed=int(t_seed),
        ))

    # --- content edit (equivariance) ---
    raw_edit, notes_first_edit, notes_second_edit = make_pole_clips(item, "content_edit", home_path)
    clips_edit, _ = render.finalize_clips(raw_edit, sr, cfg.target_lufs, n_samples)
    paths_edit = _clip_paths(cfg.audio_dir, item_id, "eq")
    _write_all(paths_edit, clips_edit, sr)
    content_key_edit = symbolic.content_hash({"base_content_key": item.content_key, "pole": "edit",
                                               "gold": item.gold_content_edit})
    trials.append(manifest.new_trial(
        trial_id=f"{item_id}__eq", item_id=item_id, task=task,
        answer=item.gold_content_edit, expected_role="equivariance", factor_changed=content_dial,
        magnitude=item.ladder_value, magnitude_unit=item.ladder_unit,
        clip_paths=paths_edit, duration_samples=n_samples,
        onset_times_s=_onset_times(task, notes_first_edit, notes_second_edit),
        content_key=content_key_edit, split=split, is_catch=False, seed=int(seed) + 1,
    ))

    _verify_realisation(task, item, clips_base, sr, cfg, verify_errors, item_id, home_path)
    return trials


def _verify_realisation(task, item, clips_base, sr, cfg, verify_errors, item_id, sf2_path):
    """Q1/Q2 pitch checks render short SOLO tones purely for measurement --
    pyin (monophonic) cannot isolate one note out of a 4-note melody phrase
    or one voice out of a 3-note simultaneous chord, so analysing the full
    clips_base directly (as a first attempt did) produced nonsense readings
    (e.g. -2590c) that were a verification-tooling bug, not a generation bug."""
    tol = cfg.tolerances
    if task == "Q1_melody":
        p = item.params
        idx = p["edited_note_index"]
        dur = p["note_dur_seconds"]
        note1, note2 = p["motif_first"][idx], p["motif_second_base"][idx]
        y1 = render.synth_notes([dict(midi=note1, start=0.0, dur=dur)], dur, sf2_path, sr)
        y2 = render.synth_notes([dict(midi=note2, start=0.0, dur=dur)], dur, sf2_path, sr)
        expected_offset = (note2 - note1) * 100.0
        measured = render.measure_interval_cents(y1, y2, sr)
        if measured is not None and abs(measured - expected_offset) > tol["pitch_cents"]:
            verify_errors.append(f"{item_id}: edited-note interval {measured:.1f}c vs intended "
                                  f"{expected_offset:.1f}c (tolerance {tol['pitch_cents']}c)")
    if task == "Q2_harmony":
        p = item.params
        root, dur = p["root_midi"], p["chord_dur_seconds"]
        third1 = root + round(p["third_first_cents"] / 100.0)
        third2 = root + round(p["third_second_base_cents"] / 100.0)
        y1 = render.synth_notes([dict(midi=third1, start=0.0, dur=dur)], dur, sf2_path, sr)
        y2 = render.synth_notes([dict(midi=third2, start=0.0, dur=dur)], dur, sf2_path, sr)
        expected_offset = p["third_second_base_cents"] - p["third_first_cents"]
        measured = render.measure_interval_cents(y1, y2, sr)
        if measured is not None and abs(measured - expected_offset) > tol["pitch_cents"]:
            verify_errors.append(f"{item_id}: third-note interval {measured:.1f}c vs intended "
                                  f"{expected_offset:.1f}c (tolerance {tol['pitch_cents']}c)")
    if task == "Q4_rhythm" and len(clips_base) == 2:
        expected_rate = item.params["base_rate_hz"] * (item.params["ratio"] if item.gold_base == "first" else 1.0)
        other_rate = item.params["base_rate_hz"] * (1.0 if item.gold_base == "first" else item.params["ratio"])
        measured0 = render.measure_click_rate_hz(clips_base[0], sr)
        measured1 = render.measure_click_rate_hz(clips_base[1], sr)
        for measured, expected in ((measured0, expected_rate), (measured1, other_rate)):
            if measured is not None:
                ratio_err = abs(measured - expected) / expected
                if ratio_err > tol["tempo_ratio"]:
                    verify_errors.append(f"{item_id}: click rate {measured:.3f}Hz vs expected {expected:.3f}Hz "
                                          f"(err {ratio_err:.3f} > {tol['tempo_ratio']})")


def generate(cfg, tasks=None, items_per_cell_cap=None, fail_on_verify_error=True):
    active = list(cfg.tasks.keys())
    tasks = tasks or active
    all_trials = []
    verify_errors = []
    stats = {"items": 0, "trials": 0}

    for task in tasks:
        if task not in active:
            continue
        spec = cfg.tasks[task]
        cells = _rung_cells(cfg, task)
        weights = [w for _, _, w in cells]

        for split in ("train", "test"):
            total = spec["budget"][f"{split}_total"]
            counts = _distribute(total, weights)
            for (rung_value, rung_tag, _weight), n_items in zip(cells, counts):
                if items_per_cell_cap is not None:
                    n_items = min(n_items, items_per_cell_cap)
                for idx in range(n_items):
                    trials = build_render_set(task, split, rung_value, idx, cfg, cfg.seed, verify_errors)
                    all_trials.extend(trials)
                    stats["items"] += 1
                    stats["trials"] += len(trials)

            if spec.get("catch_trials"):
                n_catch = max(1, round(total * spec["catch_trial_fraction"]))
                if items_per_cell_cap is not None:
                    n_catch = min(n_catch, items_per_cell_cap)
                for idx in range(n_catch):
                    trials = build_render_set(task, split, "__catch__", idx, cfg, cfg.seed, verify_errors)
                    all_trials.extend(trials)
                    stats["items"] += 1
                    stats["trials"] += len(trials)

    if verify_errors and fail_on_verify_error:
        raise SystemExit("Realisation verification FAILED:\n" + "\n".join(verify_errors))
    return all_trials, stats, verify_errors


def _pseudo_commit_hash():
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=config_mod.PROJECT_ROOT)
        if out.returncode == 0:
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha256()
    for root, _, files in os.walk(pkg_dir):
        for fn in sorted(files):
            if fn.endswith((".py", ".yaml")):
                with open(os.path.join(root, fn), "rb") as f:
                    h.update(f.read())
    return f"no-git:{h.hexdigest()[:16]}"


def _hash_file(name):
    path = os.path.join(config_mod.CONFIG_DIR, name)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--items-per-cell-cap", type=int, default=None,
                     help="cap items per (task, rung, split) cell -- for smoke tests / the throughput probe")
    ap.add_argument("--no-fail-on-verify-error", action="store_true")
    args = ap.parse_args()

    cfg = config_mod.load_config(output_root=args.output_root)
    t0 = time.time()
    trials, stats, verify_errors = generate(cfg, tasks=args.tasks, items_per_cell_cap=args.items_per_cell_cap,
                                             fail_on_verify_error=not args.no_fail_on_verify_error)
    elapsed = time.time() - t0

    header = manifest.make_header(
        commit_hash=_pseudo_commit_hash(), seed=cfg.seed,
        config_hashes={"base": _hash_file("base.yaml"), "tasks": _hash_file("tasks.yaml"),
                        "transforms": _hash_file("transforms.yaml"), "splits": _hash_file("splits.yaml")},
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )
    manifest.write_manifest(cfg.manifest_path, header, trials, mode="w")
    print(f"Wrote {stats['items']} render sets ({stats['trials']} trials) -> {cfg.manifest_path}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / max(1, stats['items']):.3f}s/item)")
    if verify_errors:
        print(f"[warn] {len(verify_errors)} realisation-verification warnings:")
        for e in verify_errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
