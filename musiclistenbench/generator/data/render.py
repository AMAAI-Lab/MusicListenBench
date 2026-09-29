"""FluidSynth synthesis + the fixed production-transform chain.
Redesigned 2026-08-19 for the four-row grid (Q1_melody, Q2_harmony,
Q3_timbre, Q4_rhythm).

Pipeline order, fixed: symbolic -> synth/soundfont render -> room -> codec ->
EQ/hiss -> decode-and-trim to the exact intended sample count -> loudness
normalisation (per-clip everywhere now that C3_loudness is gone) -> write
WAV @ 44.1kHz/16-bit/mono.

`instrument`/`music_resample` (render-time families) re-run the synth step;
`room`/`codec`/`eq`/`hiss` transform the already-synthesized waveform;
`transposition` (rows 1-2 only) shifts the second clip's pitch, same
"perturb clip B, keep clip A fixed" pattern as every other nuisance.
"""

import os
import subprocess
import tempfile

import numpy as np
import pretty_midi
import soundfile as sf

from musiclistenbench.generator.data import transforms
from musiclistenbench.generator.data.symbolic import note_and_bend


def cents_to_bend(cents, max_cents=200):
    cents = max(-max_cents, min(max_cents, cents))
    return int(round(cents / max_cents * 8191))


def _notes_to_midi(notes, tempo=120):
    """notes: list of dict(midi, cents=0.0, start, dur, program=0, is_drum=False)."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    by_channel = {}
    for n in notes:
        key = (n.get("program", 0), n.get("is_drum", False))
        if key not in by_channel:
            inst = pretty_midi.Instrument(program=key[0], is_drum=key[1])
            by_channel[key] = inst
            pm.instruments.append(inst)
        inst = by_channel[key]
        final_midi, remainder_cents = note_and_bend(n["midi"], n.get("cents", 0.0))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=int(final_midi),
                                            start=n["start"], end=n["start"] + n["dur"]))
        if abs(remainder_cents) > 1e-6:
            bend = cents_to_bend(remainder_cents)
            inst.pitch_bends.append(pretty_midi.PitchBend(bend, n["start"]))
            inst.pitch_bends.append(pretty_midi.PitchBend(0, n["start"] + n["dur"]))
    return pm


def _fluidsynth_render(pm, sf2_path, sr):
    with tempfile.TemporaryDirectory() as tmp:
        midi_path = os.path.join(tmp, "clip.mid")
        wav_path = os.path.join(tmp, "clip.wav")
        pm.write(midi_path)
        subprocess.run(
            ["fluidsynth", "-ni", "-F", wav_path, "-r", str(sr), sf2_path, midi_path],
            check=True, capture_output=True,
        )
        y, file_sr = sf.read(wav_path, always_2d=True)
        y = y.mean(axis=1).astype(np.float32)
        if file_sr != sr:
            import librosa
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def synth_notes(notes, window_seconds, sf2_path, sr):
    """Render `notes` (possibly empty -> pure silence) and trim/pad to
    exactly round(window_seconds * sr) samples."""
    n_samples = round(window_seconds * sr)
    if not notes:
        y = np.zeros(n_samples, dtype=np.float32)
    else:
        pm = _notes_to_midi(notes)
        y = _fluidsynth_render(pm, sf2_path, sr)
    return transforms.trim_or_pad(y, n_samples)


def add_noise_floor(y, sr, level_dbfs, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=len(y)).astype(np.float32)
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-12
    target_rms = 10 ** (level_dbfs / 20.0)
    return y + noise * (target_rms / noise_rms)


# --- Per-row note-list builders: (item, pole) -> (notes_first, notes_second) ---

def _gold(item, pole):
    return item.gold_base if pole == "base" else item.gold_content_edit


def notes_Q1(item, pole, program=0):
    p = item.params
    motif2 = p["motif_second_base"] if pole == "base" else p["motif_second_edit"]
    dur = p["note_dur_seconds"]
    m1 = [dict(midi=m, start=i * dur, dur=dur, program=program) for i, m in enumerate(p["motif_first"])]
    m2 = [dict(midi=m, start=i * dur, dur=dur, program=program) for i, m in enumerate(motif2)]
    return m1, m2


def notes_Q2(item, pole, program=0):
    p = item.params
    third_cents = p["third_second_base_cents"] if pole == "base" else p["third_second_edit_cents"]
    root = p["root_midi"]
    dur = p["chord_dur_seconds"]

    def triad(third_cents_val):
        third_semitones = round(third_cents_val / 100.0)
        return [
            dict(midi=root, start=0.0, dur=dur, program=program),
            dict(midi=root + third_semitones, start=0.0, dur=dur, program=program),
            dict(midi=root + 7, start=0.0, dur=dur, program=program),
        ]

    chord1 = triad(p["third_first_cents"])
    chord2 = triad(third_cents)
    return chord1, chord2


def notes_Q3(item, pole, program=None):
    """`program` is ignored -- Q3's programs come from program_pair_base/edit,
    not the caller (unlike Q1/Q2/Q4, which do take an overridable `program`
    for the render-time instrument family)."""
    p = item.params
    pair = p["program_pair_base"] if pole == "base" else p["program_pair_edit"]
    dur = p["note_dur_seconds"]
    m1 = [dict(midi=m, start=i * dur, dur=dur, program=pair[0]) for i, m in enumerate(p["melody"])]
    m2 = [dict(midi=m, start=i * dur, dur=dur, program=pair[1]) for i, m in enumerate(p["melody"])]
    return m1, m2


def notes_Q4(item, pole, program=0):
    p = item.params
    gold = _gold(item, pole)
    ref = [dict(midi=p["click_midi"], start=t, dur=p["click_dur_seconds"], is_drum=True) for t in p["onsets_ref"]]
    fast = [dict(midi=p["click_midi"], start=t, dur=p["click_dur_seconds"], is_drum=True) for t in p["onsets_fast"]]
    first, second = (fast, ref) if gold == "first" else (ref, fast)
    return first, second


def build_notes(task, item, pole, program=0):
    """Uniform dispatcher: (notes_first, notes_second)."""
    if task == "Q1_melody":
        return notes_Q1(item, pole, program)
    if task == "Q2_harmony":
        return notes_Q2(item, pole, program)
    if task == "Q3_timbre":
        return notes_Q3(item, pole)
    if task == "Q4_rhythm":
        return notes_Q4(item, pole, program)
    raise ValueError(f"unknown task: {task}")


def synth_transposed_invariance(task, item, window, sf2_path, sr, program=0):
    """rows 1-2 only: the second clip's pitch shifted by
    item.params['transposition_cents'], first clip untouched -- same
    "perturb clip B, keep clip A fixed" pattern as every other nuisance
    family."""
    transpose_cents = item.params["transposition_cents"]
    notes1, notes2 = build_notes(task, item, "base", program)
    notes2_t = [{**n, "cents": n.get("cents", 0.0) + transpose_cents} for n in notes2]
    return synth_notes(notes1, window, sf2_path, sr), synth_notes(notes2_t, window, sf2_path, sr)


def apply_family_to_pair(clips, sr, family, family_cfg, rng, seed):
    """Draw ONE set of family parameters and apply them identically to every
    clip in `clips`, so a re-render represents a single coherent production
    path (both clips 'recorded in the same room'), not independent draws."""
    if family == "room":
        rt60 = float(rng.choice(family_cfg["rt60_values_nuisance"]))
        room_dim = tuple(family_cfg["room_dim"])
        out = [transforms.apply_room(y, sr, rt60, room_dim, seed)[0] for y in clips]
        return out, {"rt60": rt60, "room_dim": list(room_dim), "seed": seed}
    if family == "codec":
        setting = family_cfg["settings_nuisance"][rng.integers(0, len(family_cfg["settings_nuisance"]))]
        out = [transforms.apply_codec(y, sr, setting["codec"], setting["bitrate_kbps"])[0] for y in clips]
        return out, dict(setting)
    if family == "eq":
        gain = float(rng.choice(family_cfg["gain_db_choices"]))
        out = [transforms.apply_eq(y, sr, gain, family_cfg["low_shelf_hz"], family_cfg["high_shelf_hz"])[0]
               for y in clips]
        return out, {"gain_db": gain, "low_shelf_hz": family_cfg["low_shelf_hz"],
                      "high_shelf_hz": family_cfg["high_shelf_hz"]}
    if family == "hiss":
        out = [transforms.apply_hiss(y, sr, family_cfg["level_dbfs"], seed)[0] for y in clips]
        return out, {"level_dbfs": family_cfg["level_dbfs"], "seed": seed}
    raise ValueError(f"apply_family_to_pair: {family} is not a post-render family")


def finalize_clips(clips, sr, target_lufs, n_samples):
    """Trim to exact length, then loudness-normalise per-clip (every row,
    now that C3_loudness -- the only pairwise exception -- is gone)."""
    finalized = []
    lufs_params = None
    for y in clips:
        y_t = transforms.trim_or_pad(y, n_samples)
        y_n, lufs_params = transforms.loudness_normalize(y_t, sr, target_lufs)
        finalized.append(y_n)
    return finalized, lufs_params


def write_clip(y, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")


# --- Realisation verification ---

def _median_f0(y, sr):
    import librosa
    f0, voiced, _ = librosa.pyin(y.astype(np.float64), fmin=50, fmax=2000, sr=sr)
    voiced_f0 = f0[voiced] if voiced is not None else f0[~np.isnan(f0)]
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)] if voiced_f0 is not None else np.array([])
    if len(voiced_f0) == 0:
        return None
    return float(np.median(voiced_f0))


def measure_interval_cents(y1, y2, sr):
    """Measured interval between two rendered tones, in cents -- relative,
    not compared against an idealised 12-TET/A440 frequency, since real
    sampled soundfonts carry several cents of their own per-sample
    mistuning (see base.yaml tolerances.pitch_cents note)."""
    f1, f2 = _median_f0(y1, sr), _median_f0(y2, sr)
    if f1 is None or f2 is None:
        return None
    return 1200.0 * np.log2(f2 / f1)


def measure_click_rate_hz(y, sr, min_gap_seconds=0.2):
    """min_gap_seconds must clear a single click's own decay ripple (its
    hi-hat sample can dip below the energy threshold mid-note and re-cross
    it, registering as a second onset) while staying well under the
    fastest true inter-click period this grid ever needs to resolve
    (Q4_rhythm ratio=1.50 gives 1/(2.0*1.5)=0.33s). The old default (0.05s,
    exactly equal to click_dur_seconds) let exactly one such ripple through
    per affected clip -- confirmed by the full-scale run: (n+1-1)/span
    with the true n producing e.g. 2.0Hz gives ~2.286Hz with one extra
    onset, matching the observed 2.284Hz almost exactly."""
    frame = max(1, int(0.005 * sr))
    n_frames = len(y) // frame
    if n_frames < 2:
        return None
    env = np.array([np.sqrt(np.mean(y[i * frame:(i + 1) * frame] ** 2)) for i in range(n_frames)])
    thresh = 0.3 * env.max()
    if thresh <= 0:
        return None
    above = env > thresh
    rising = np.where(above[1:] & ~above[:-1])[0] + 1
    onset_times = rising * frame / sr
    onset_times = onset_times[np.concatenate(([True], np.diff(onset_times) > min_gap_seconds))] \
        if len(onset_times) else onset_times
    if len(onset_times) < 2:
        return None
    return (len(onset_times) - 1) / (onset_times[-1] - onset_times[0])
