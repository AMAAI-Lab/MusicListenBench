"""Post-render production nuisance transforms (generator design).

`instrument` (soundfont-file swap) and `pitch_resample` (A4's substitute
family) are *render-time* choices -- they change what gets synthesized, not
an already-rendered waveform -- so they live in render.py's synth step, not
here. Everything in this file operates on a rendered mono float32 array at
a fixed sample rate and returns (transformed_array, params_dict), the
params_dict going straight into the manifest's `transform_params`.

This file holds the room, codec and loudness helpers plus the `eq` (pedalboard
shelving EQ) and `hiss` (additive noise) families.
"""

import os
import subprocess
import tempfile

import numpy as np
import pyloudnorm as pyln
import pyroomacoustics as pra
import soundfile as sf


def load_mono(path, sr, offset=0.0, duration=None):
    y, file_sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        import librosa
        y = librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    if duration is not None:
        start = int(offset * sr)
        end = start + int(duration * sr)
        y = y[start:end]
        if len(y) < int(duration * sr):
            y = np.pad(y, (0, int(duration * sr) - len(y)))
    return y.astype(np.float32)


def loudness_normalize(y, sr, target_lufs):
    """Per-clip ITU-R BS.1770 normalisation, used everywhere
    now that C3_loudness (the row that needed pairwise normalisation) is
    gone from the four-row grid."""
    meter = pyln.Meter(sr)
    y64 = y.astype(np.float64)
    loudness = meter.integrated_loudness(y64)
    if not np.isfinite(loudness):
        return y, {"target_lufs": target_lufs, "measured_lufs": None}
    normed = pyln.normalize.loudness(y64, loudness, target_lufs)
    peak = np.abs(normed).max()
    if peak > 0.99:
        normed = normed * (0.99 / peak)
    return normed.astype(np.float32), {"target_lufs": target_lufs, "measured_lufs": float(loudness)}


def apply_room(y, sr, rt60, room_dim, seed):
    rng = np.random.default_rng(seed)
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)
    src_pos = rng.uniform(low=[0.5, 0.5, 0.5], high=[d - 0.5 for d in room_dim])
    mic_pos = rng.uniform(low=[0.5, 0.5, 0.5], high=[d - 0.5 for d in room_dim])
    room.add_source(src_pos.tolist(), signal=y.astype(np.float64))
    room.add_microphone(mic_pos.tolist())
    room.simulate()
    out = room.mic_array.signals[0]
    peak = np.abs(out).max()
    ref_peak = np.abs(y).max()
    if peak > 0 and ref_peak > 0:
        out = out / peak * ref_peak
    return out.astype(np.float32), {"rt60": rt60, "room_dim": list(room_dim), "seed": seed}


_CODEC_FORMAT = {"mp3": "mp3", "opus": "ogg"}
_CODEC_ENCODER = {"mp3": None, "opus": "libopus"}


def _run_ffmpeg(args, attempt_desc):
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({attempt_desc}): {' '.join(args)}\n{proc.stderr}")


def apply_codec(y, sr, codec, bitrate_kbps, retries=2):
    """Round-trip through a lossy codec. Returns audio at its ORIGINAL
    (possibly padded) length; render.py's trim step is
    responsible for cutting back to the exact intended sample count, so
    codec padding is caught by that single shared step rather than here."""
    ext = {"mp3": "mp3", "opus": "opus"}[codec]
    fmt = _CODEC_FORMAT[codec]
    last_err = None
    for attempt in range(retries + 1):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav_in = os.path.join(tmp, "in.wav")
                enc = os.path.join(tmp, f"enc.{ext}")
                wav_out = os.path.join(tmp, "out.wav")
                sf.write(wav_in, y, sr)
                encode_args = ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_in]
                if _CODEC_ENCODER[codec]:
                    encode_args += ["-c:a", _CODEC_ENCODER[codec]]
                encode_args += ["-b:a", f"{bitrate_kbps}k", "-f", fmt, enc]
                _run_ffmpeg(encode_args, "encode")
                if os.path.getsize(enc) == 0:
                    raise RuntimeError(f"ffmpeg encode produced an empty {codec} file")
                _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error", "-f", fmt, "-i", enc, wav_out], "decode")
                out, out_sr = sf.read(wav_out, always_2d=True)
                out = out.mean(axis=1)
                if out_sr != sr:
                    import librosa
                    out = librosa.resample(out.astype(np.float32), orig_sr=out_sr, target_sr=sr)
            return out.astype(np.float32), {"codec": codec, "bitrate_kbps": bitrate_kbps}
        except (RuntimeError, subprocess.CalledProcessError) as e:
            last_err = e
            if attempt < retries:
                continue
    raise RuntimeError(f"apply_codec failed after {retries + 1} attempts: {last_err}")


def apply_eq(y, sr, gain_db, low_shelf_hz, high_shelf_hz):
    """Shelving EQ via pedalboard (generator design). One low shelf and one high
    shelf, both at `gain_db` (a single signed value drawn per trial)."""
    from pedalboard import HighShelfFilter, LowShelfFilter, Pedalboard

    board = Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=low_shelf_hz, gain_db=gain_db),
        HighShelfFilter(cutoff_frequency_hz=high_shelf_hz, gain_db=gain_db),
    ])
    out = board(y.astype(np.float32), sr)
    return out.astype(np.float32), {"gain_db": gain_db, "low_shelf_hz": low_shelf_hz,
                                     "high_shelf_hz": high_shelf_hz}


def apply_hiss(y, sr, level_dbfs, seed):
    """Additive white noise at a fixed dBFS level (generator design)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=len(y)).astype(np.float32)
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-12
    target_rms = 10 ** (level_dbfs / 20.0)
    noise = noise * (target_rms / noise_rms)
    out = y + noise
    peak = np.abs(out).max()
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32), {"level_dbfs": level_dbfs, "seed": seed}


def trim_or_pad(y, n_samples):
    """Pipeline-order step: decode-and-trim back to the
    exact intended sample count after every transform (room lengthens via
    reverb tail, codec pads on encode)."""
    if len(y) == n_samples:
        return y
    if len(y) > n_samples:
        return y[:n_samples]
    return np.pad(y, (0, n_samples - len(y)))


def apply_family(y, sr, family, family_cfg, seed):
    """Dispatcher for the four post-render families. `instrument` and
    `pitch_resample` are handled at synth time (render.py), not here."""
    if family == "room":
        rng = np.random.default_rng(seed)
        rt60 = float(rng.choice(family_cfg["rt60_values_nuisance"]))
        return apply_room(y, sr, rt60, tuple(family_cfg["room_dim"]), seed)
    if family == "codec":
        rng = np.random.default_rng(seed)
        setting = family_cfg["settings_nuisance"][rng.integers(0, len(family_cfg["settings_nuisance"]))]
        return apply_codec(y, sr, setting["codec"], setting["bitrate_kbps"])
    if family == "eq":
        rng = np.random.default_rng(seed)
        gain = float(rng.choice(family_cfg["gain_db_choices"]))
        return apply_eq(y, sr, gain, family_cfg["low_shelf_hz"], family_cfg["high_shelf_hz"])
    if family == "hiss":
        return apply_hiss(y, sr, family_cfg["level_dbfs"], seed)
    raise ValueError(f"apply_family: {family} is a render-time family, not a post-render transform")
