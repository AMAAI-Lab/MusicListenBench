"""Audio loading shared by every model backend."""

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100   # native rate of every released clip


def load_mono(path, sr=SAMPLE_RATE, offset=0.0, duration=None):
    """Reads a clip, downmixes to mono and resamples to `sr` (each model's own rate)."""
    y, file_sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1)  # downmix to mono
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
