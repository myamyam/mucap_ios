"""NumPy normalization adapted from local Sound2Hap DSP implementations.

Audio: utils/normalization.py peak strategy with default -1 dB headroom.
Vibration: FreqShift.py divides by RMS * sqrt(2), targeting 1/sqrt(2).
Unlike Percept.py's per-segment scaling, we use one whole-clip multiplier
so temporal amplitude differences survive. AHAP chooses its own target RMS.
"""
import numpy as np

SOUND2HAP_RMS = 1.0 / np.sqrt(2.0)


def _signal(signal):
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError('Expected nonempty finite mono audio')
    return x


def peak_normalize(signal, headroom_db=1.0):
    if not np.isfinite(headroom_db) or headroom_db < 0:
        raise ValueError('peak headroom must be finite and nonnegative')
    x = _signal(signal)
    peak = float(np.max(np.abs(x)))
    target = float(10 ** (-headroom_db / 20))
    scale = target / peak if peak > 1e-12 else 1.0
    out = (x * scale).astype(np.float32)
    return out, dict(peak_before=peak, peak_after=float(np.max(np.abs(out))),
                     target_peak=target, multiplier=scale)


def rms_normalize(signal, target_rms=SOUND2HAP_RMS):
    if not np.isfinite(target_rms) or target_rms <= 0:
        raise ValueError('target RMS must be finite and positive')
    x = _signal(signal)
    before = float(np.sqrt(np.mean(x * x)))
    scale = target_rms / before if before > 1e-12 else 1.0
    out = (x * scale).astype(np.float32)
    # This is an analysis waveform, not PCM playback. Do not clip peaks here;
    # only the eventual AHAP intensity is bounded, with saturation reported.
    return out, dict(rms_before=before, rms_after=float(np.sqrt(np.mean(out.astype(float) ** 2))),
                     target_rms=float(target_rms), multiplier=float(scale))
