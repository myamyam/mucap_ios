"""Experimental copy: gamma boost and transient curve compensation.

Production wav_to_ahap.py is unchanged. Single-pattern export raises the
shared intensity curve at impact times; overlapping continuous haptics
are also briefly raised. This is an A/B test, not independent tracks.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

if __package__:
    from .hybrid_test import PRESETS, apply_quality, load_audio_mono, render_hybrid
else:
    from hybrid_test import PRESETS, apply_quality, load_audio_mono, render_hybrid


#HapticIntensity 계산
def rms_envelope(signal, sr, times, window_ms=30.0):
    signal = np.asarray(signal, dtype=np.float64)
    if not len(signal):
        return np.zeros_like(times, dtype=float)
    power = uniform_filter1d(np.square(signal),
                             size=max(1, round(sr * window_ms / 1000)),
                             mode="constant")
    return np.sqrt(np.maximum(0, np.interp(times * sr, np.arange(len(signal)), power, right=0)))


"""
제어 곡선 점 개수 줄임 : 햅틱 전송 용량 제한 해결
input: 원본 times, 원본 values, tolerance(허용 오차), forced_times
output: 줄어든 times, values
"""
def simplify_curve(times, values, tolerance, forced_times=()):
    if len(times) <= 2:
        return times, values
    keep = {0, len(times) - 1}
    for t in forced_times:
        keep.add(int(np.argmin(np.abs(times - t))))
    anchors=sorted(keep)
    pending = list(zip(anchors[:-1], anchors[1:]))
    while pending:
        first, last = pending.pop()
        if last - first <= 1:
            continue
        predicted = np.interp(times[first:last + 1],
                              [times[first], times[last]], [values[first], values[last]])
        errors = np.abs(values[first:last + 1] - predicted)
        split = first + int(np.argmax(errors))
        if errors.max() > tolerance:
            keep.add(split)
            pending.extend([(first, split), (split, last)])
    indices = sorted(keep)
    return times[indices], values[indices]

"""
simplify_curve() -> parameter_curves()-> .ahap file
time, value array를 dictionary로 변환
curve 하나에 최대 16개 control point 허용
"""
def parameter_curves(parameter_id, times, values):
    curves = []
    if len(times)<2:
        return curves
    for start in range(0, len(times) - 1, 15):
        stop = min(start + 16, len(times))
        origin = float(times[start])
        curves.append({"ParameterCurve": {
            "ParameterID": parameter_id,
            "Time": origin,
            "ParameterCurveControlPoints": [
                {"Time": float(times[i] - origin), "ParameterValue": float(values[i])}
                for i in range(start, stop)
            ],
        }})
    return curves

"""
진동 지속 구간 추출 (max_gap 보다 큰 구간 반환)
output: [(start_time, end_time), ...]
"""
def active_regions(times, energy, threshold=0.045, min_duration=0.07, max_gap=0.07):
    active = np.asarray(energy) >= threshold
    if not np.any(active):
        return []
    starts = np.where(active & ~np.r_[False, active[:-1]])[0]
    ends = np.where(active & ~np.r_[active[1:], False])[0]
    regions = []
    for si, ei in zip(starts, ends):
        start, end = float(times[si]), float(times[min(ei + 1, len(times) - 1)])
        if regions and start - regions[-1][1] <= max_gap:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))
    return [(a, b) for a, b in regions if b - a >= min_duration]


"""
순간 충격 추출
output: (onset_times, onset_strengths) ---array of times, array of strengths
"""
def impact_features(y, sr, hop):
    """Raw Mel power, before hybrid normalization/gamma; centered timestamps."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=hop, n_mels=40,
        fmin=40, fmax=min(12000, sr / 2), power=2, center=True)
    db = 10 * np.log10(np.maximum(mel.sum(axis=0), 1e-12))
    log_mel = librosa.power_to_db(mel, ref=1.0)
    flux = np.maximum(np.diff(log_mel, axis=1, prepend=log_mel[:, :1]), 0).mean(axis=0)
    return np.arange(len(db)) * hop / sr, db, flux, float(np.percentile(db, 30))


def detect_impacts(y, sr, hop, *, onset_threshold=0.25,
                   min_onset_interval=0.10, max_transients_per_sec=8.0):
    """Select salient energy pillars with a positive spectral-flux attack.

    No expected impact count is supplied. Thresholds are experimental dB
    margins over this clip's background; continuous/mixed recordings still
    require device and dataset evaluation.
    """
    if len(y) < 2 or np.max(np.abs(y)) < 1e-7:
        return np.array([]), np.array([])
    times, db, flux, background = impact_features(y, sr, hop)
    peaks, _ = find_peaks(db, height=background + 12, prominence=10,
                         distance=max(1, round(min_onset_interval * sr / hop)))
    lookback = max(1, round(0.04 * sr / hop))
    scale = max(float(np.max(flux)), 1e-10)
    selected = []
    for peak in peaks:
        first = max(0, peak - lookback)
        # Require a rapid rise, not merely a loud sustained signal.
        if db[peak] - float(np.min(db[first:peak + 1])) < 6:
            continue
        strength = float(np.max(flux[first:peak + 1]) / scale)
        if strength >= onset_threshold:
            selected.append((float(times[peak]), strength))
    limited = []
    for sec in range(int(np.ceil(len(y) / sr))):
        group = [v for v in selected if sec <= v[0] < sec + 1]
        limited.extend(sorted(group, key=lambda v: v[1], reverse=True)
                       [:max(1, int(max_transients_per_sec))])
    limited.sort()
    if not limited:
        return np.array([]), np.array([])
    return np.array([v[0] for v in limited]), np.array([v[1] for v in limited])

"""
active_regions()에서 추출한 region으로 event(dict) 생성
"""
def make_event(t, event_type, intensity, sharpness, duration=None):
    event = {"Time": float(t), "EventType": event_type,
             "EventParameters": [
                 {"ParameterID": "HapticIntensity", "ParameterValue": float(np.clip(intensity, 0, 1))},
                 {"ParameterID": "HapticSharpness", "ParameterValue": float(np.clip(sharpness, 0, 1))},
             ]}
    if duration is not None:
        event["EventDuration"] = float(duration)
    return {"Event": event}


'''
main function으로, wav 파일 입력받아 ahap 파일로 변환
./multimodal_caption/hybrid_test.py의 PRESETS, apply_quality, load_audio_mono, render_hybrid 사용

- render_hybrid()
output: (out, info) rendered audio, info dict

info dict keys:
- sr: sample rate
- hop: hop length
- nfft: FFT size
- ncar: number of mel bands


example usage:
python -m multimodal_caption.wav_to_ahap \                         
  --in tests/ttock.wav \
  --out tests/ttock_10hz.ahap \
  --preset ambience \
  --control-hz 100 \
  --curve-tolerance 0.01
'''

def convert_wav(in_path, out_path=None, *, preset="ambience", quality="mid",
                strength=1.0, gain=1.0, reference_rms=0.08, control_hz=100.0,
                curve_tolerance=0.01, onset_threshold=0.25, 
                min_onset_interval=0.10, max_transients_per_sec=8.0,
                continuous_threshold=0.045, transient_gain=0.8):

    controls=(strength, gain, reference_rms, control_hz, curve_tolerance,
              onset_threshold, min_onset_interval, max_transients_per_sec,
              continuous_threshold, transient_gain)

    if not all(np.isfinite(v) for v in controls):
        raise ValueError("Controls must be finite")

    if not 0 <= curve_tolerance <= 1 or not 0<= onset_threshold <= 1 or not 0 <= continuous_threshold <= 1:
        raise ValueError("Thresholds and curve_tolerance must be in [0, 1]")

    if strength < 0 or gain < 0 or reference_rms <= 0 or not 1 <= control_hz <= 200:
        raise ValueError("strength/gain >= 0, reference_rms > 0, control_hz in [1, 200] required")

    if min_onset_interval <= 0 or max_transients_per_sec <= 0 or transient_gain < 0:
        raise ValueError("min_onset_interval/max_transients_per_sec > 0, transient_gain >= 0 required")
    
    source = Path(in_path)
    if source.suffix.lower() != ".wav":
        raise ValueError("Input must be a WAV file")
    if not source.is_file():
        raise ValueError(f"Input not found: {source}")
    
    destination = Path(out_path) if out_path else source.with_suffix(".ahap")
    if destination.suffix.lower() != ".ahap" or destination.resolve() == source.resolve():
        raise ValueError("Output must be a separate .ahap file")

    p = apply_quality(replace(PRESETS[preset]), quality)
    y, sr = load_audio_mono(str(source), p.sr)

    if not len(y) or not np.all(np.isfinite(y)):
        raise ValueError("WAV must contain nonempty, finite audio samples")
    
    duration = len(y) / sr

    padded = np.pad(y, (0, max(0, p.nfft - len(y))))

    features = {}
    rendered, info = render_hybrid(padded, sr, p, strength, 1.0, 1.0, features=features)
    rendered = rendered[:len(y)]
    
    times = np.r_[np.arange(0, duration, 1.0 / control_hz), duration]

    low_rms= rms_envelope(features["low"][:len(y)], sr, times)
    rem_rms = rms_envelope(features["rem"][:len(y)], sr, times)
    raw_rms= rms_envelope(y, sr, times)

    base=np.clip(gain*rms_envelope(rendered, sr, times)/reference_rms, 0, 1)
    low_ratio=np.clip(low_rms/(raw_rms+1e-8), 0, 1)
    weights = features["A"].astype(np.float64) ** 2

    positions = np.linspace(0, 1, p.ncar)
    mel_texture = (weights * positions[:, None]).sum(axis=0) / (weights.sum(axis=0) + 1e-12)
    frame_times = features["frame_times"]
    texture = np.interp(times, frame_times, mel_texture, left=mel_texture[0], right=mel_texture[-1]) if len(frame_times) else np.zeros_like(times)

    rem_fraction= rem_rms**2 / (rem_rms**2 + low_rms**2 + 1e-12)
    
    sharpness = np.clip((0.2 * (1 - low_ratio) + 0.8 * texture) * rem_fraction, 0, 1)

    onset_times, onset_strengths = detect_impacts(y, sr, p.hop, onset_threshold=onset_threshold,
            min_onset_interval=min_onset_interval, max_transients_per_sec=max_transients_per_sec)

    continuous = base.copy()
    for t, s in zip(onset_times, onset_strengths):
        notch = np.exp(-0.5 * ((times - t) / 0.035) ** 2)
        continuous *= 1.0 - 0.35 * float(s) * notch

    # Existing mel sqrt epsilon otherwise produces a synthetic floor for exact silence/DC.
    if np.max(np.abs(y.astype(np.float64) - np.mean(y, dtype=np.float64))) < 1e-8:
        continuous[:] = 0
        sharpness[:] = 0
        onset_times, onset_strengths = np.array([]), np.array([])

    fade = min(p.fade_ms / 1000, duration / 2)
    if fade>0:
        continuous *= np.minimum(np.clip(times / fade, 0, 1),
                                 np.clip((duration - times) / fade, 0, 1))
    continuous[0] = continuous[-1] = 0

    pattern = []
    # start = 0.0

    # Ordinary sustained activity is gated by original Mel energy. Remove only
    # the brief attack; decide whether its low-frequency decay deserves a tail.
    energy_times, energy_db, _, noise_db = impact_features(y, sr, p.hop)
    audible = np.interp(times, energy_times, energy_db) >= noise_db + 12
    continuous[~audible] = 0
    for impact_time in onset_times:
        continuous[(times >= impact_time - 0.04) &
                   (times <= impact_time + 0.03)] = 0
    sustained_regions = active_regions(times, continuous, threshold=continuous_threshold,
                                      min_duration=0.15, max_gap=0.03)
    sustained_mask = np.zeros(len(times), dtype=bool)
    for a, b in sustained_regions:
        sustained_mask |= (times >= a) & (times < b)
    continuous[~sustained_mask] = 0

    # Keep measurable low-frequency decay after bass-heavy impacts. The floor
    # is determined before gain/boost; tails stop at silence or the next attack.
    noise_low = max(float(np.percentile(low_rms, 30)), 1e-6)
    tail = np.zeros_like(times)
    for i, t in enumerate(onset_times):
        impact_low_ratio = float(np.interp(t, times, low_ratio))
        if impact_low_ratio < 0.75:
            continue
        peak_low = float(np.interp(t, times, low_rms))
        threshold = max(4 * noise_low, 0.1 * peak_low)
        end_limit = onset_times[i + 1] - 0.02 if i + 1 < len(onset_times) else duration
        indices = np.flatnonzero((times >= t + 0.03) & (times < end_limit))
        if not len(indices):
            continue
        below = np.flatnonzero(low_rms[indices] < threshold)
        if len(below):
            indices = indices[:below[0]]
        if len(indices) < 2 or times[indices[-1]] - times[indices[0]] < 0.05:
            continue
        magnitude = gain * np.maximum(low_rms[indices] - noise_low, 0) / reference_rms
        # Soft saturation preserves amplitude differences even above reference.
        values = (magnitude / (magnitude + 0.5)) ** 0.85
        taper = np.minimum(np.clip((times[indices] - times[indices[0]]) / 0.02, 0, 1),
                           np.clip((times[indices[-1]] - times[indices]) / 0.03, 0, 1))
        tail[indices] = np.maximum(tail[indices], values * taper)
    continuous = np.maximum(continuous ** 0.75, tail)
    regions = active_regions(times, continuous, threshold=continuous_threshold,
                             min_duration=0.05, max_gap=0.0)

    # Detect activity BEFORE boosting, so gamma does not activate quiet gaps.
    mask = np.zeros(len(times), dtype=bool)
    for a, b in regions:
        mask |= (times >= a) & (times <= b)
        start = a
        while start < b - 1e-7:
            length = min(30.0, b - start)
            pattern.append(make_event(start, "HapticContinuous", 1.0, 0.0, length))
            start += length
    continuous[~mask] = 0
    sharpness[~mask] = 0

    impacts = []
    for t, s in zip(onset_times, onset_strengths):
        if not 0 <= t < duration:
            continue
        local_rms = float(np.interp(t, times, raw_rms))
        if local_rms < reference_rms * 0.008:
            continue
        local_tex = float(np.interp(t, times, texture))
        local_low_ratio = float(np.interp(t, times, low_ratio))
        level = gain * local_rms / reference_rms
        # Avoid a hard clamp at reference_rms: louder hits retain headroom.
        loudness = (level / (level + 0.5)) ** 0.85
        attack = float(np.clip(transient_gain * (0.7 + 0.3 * s) * loudness, 0, 1))
        if attack < 0.08:
            continue
        # Spectral balance, rather than onset strength, determines hit texture.
        impact_sharpness = float(np.clip(
            0.15 + 0.75 * (1 - local_low_ratio ** 2) + 0.1 * local_tex, 0, 1))
        impacts.append((float(t), attack, impact_sharpness))

    # Include exact impact timestamps, with narrow curve supports. Raising the
    # multiplier makes target / multiplier <= 1, avoiding clipped compensation.
    forced = sorted({v for t, _, _ in impacts
                     for v in (max(0., t - 0.01), t, min(duration, t + 0.01))})
    curve_times = np.unique(np.r_[times, forced])
    intensity_values = np.interp(curve_times, times, continuous)
    sharpness_values = np.interp(curve_times, times, sharpness)
    for t, target, impact_sharpness in impacts:
        index = int(np.searchsorted(curve_times, t))
        intensity_values[index] = max(intensity_values[index], target)
        event_intensity = target / intensity_values[index] if regions else target
        pattern.append(make_event(t, "HapticTransient",
                                  event_intensity, impact_sharpness))

    if regions:
        ti, vi = simplify_curve(curve_times, intensity_values, curve_tolerance, forced)
        ts, vs = simplify_curve(curve_times, sharpness_values, curve_tolerance, forced)
        pattern.extend(parameter_curves("HapticIntensityControl", ti, vi))
        pattern.extend(parameter_curves("HapticSharpnessControl", ts, vs))

    pattern.sort(key=lambda obj: obj.get("Event", obj.get("ParameterCurve"))["Time"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"Version": 1.0, "Pattern": pattern}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination, dict(info, duration=duration, control_points=len(times),
                             continuous_events=sum("Event" in e and e["Event"]["EventType"] == "HapticContinuous" for e in pattern),
                             transient_events=sum("Event" in e and e["Event"]["EventType"] == "HapticTransient" for e in pattern),
                             peak_intensity=float(base.max()), curve_tolerance=curve_tolerance)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, help="Input WAV")
    parser.add_argument("--out", dest="out_path", help="Output AHAP (default: beside input)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="ambience")
    parser.add_argument("--quality", choices=["low", "mid", "high"], default="mid")
    parser.add_argument("--strength", type=float, default=1.0, help="Hybrid remap ratio multiplier")
    parser.add_argument("--gain", type=float, default=1.0, help="Final AHAP intensity multiplier")
    parser.add_argument("--reference-rms", type=float, default=0.08)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--curve-tolerance", type=float, default=0.01)
    parser.add_argument("--onset-threshold", type=float, default=0.25)
    parser.add_argument("--min-onset-interval", type=float, default=0.10)
    parser.add_argument("--max-transients-per-sec", type=float, default=8.0)
    parser.add_argument("--continuous-threshold", type=float, default=0.045)
    parser.add_argument("--transient-gain", type=float, default=0.8)
    args = parser.parse_args()

    try:
        path, info = convert_wav(**vars(args))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Written: {path}\n" + json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
