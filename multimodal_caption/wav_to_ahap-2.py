#!/usr/bin/env python3
"""Environmental WAV -> hybrid DSP features -> Core Haptics AHAP.

Continuous texture and sparse transient impacts are encoded separately.
All mapping constants are prototype defaults, not perceptually validated.
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


def rms_envelope(signal, sr, times, window_ms=30.0):
    signal = np.asarray(signal, dtype=np.float64)
    if not len(signal):
        return np.zeros_like(times, dtype=float)
    power = uniform_filter1d(np.square(signal), size=max(1, round(sr * window_ms / 1000)), mode="constant")
    return np.sqrt(np.maximum(0, np.interp(np.asarray(times) * sr, np.arange(len(signal)), power, right=0)))


def simplify_curve(times, values, tolerance, forced_times=()):
    """Bound linear interpolation error, preserving onset-adjacent control points."""
    if len(times) <= 2:
        return times, values
    keep = {0, len(times) - 1}
    for t in forced_times:
        keep.add(int(np.argmin(np.abs(times - t))))
    anchors = sorted(keep)
    pending = list(zip(anchors[:-1], anchors[1:]))
    while pending:
        first, last = pending.pop()
        if last - first <= 1:
            continue
        predicted = np.interp(times[first:last + 1], [times[first], times[last]], [values[first], values[last]])
        errors = np.abs(values[first:last + 1] - predicted)
        split = first + int(np.argmax(errors))
        if errors.max() > tolerance:
            keep.add(split)
            pending.extend(((first, split), (split, last)))
    indices = sorted(keep)
    return times[indices], values[indices]


def parameter_curves(parameter_id, times, values):
    """Chunk curves with shared boundaries; control-point times are curve-relative."""
    curves = []
    if len(times) < 2:
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


def active_regions(times, energy, threshold=0.045, min_duration=0.07, max_gap=0.07):
    """Group active frames, joining short gaps, to avoid silent continuous events."""
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


def detect_impacts(y, sr, hop, *, onset_threshold=0.25,
                   min_onset_interval=0.10, max_transients_per_sec=8.0):
    """Spectral-flux peak picking with local prominence and density limits."""
    if len(y) < 2 or np.max(np.abs(y)) < 1e-7:
        return np.array([]), np.array([])
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, center=False)
    if not len(onset) or np.max(onset) <= 1e-10:
        return np.array([]), np.array([])
    # Robust dynamic-range scaling within this clip.
    scale = max(float(np.percentile(onset, 95)), 1e-10)
    normalized = np.clip(onset / scale, 0, 1)
    min_dist = max(1, int(round(min_onset_interval * sr / hop)))
    peaks, props = find_peaks(normalized, height=onset_threshold,
                              prominence=max(0.08, onset_threshold * 0.35), distance=min_dist)
    if len(peaks) == 0:
        return np.array([]), np.array([])
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    peak_strength = props["peak_heights"]
    # Limit dense stochastic sounds (rain/gravel) per-second rather than vibrating on every peak.
    keep = []
    for sec in range(int(np.ceil(len(y) / sr))):
        indices = np.where((peak_times >= sec) & (peak_times < sec + 1))[0]
        limit = max(1, int(max_transients_per_sec))
        selected = indices[np.argsort(peak_strength[indices])[-limit:]]
        keep.extend(selected.tolist())
    keep = np.array(sorted(keep), dtype=int)
    return peak_times[keep], peak_strength[keep]


def make_event(t, event_type, intensity, sharpness, duration=None):
    event = {"Time": float(t), "EventType": event_type,
             "EventParameters": [
                 {"ParameterID": "HapticIntensity", "ParameterValue": float(np.clip(intensity, 0, 1))},
                 {"ParameterID": "HapticSharpness", "ParameterValue": float(np.clip(sharpness, 0, 1))},
             ]}
    if duration is not None:
        event["EventDuration"] = float(duration)
    return {"Event": event}


def convert_wav(in_path, out_path=None, *, preset="ambience", quality="mid",
                strength=1.0, gain=1.0, reference_rms=0.25, control_hz=100.0,
                curve_tolerance=0.01, onset_threshold=0.25,
                min_onset_interval=0.10, max_transients_per_sec=8.0,
                continuous_threshold=0.045, transient_gain=0.8):
    """Convert environmental audio to AHAP. Returns (path, metrics)."""
    controls = (strength, gain, reference_rms, control_hz, curve_tolerance,
                onset_threshold, min_onset_interval, max_transients_per_sec,
                continuous_threshold, transient_gain)
    if not all(np.isfinite(v) for v in controls):
        raise ValueError("Controls must be finite")
    if strength < 0 or gain < 0 or reference_rms <= 0 or not 1 <= control_hz <= 200:
        raise ValueError("Invalid strength, gain, reference_rms or control_hz")
    if not 0 <= curve_tolerance <= 1 or not 0 <= onset_threshold <= 1 or not 0 <= continuous_threshold <= 1:
        raise ValueError("Thresholds and curve tolerance must be in [0,1]")
    if min_onset_interval <= 0 or max_transients_per_sec <= 0 or transient_gain < 0:
        raise ValueError("Invalid transient controls")
    source = Path(in_path)
    if source.suffix.lower() != ".wav" or not source.is_file():
        raise ValueError("Input must be an existing WAV file")
    destination = Path(out_path) if out_path else source.with_suffix(".ahap")
    if destination.suffix.lower() != ".ahap" or destination.resolve() == source.resolve():
        raise ValueError("Output must be a separate .ahap file")

    p = apply_quality(replace(PRESETS[preset]), quality)
    y, sr = load_audio_mono(str(source), p.sr)
    if not len(y) or not np.all(np.isfinite(y)):
        raise ValueError("Audio must contain finite, nonempty samples")
    duration = len(y) / sr
    padded = np.pad(y, (0, max(0, p.nfft - len(y))))
    features = {}
    rendered, info = render_hybrid(padded, sr, p, strength, 1.0, 1.0, features=features)
    rendered = rendered[:len(y)]
    times = np.r_[np.arange(0, duration, 1.0 / control_hz), duration]

    low_rms = rms_envelope(features["low"][:len(y)], sr, times)
    rem_rms = rms_envelope(features["rem"][:len(y)], sr, times)
    raw_rms = rms_envelope(y, sr, times)
    # Keep existing hybrid output level as the baseline; suppress percussive peaks
    # slightly in the continuous bed, so they can be represented as transient events.
    base = np.clip(gain * rms_envelope(rendered, sr, times) / reference_rms, 0, 1)
    low_ratio = np.clip(low_rms / (raw_rms + 1e-8), 0, 1)
    weights = features["A"].astype(np.float64) ** 2
    positions = np.linspace(0, 1, p.ncar)
    mel_texture = (weights * positions[:, None]).sum(axis=0) / (weights.sum(axis=0) + 1e-12)
    frame_times = features["frame_times"]
    texture = np.interp(times, frame_times, mel_texture, left=mel_texture[0], right=mel_texture[-1]) if len(frame_times) else np.zeros_like(times)
    rem_fraction = rem_rms ** 2 / (low_rms ** 2 + rem_rms ** 2 + 1e-12)
    sharpness = np.clip((0.2 * (1 - low_ratio) + 0.8 * texture) * rem_fraction, 0, 1)

    onset_times, onset_strengths = detect_impacts(y, sr, p.hop, onset_threshold=onset_threshold,
        min_onset_interval=min_onset_interval, max_transients_per_sec=max_transients_per_sec)
    # Reduce continuous bed at impacts, preventing obvious double emphasis. Make this
    # notch broad enough to reach nearby curve points, but shorter than a footstep.
    continuous = base.copy()
    for t, s in zip(onset_times, onset_strengths):
        notch = np.exp(-0.5 * ((times - t) / 0.035) ** 2)
        continuous *= 1.0 - 0.35 * float(s) * notch
    
    if np.max(np.abs(y - np.mean(y))) < 1e-8:
        continuous[:] = 0
        sharpness[:] = 0
        onset_times, onset_strengths = np.array([]), np.array([])
    fade = min(p.fade_ms / 1000, duration / 2)
    if fade > 0:
        continuous *= np.minimum(np.clip(times / fade, 0, 1), np.clip((duration - times) / fade, 0, 1))
    continuous[0] = continuous[-1] = 0

    pattern = []
    # Background segments only, with a small minimum activity threshold.
    regions = active_regions(times, continuous, threshold=continuous_threshold)
    for a, b in regions:
        while a < b - 1e-7:
            length = min(30.0, b - a)
            pattern.append(make_event(a, "HapticContinuous", 1.0, 0.0, length))
            a += length
    # Curve applies globally to the whole pattern, including transients, so the transient
    # base intensity is compensated for local curve scaling where possible.
    for t, s in zip(onset_times, onset_strengths):
        if t < 0 or t >= duration:
            continue
        local_rms = float(np.interp(t, times, raw_rms))
        if local_rms < reference_rms * 0.008:
            continue
        local_tex = float(np.interp(t, times, texture))
        attack = np.clip(transient_gain * s * max(0.2, min(1., local_rms / reference_rms)), 0, 1)
        # Avoid excessive transients in the densest noisy material.
        if attack < 0.08:
            continue
        transient_sharp = np.clip(0.25 + 0.55 * s + 0.2 * local_tex, 0, 1)
        # AHAP intensity controls can also modulate transients. We intentionally use
        # base intensity as a conservative lower bound, not a guarantee of exact strength.
        pattern.append(make_event(t, "HapticTransient", attack, transient_sharp))

    if regions:
        # Use zero intensity outside active regions; expand global curves to ensure
        # quiet intervals are truly quiet. Preserve onset-adjacent points through simplification.
        mask = np.zeros(len(times), dtype=bool)
        for a, b in regions:
            mask |= (times >= a) & (times <= b)
        continuous[~mask] = 0
        sharpness[~mask] = 0
        forced = [v for t in onset_times for v in (max(0, t - 0.04), t, min(duration, t + 0.04))]
        ti, vi = simplify_curve(times, continuous, curve_tolerance, forced)
        ts, vs = simplify_curve(times, sharpness, curve_tolerance, forced)
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
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="ambience")
    parser.add_argument("--quality", choices=["low", "mid", "high"], default="mid")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--reference-rms", type=float, default=0.25)
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
