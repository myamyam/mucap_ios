"""
rms normalization 없이 peak normalization만 수행

AHAP v2: input peak normalization only; no vibration RMS normalization.

Intensity = clip(gain * hybrid RMS envelope / reference_rms, 0, 1).
Transient selection uses the original audio.

Continuous haptics use soft background suppression and low-frequency tail
protection. Only accepted transients trigger notches. Curves have a serialized
point budget; existing global-curve attenuation of transients remains.
Original converters and mobile routing are not modified.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import librosa
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks, butter, sosfiltfilt

if __package__:
    from .hybrid_test import PRESETS, apply_quality, load_audio_mono, render_hybrid
    from .normalization_s2h import peak_normalize
    from .wav_to_ahap import rms_envelope, simplify_curve, parameter_curves, active_regions, detect_impacts, make_event
else:
    from hybrid_test import PRESETS, apply_quality, load_audio_mono, render_hybrid
    from normalization_s2h import peak_normalize
    from wav_to_ahap import rms_envelope, simplify_curve, parameter_curves, active_regions, detect_impacts, make_event


def select_transients(original, sr, hop, *, onset_threshold=0.25,
                      min_onset_interval=0.10, max_transients_per_sec=8.0,
                      background_margin_db=12.0, attack_rise_db=6.0):
    """Select impacts on unnormalized audio; no filtering of continuous events.

    Mel energy peaks require local/global background contrast, prominence,
    a fast rise within 40 ms, and positive spectral flux. Nearby peaks are
    merged by find_peaks' minimum distance. Thresholds are experimental;
    no known event count or filename-specific behavior is used.
    """
    original = np.asarray(original, dtype=float)
    original = original - np.mean(original)
    if np.max(np.abs(original)) < 1e-10:
        return np.array([]), np.array([]), {"candidates": 0, "selected": 0}
    mel = librosa.feature.melspectrogram(
        y=original, sr=sr, n_fft=1024, hop_length=hop, n_mels=40,
        fmin=40, fmax=min(12000, sr/2), power=2, center=True)
    energy = 10 * np.log10(np.maximum(mel.sum(axis=0), 1e-20))
    log_mel = librosa.power_to_db(mel, ref=1.0)
    flux = np.maximum(np.diff(log_mel, axis=1, prepend=log_mel[:, :1]), 0).mean(axis=0)
    # Use a 500 ms local context as well as the clip's lower-energy baseline.
    width = max(3, round(.5 * sr / hop) | 1)
    background = np.maximum(np.percentile(energy, 30),
                            percentile_filter(energy, 20, size=width, mode='nearest'))
    peaks, _ = find_peaks(energy, prominence=attack_rise_db,
                         distance=max(1, round(min_onset_interval * sr / hop)))
    selected = []
    lookback = max(1, round(.04 * sr / hop))
    flux_scale = max(float(np.max(flux)), 1e-12)
    for peak in peaks:
        start = max(0, peak-lookback)
        strength = float(np.max(flux[start:peak+1]) / flux_scale)
        if (energy[peak] - background[peak] < background_margin_db
                or energy[peak] - np.min(energy[start:peak+1]) < attack_rise_db
                or strength < onset_threshold):
            continue
        t = float(peak * hop / sr)
        if t < len(original)/sr:
            selected.append((t, strength))
    kept = []
    for second in range(int(np.ceil(len(original)/sr))):
        group = [p for p in selected if second <= p[0] < second+1]
        kept.extend(sorted(group, key=lambda p: p[1], reverse=True)
                    [:max(1, int(max_transients_per_sec))])
    kept.sort()
    return (np.array([p[0] for p in kept]), np.array([p[1] for p in kept]),
            {"candidates": len(peaks), "selected": len(kept),
             "background_margin_db": background_margin_db, "attack_rise_db": attack_rise_db})


def background_envelope(original, sr, times, impact_times, low_cut=350.0):
    """Soft suppression from unnormalized RMS, with conservative confidence.

    A low-dynamic-range clip has no reliable quiet background: leave it alone.
    This heuristic cannot distinguish all steady noise from meaningful sound.
    """
    x = np.asarray(original, dtype=float)
    x = x - x.mean()
    rms = rms_envelope(x, sr, times)
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    spread = float(np.percentile(db, 90) - np.percentile(db, 20))
    confidence = float(np.clip((spread - 8) / 12, 0, 1))
    step = float(np.median(np.diff(times))) if len(times) > 1 else .01
    width = max(3, round(.5 / step) | 1)
    background = percentile_filter(db, 20, size=width, mode='nearest')
    background = np.maximum(background, np.percentile(db, 20))
    u = np.clip((db - background - 3) / 9, 0, 1)
    desired = 1 - confidence * (1 - (0.1 + 0.9 * u*u*(3-2*u)))
    # A strong low-frequency decay must not be mistaken for quiet background.
    low = sosfiltfilt(butter(4, low_cut, fs=sr, output='sos'), x,
                      padlen=min(15, len(x)-1))
    low_rms = rms_envelope(low, sr, times)
    floor = max(float(np.percentile(low_rms, 20)), 1e-10)
    protection = np.zeros_like(times)
    for i, t in enumerate(impact_times):
        peak = float(np.interp(t, times, low_rms))
        full = float(np.interp(t, times, rms))
        if peak / max(full, 1e-10) < .75:
            continue
        limit = impact_times[i+1] if i+1 < len(impact_times) else times[-1]
        idx = np.flatnonzero((times >= t) & (times < limit))
        below = np.flatnonzero(low_rms[idx] < max(floor * 2, peak * .05))
        if len(below):
            idx = idx[:below[0]]
        protection[idx] = np.maximum(protection[idx], np.clip(
            (low_rms[idx] - floor) / max(peak - floor, 1e-10), 0, 1))
    desired = np.maximum(desired, protection)
    envelope = desired.copy()
    for i in range(1, len(times)):
        tau = .005 if desired[i] > envelope[i-1] else .08
        alpha = np.exp(-(times[i]-times[i-1])/tau)
        envelope[i] = alpha * envelope[i-1] + (1-alpha) * desired[i]
    return envelope, {"confidence": confidence, "min_multiplier": float(envelope.min()),
                      "mean_multiplier": float(envelope.mean())}


def compact_curves(times, intensity, sharpness, tolerance, forced, budget=192):
    """Bound serialized points (including shared boundaries), not just error.

    Budget is an experiment setting, NOT a guaranteed Core Haptics device limit.
    Report actual interpolation error when the requested tolerance is relaxed.
    """
    used = tolerance
    keep_forced = forced
    for attempt in range(35):
        ti, vi = simplify_curve(times, intensity, used, keep_forced)
        ts, vs = simplify_curve(times, sharpness, used, keep_forced)
        def pack(pid, t, v):
            # The imported legacy helper drops exactly two points.
            if len(t) == 2:
                return [{"ParameterCurve": {"ParameterID": pid, "Time": float(t[0]),
                    "ParameterCurveControlPoints": [
                        {"Time": float(a-t[0]), "ParameterValue": float(b)} for a, b in zip(t, v)]}}]
            return parameter_curves(pid, t, v)
        curves = pack("HapticIntensityControl", ti, vi) + pack("HapticSharpnessControl", ts, vs)
        count = sum(len(c['ParameterCurve']['ParameterCurveControlPoints']) for c in curves)
        if count <= budget:
            return curves, {"points": count, "curves": len(curves), "tolerance_used": used,
                "forced_points_relaxed": not bool(keep_forced) and bool(forced),
                "max_error": float(max(np.max(np.abs(intensity-np.interp(times,ti,vi))),
                                       np.max(np.abs(sharpness-np.interp(times,ts,vs)))))}
        used = min(1., max(.005, used*1.5))
        if used >= 1:
            keep_forced = ()
    raise ValueError('Unable to reduce AHAP curves within budget')


def convert_wav(in_path, out_path=None, *, preset="ambience", quality="mid",
                strength=1.0, gain=1.0, reference_rms=0.25, control_hz=100.0,
                curve_tolerance=0.01, onset_threshold=0.25, 
                min_onset_interval=0.10, max_transients_per_sec=8.0,
                continuous_threshold=0.045, transient_gain=0.8,
                normalization="peak", audio_peak_headroom_db=1.0,
                background_margin_db=12.0, attack_rise_db=6.0,
                suppress_background=True):

    if not all(np.isfinite([background_margin_db, attack_rise_db])) or min(background_margin_db, attack_rise_db) <= 0:
        raise ValueError("Transient energy margins must be finite and positive")
    if normalization not in ("none", "peak", "sound2hap"):
        raise ValueError("normalization must be none, peak, or sound2hap (peak-only alias)")
    
    if not np.isfinite(audio_peak_headroom_db) or audio_peak_headroom_db < 0:
        raise ValueError("audio_peak_headroom_db must be finite and nonnegative")
    
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
    
    destination = Path(out_path) if out_path else source.with_name(source.stem + "_v2.ahap")
    if destination.suffix.lower() != ".ahap" or destination.resolve() == source.resolve():
        raise ValueError("Output must be a separate .ahap file")

    p = apply_quality(replace(PRESETS[preset]), quality)
    y, sr = load_audio_mono(str(source), p.sr)

    if not len(y) or not np.all(np.isfinite(y)):
        raise ValueError("WAV must contain nonempty, finite audio samples")
    
    original_audio = y.copy()
    normalization_info = {"mode": "peak" if normalization != "none" else "none",
                          "haptic_rms_normalized": False, "reference_rms": reference_rms}
    if normalization != "none":
        # Remove DC before normalization: a constant recording must remain silent.
        y = (y.astype(np.float64) - np.mean(y, dtype=np.float64)).astype(np.float32)
        y, normalization_info["audio"] = peak_normalize(y, audio_peak_headroom_db)
    # No whole-waveform RMS matching or normalization. The remap channel's
    # original relative-level policy remains; reference_rms only scales intensity.
    p.match_rms = False
    p.out_peak = float(np.finfo(np.float32).max)
    duration = len(y) / sr

    padded = np.pad(y, (0, max(0, p.nfft - len(y))))

    features = {}
    rendered, info = render_hybrid(padded, sr, p, strength, 1.0, 1.0, features=features)
    rendered = rendered[:len(y)]
    if np.max(np.abs(y.astype(float) - np.mean(y))) < 1e-12:
        rendered = np.zeros_like(rendered)
    normalization_info["haptic_rms"] = float(np.sqrt(np.mean(rendered.astype(float) ** 2)))

    times = np.r_[np.arange(0, duration, 1.0 / control_hz), duration]

    low_rms= rms_envelope(features["low"][:len(y)], sr, times)
    rem_rms = rms_envelope(features["rem"][:len(y)], sr, times)
    raw_rms= rms_envelope(y, sr, times)

    unbounded_intensity = gain * rms_envelope(rendered, sr, times) / reference_rms
    base = np.clip(unbounded_intensity, 0, 1)
    normalization_info["intensity_saturation_fraction"] = float(np.mean(unbounded_intensity > 1))
    low_ratio=np.clip(low_rms/(raw_rms+1e-8), 0, 1)
    weights = features["A"].astype(np.float64) ** 2

    positions = np.linspace(0, 1, p.ncar)
    mel_texture = (weights * positions[:, None]).sum(axis=0) / (weights.sum(axis=0) + 1e-12)
    frame_times = features["frame_times"]
    texture = np.interp(times, frame_times, mel_texture, left=frame_times[0], right=frame_times[-1]) if len(frame_times) else np.zeros_like(times)

    rem_fraction= rem_rms**2 / (rem_rms**2 + low_rms**2 + 1e-12)
    
    sharpness = np.clip((0.2 * (1 - low_ratio) + 0.8 * texture) * rem_fraction, 0, 1)

    transient_times, transient_strengths, selection_info = select_transients(
        original_audio, sr, p.hop, onset_threshold=onset_threshold,
        min_onset_interval=min_onset_interval, max_transients_per_sec=max_transients_per_sec,
        background_margin_db=background_margin_db, attack_rise_db=attack_rise_db)

    accepted = []
    # Preserve baseline strength mapping and global curve behavior. Selection
    # uses original audio; raw_rms here is measured AFTER peak normalization.
    for t, s in zip(transient_times, transient_strengths):
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
        accepted.append((float(t), float(s), float(attack), float(transient_sharp)))
    
    envelope, suppression_info = background_envelope(
        original_audio, sr, times, [t for t, _, _, _ in accepted], p.low_cut)
    if not suppress_background:
        envelope[:] = 1
        suppression_info["min_multiplier"] = suppression_info["mean_multiplier"] = 1.0
    suppression_info["enabled"] = suppress_background
    continuous = base * envelope
    for t, s, _, _ in accepted:
        notch = np.exp(-0.5 * ((times - t) / 0.035) ** 2)
        continuous *= 1.0 - 0.35 * float(s) * notch

    # Existing mel sqrt epsilon otherwise produces a synthetic floor for exact silence/DC.
    if np.max(np.abs(y.astype(np.float64) - np.mean(y, dtype=np.float64))) < 1e-8:
        continuous[:] = 0
        sharpness[:] = 0
        accepted = []

    fade = min(p.fade_ms / 1000, duration / 2)
    if fade>0:
        continuous *= np.minimum(np.clip(times / fade, 0, 1),
                                 np.clip((duration - times) / fade, 0, 1))
    continuous[0] = continuous[-1] = 0

    pattern = []
    # start = 0.0

    regions = active_regions(times, continuous, threshold=continuous_threshold)
    
    for a, b in regions:
        while a < b - 1e-7:
            length = min(30.0, b - a)
            pattern.append(make_event(a, "HapticContinuous", 1.0, 0.0, length))
            a += length
    
    pattern.extend(make_event(t, "HapticTransient", attack, sharp)
                   for t, _, attack, sharp in accepted)

    curve_info = {"points": 0, "curves": 0}
    if regions:
        # Use zero intensity outside active regions; expand global curves to ensure
        # quiet intervals are truly quiet. Preserve onset-adjacent points through simplification.
        mask = np.zeros(len(times), dtype=bool)
        for a, b in regions:
            mask |= (times >= a) & (times <= b)
        
        continuous[~mask] = 0
        sharpness[~mask] = 0
        forced = [v for t, _, _, _ in accepted for v in (max(0, t - 0.04), t, min(duration, t + 0.04))]
        curves, curve_info = compact_curves(times, continuous, sharpness, curve_tolerance, forced)
        pattern.extend(curves)

    pattern.sort(key=lambda obj: obj.get("Event", obj.get("ParameterCurve"))["Time"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"Version": 1.0, "Pattern": pattern}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination, dict(info, background_suppression=suppression_info, curve_summary=curve_info, transient_selection=selection_info, normalization=normalization_info, duration=duration, control_points=len(times),
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
    parser.add_argument("--reference-rms", type=float, default=0.25)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--curve-tolerance", type=float, default=0.01)
    parser.add_argument("--onset-threshold", type=float, default=0.25)
    parser.add_argument("--min-onset-interval", type=float, default=0.10)
    parser.add_argument("--max-transients-per-sec", type=float, default=8.0)
    parser.add_argument("--continuous-threshold", type=float, default=0.045)
    parser.add_argument("--transient-gain", type=float, default=0.8)
    parser.add_argument("--normalization", choices=["none", "peak", "sound2hap"], default="peak",
                        help="Input peak normalization only; sound2hap is a peak-only compatibility alias")
    parser.add_argument("--audio-peak-headroom-db", type=float, default=1.0)
    parser.add_argument("--background-margin-db", type=float, default=12.0)
    parser.add_argument("--attack-rise-db", type=float, default=6.0)
    parser.add_argument("--no-background-suppression", dest="suppress_background",
                        action="store_false", help="Compare accepted-transient notches without soft suppression")
    args = parser.parse_args()

    try:
        path, info = convert_wav(**vars(args))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Written: {path}\n" + json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
