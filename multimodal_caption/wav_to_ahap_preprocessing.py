"""Sound2Hap-inspired peak audio / RMS vibration normalization for AHAP.

Original wav_to_ahap.py and mobile routing are unchanged.
Audio defaults to -1 dBFS peak (Sound2Hap FreqShift). Hybrid vibration is
whole-clip RMS normalized before envelope extraction. Default target RMS
is reference_rms (0.25), an explicit AHAP adaptation to avoid the much
higher saturation of Sound2Hap's 1/sqrt(2) waveform target. Use
--haptic-target-rms 0.70710678 for that target, or --normalization none
for the original converter behavior. This replaces the earlier -26 dBFS
gain heuristic; it is not per-frame normalization or noise removal.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

if __package__:
    from .hybrid_test import PRESETS, apply_quality, load_audio_mono, render_hybrid
    from .normalization_s2h import peak_normalize, rms_normalize
    from .wav_to_ahap import rms_envelope, simplify_curve, parameter_curves, active_regions, detect_impacts, make_event
else:
    from hybrid_test import PRESETS, apply_quality, load_audio_mono, render_hybrid
    from normalization_s2h import peak_normalize, rms_normalize
    from wav_to_ahap import rms_envelope, simplify_curve, parameter_curves, active_regions, detect_impacts, make_event


def convert_wav(in_path, out_path=None, *, preset="ambience", quality="mid",
                strength=1.0, gain=1.0, reference_rms=0.25, control_hz=100.0,
                curve_tolerance=0.01, onset_threshold=0.25, 
                min_onset_interval=0.10, max_transients_per_sec=8.0,
                continuous_threshold=0.045, transient_gain=0.8,
                normalization="sound2hap", audio_peak_headroom_db=1.0,
                haptic_target_rms=None):

    if normalization not in ("none", "sound2hap"):
        raise ValueError("normalization must be none or sound2hap")
    target_rms = reference_rms if haptic_target_rms is None else haptic_target_rms
    if not np.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("haptic_target_rms must be finite and positive")
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
    
    destination = Path(out_path) if out_path else source.with_name(source.stem + "_preprocessed.ahap")
    if destination.suffix.lower() != ".ahap" or destination.resolve() == source.resolve():
        raise ValueError("Output must be a separate .ahap file")

    p = apply_quality(replace(PRESETS[preset]), quality)
    y, sr = load_audio_mono(str(source), p.sr)

    if not len(y) or not np.all(np.isfinite(y)):
        raise ValueError("WAV must contain nonempty, finite audio samples")
    
    normalization_info = {"mode": normalization}
    if normalization == "sound2hap":
        # Remove DC before normalization: a constant recording must remain silent.
        y = (y.astype(np.float64) - np.mean(y, dtype=np.float64)).astype(np.float32)
        y, normalization_info["audio"] = peak_normalize(y, audio_peak_headroom_db)
        # Avoid original RMS matching and waveform clipping undoing or distorting
        # the dedicated normalization stage. Presets are copied above.
        p.match_rms = False
        p.out_peak = float(np.finfo(np.float32).max)
    duration = len(y) / sr

    padded = np.pad(y, (0, max(0, p.nfft - len(y))))

    features = {}
    rendered, info = render_hybrid(padded, sr, p, strength, 1.0, 1.0, features=features)
    rendered = rendered[:len(y)]
    if normalization == "sound2hap":
        if np.max(np.abs(y)) < 1e-12:
            rendered = np.zeros_like(rendered)
        rendered, normalization_info["haptic"] = rms_normalize(rendered, target_rms)
    
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
    return destination, dict(info, normalization=normalization_info, duration=duration, control_points=len(times),
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
    parser.add_argument("--normalization", choices=["none", "sound2hap"], default="sound2hap",
                        help="Peak-normalize input and RMS-normalize hybrid waveform; none restores legacy behavior")
    parser.add_argument("--audio-peak-headroom-db", type=float, default=1.0)
    parser.add_argument("--haptic-target-rms", type=float, default=None,
                        help="Default: reference-rms (AHAP adaptation); 0.70710678 uses Sound2Hap FreqShift target")
    args = parser.parse_args()

    try:
        path, info = convert_wav(**vars(args))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Written: {path}\n" + json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
