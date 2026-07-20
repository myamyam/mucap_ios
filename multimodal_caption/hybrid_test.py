#!/usr/bin/env python3
"""
Hybrid haptic renderer (librosa mel-based remapping + low-frequency passthrough)
with an AUTO policy, plus *short commands* via presets.

Goal: avoid long CLI by providing:
- simple preset:  python hybrid_auto.py --in a.wav --out b.wav --preset generic
- optional small overrides: --strength / --low-cut / --quality

Presets set sensible defaults for:
- low-cut, src-flo/src-fhi, ncar, fmin/fmax, nfft/hop, smoothing, auto-remap policy, rms matching.

Dependencies:
  pip install numpy scipy soundfile librosa
Optional:
  ffmpeg installed

Examples:
  # shortest (generic, general)
  python hybrid_auto.py --in raining.wav --out haptic.wav

  # explicitly select preset
  python hybrid_auto.py --in violin.wav --out haptic.wav --preset melodic

  # quick tuning
  python hybrid_auto.py --in input.wav --out haptic.wav --strength 1.2
  python hybrid_auto.py --in input.wav --out haptic.wav --low-cut 300
  python hybrid_auto.py --in input.wav --out haptic.wav --quality high
"""

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "mucap" / "input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mucap" / "output"


# -------------------------
# Presets (short commands)
# -------------------------

@dataclass
class Preset:
    # Core
    sr: int = 48000
    low_cut: float = 250.0

    # Mel analysis for remap (on highpassed audio)
    src_flo: float = 200.0
    src_fhi: float = 12000.0

    # STFT params
    nfft: int = 2048
    hop: int = 256

    # Tactile oscillator bank
    fmin: float = 30.0
    fmax: float = 280.0
    ncar: int = 12

    # Envelope shaping
    amp_lpf: float = 25.0
    gamma: float = 0.85
    gate: float = 0.02

    # Auto policy
    auto_remap: bool = True
    remap_ratio_min: float = 0.15
    remap_ratio_max: float = 1.20
    low_strong_thr: float = 0.55
    low_weak_thr: float = 0.15
    base_rms_floor_frac: float = 0.20

    # Output level policy
    original_mix: float = 0.8
    haptic_mix: float = 0.2
    match_rms: bool = True
    match_rms_factor: float = 1.0

    # Output safety
    out_peak: float = 0.9
    fade_ms: float = 10.0


PRESETS: dict[str, Preset] = {
    # General default: good for mixed content; preserves low; adds remap as needed.
    "generic": Preset(),

    # Environmental / noise-like (rain/wind/ambience): often has plenty of low already.
    # Suppress remap more aggressively to avoid "artificial buzz".
    "ambience": Preset(
        low_cut=350.0,
        ncar=8,
        fmax=220.0,
        remap_ratio_min=0.05,
        remap_ratio_max=0.60,
        low_strong_thr=0.45,
        low_weak_thr=0.12,
    ),

    # Melodic / high-frequency instruments (violin, flute, vocal presence):
    # Stronger remap so violin/flute haptic is more perceptible.
    "melodic": Preset(
        low_cut=200.0,
        ncar=14,
        fmax=300.0,
        remap_ratio_min=0.28,
        remap_ratio_max=2.20,
        low_strong_thr=0.65,
        low_weak_thr=0.15,
        amp_lpf=17.0,
    ),

    # Percussive / transient: favor clearer modulation and less carrier spread.
    "percussive": Preset(
        low_cut=200.0,
        ncar=10,
        fmax=240.0,
        amp_lpf=30.0,
        remap_ratio_min=0.10,
        remap_ratio_max=1.20,
    ),
}


def apply_quality(p: Preset, quality: str) -> Preset:
    """Adjust nfft/hop for speed/quality tradeoff."""
    q = quality.lower()
    if q == "low":
        p.nfft, p.hop = 1024, 256
    elif q == "mid":
        p.nfft, p.hop = 2048, 256
    elif q == "high":
        p.nfft, p.hop = 4096, 512
    return p


# -------------------------
# IO + DSP helpers
# -------------------------

def run_ffmpeg_to_wav(in_path: str, wav_path: str, sr: int) -> None:
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", str(sr), wav_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr[:2000]}")


def load_audio_mono(in_path: str, sr: int) -> tuple[np.ndarray, int]:
    """Try librosa.load; if it fails, fall back to ffmpeg->wav."""
    try:
        y, sr2 = librosa.load(in_path, sr=sr, mono=True)
        return y.astype(np.float32), int(sr2)
    except Exception:
        tmp_wav = str(Path(in_path).with_name("_tmp_in.wav"))
        run_ffmpeg_to_wav(in_path, tmp_wav, sr=sr)
        y, sr2 = sf.read(tmp_wav, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32), int(sr2)


def butter_filter(x: np.ndarray, fs: float, cutoff_hz: float, btype: str, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    wn = min(max(cutoff_hz / nyq, 1e-6), 0.999999)
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x)


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    return butter_filter(x, fs, cutoff_hz, btype="low", order=order)


def highpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    return butter_filter(x, fs, cutoff_hz, btype="high", order=order)


def lowpass_frames(x: np.ndarray, fs_frames: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs_frames
    wn = min(max(cutoff_hz / nyq, 1e-6), 0.99)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def make_log_spaced(freq_lo: float, freq_hi: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(freq_lo), np.log(freq_hi), n))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def hard_limiter(x: np.ndarray, peak: float) -> np.ndarray:
    peak = float(peak)
    return np.clip(x, -peak, peak)


# -------------------------
# Core processing
# -------------------------

def render_hybrid(y_raw: np.ndarray, sr: int, p: Preset, strength: float, gain_low: float, gain_remap: float) -> tuple[np.ndarray, dict]:
    """
    Returns (out, info).
    - strength scales only the remap target ratio (auto policy), not the low passthrough.
    """
    info = {}

    # DC removal only (do not peak-normalize playback path)
    y_raw = (y_raw - float(np.mean(y_raw))).astype(np.float32)

    # Normalized copy for analysis stability
    pk = float(np.max(np.abs(y_raw))) + 1e-12
    y_an = (y_raw / pk).astype(np.float32)

    # Low passthrough + highpass for analysis
    y_low = lowpass(y_raw, fs=sr, cutoff_hz=p.low_cut, order=4).astype(np.float32)
    y_hi_an = highpass(y_an, fs=sr, cutoff_hz=p.low_cut, order=4).astype(np.float32)

    # mel analysis on highpassed
    S = librosa.feature.melspectrogram(
        y=y_hi_an,
        sr=sr,
        n_fft=p.nfft,
        hop_length=p.hop,
        win_length=p.nfft,
        window="hann",
        center=False,
        power=2.0,
        n_mels=int(p.ncar),
        fmin=float(p.src_flo),
        fmax=float(p.src_fhi),
    )
    A = np.sqrt(S + 1e-12).astype(np.float32)
    mx = float(np.max(A)) if A.size else 1.0
    if mx > 0:
        A /= mx

    frame_rate = sr / p.hop
    if p.amp_lpf > 0 and A.shape[1] > 2:
        for k in range(int(p.ncar)):
            A[k, :] = lowpass_frames(A[k, :], fs_frames=frame_rate, cutoff_hz=p.amp_lpf, order=4)

    A = np.clip(A, 0.0, 1.0) ** float(p.gamma)
    A[A < float(p.gate)] = 0.0

    # Synthesis: oscillator bank
    n = len(y_raw)
    t_audio = (np.arange(n, dtype=np.float32) / sr)

    if A.shape[1] == 0:
        remap = np.zeros(n, dtype=np.float32)
    else:
        n_frames = A.shape[1]
        frame_times = (np.arange(n_frames, dtype=np.float32) * (p.hop / sr))
        car_freqs = make_log_spaced(float(p.fmin), float(p.fmax), int(p.ncar))

        remap = np.zeros(n, dtype=np.float32)
        for k in range(int(p.ncar)):
            amp_k = np.interp(
                t_audio,
                frame_times,
                A[k, :],
                left=float(A[k, 0]),
                right=float(A[k, -1]),
            ).astype(np.float32)
            remap += amp_k * np.sin(2.0 * np.pi * float(car_freqs[k]) * t_audio).astype(np.float32)

    remap_pk = float(np.max(np.abs(remap))) + 1e-12
    remap_unit = (remap / remap_pk).astype(np.float32)

    # Gains
    low = (y_low * float(gain_low)).astype(np.float32)
    rem = (remap_unit * float(gain_remap)).astype(np.float32)

    # Auto policy (default on in presets)
    if p.auto_remap:
        r_in = rms(y_raw)
        r_low = rms(low)
        low_strength = float(np.clip(r_low / (r_in + 1e-12), 0.0, 1.0))

        lw = float(p.low_weak_thr)
        ls = float(p.low_strong_thr)
        if ls <= lw:
            ls = lw + 1e-3

        if low_strength <= lw:
            ratio = float(p.remap_ratio_max)
        elif low_strength >= ls:
            ratio = float(p.remap_ratio_min)
        else:
            alpha = (low_strength - lw) / (ls - lw)
            ratio = float(p.remap_ratio_max + (p.remap_ratio_min - p.remap_ratio_max) * alpha)

        # user strength scales ratio (not the low channel)
        ratio *= float(strength)

        base_rms = max(r_low, float(p.base_rms_floor_frac) * r_in)

        r_rem = rms(rem)
        if r_rem > 1e-9:
            target = ratio * base_rms
            rem = (rem * (target / r_rem)).astype(np.float32)

        info.update({
            "low_strength": low_strength,
            "auto_ratio": ratio,
            "base_rms": base_rms,
            "rms_in": r_in,
            "rms_low": r_low,
            "rms_rem_before": r_rem,
            "rms_rem_after": rms(rem),
        })

    # Haptic-only output: keep the tactile low band and add the remapped
    # 30-300 Hz component. Full-band music is played separately through A.
    out = (low + rem).astype(np.float32)

    # RMS match for fair comparison
    if p.match_rms:
        r_in = rms(y_raw)
        r_out = rms(out)
        if r_out > 1e-9:
            out = (out * (float(p.match_rms_factor) * r_in / r_out)).astype(np.float32)

    info["rms_out"] = rms(out)

    # Fade
    fade_samples = max(1, int(sr * (float(p.fade_ms) / 1000.0)))
    if len(out) >= 2 * fade_samples:
        fi = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fo = fi[::-1]
        out[:fade_samples] *= fi
        out[-fade_samples:] *= fo

    # Limiter
    out = hard_limiter(out, peak=float(p.out_peak)).astype(np.float32)
    info["peak_out"] = float(np.max(np.abs(out))) if len(out) else 0.0

    return out, info


# -------------------------
# Single-file conversion (used by both CLI modes)
# -------------------------

def convert_one(
    in_path: str,
    out_path: str,
    preset_name: str,
    quality: str = "mid",
    strength: float = 1.0,
    gain_low: float = 1.0,
    gain_remap: float = 1.0,
    sr_override: int = 0,
    low_cut_override: float = 0.0,
) -> None:
    """Load one file, render with given preset, write to out_path."""
    p = PRESETS[preset_name]
    p = apply_quality(p, quality)
    if sr_override and sr_override > 0:
        p.sr = int(sr_override)
    if low_cut_override and low_cut_override > 0:
        p.low_cut = float(low_cut_override)

    y, sr = load_audio_mono(in_path, p.sr)
    out, info = render_hybrid(
        y_raw=y,
        sr=sr,
        p=p,
        strength=float(strength),
        gain_low=float(gain_low),
        gain_remap=float(gain_remap),
    )
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, out, sr)

    msg = (
        f"Written: {out_path}\n"
        f"preset={preset_name}, quality={quality}, sr={sr}, low_cut={p.low_cut}Hz, "
        f"ncar={p.ncar}, strength={strength}\n"
    )
    if "low_strength" in info:
        msg += f"auto: low_strength={info['low_strength']:.3f}, auto_ratio={info['auto_ratio']:.3f}\n"
    print(msg)


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=None, help="input audio (single-file mode)")
    ap.add_argument("--out", default=None,
                    help="output WAV (default: mucap/output/<stem>/<stem>-haptic.wav)")

    # Batch mode: CSV + folders
    ap.add_argument("--csv", default=None, help="CSV with columns: filename,preset → batch mode")
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                    help="input folder (default: mucap/input)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="output root (default: mucap/output)")

    # Short controls
    ap.add_argument("--preset", default="melodic", choices=sorted(PRESETS.keys()),
                    help="music preset (default: melodic)")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="global remap strength multiplier (1.0=default, >1 stronger, <1 weaker)")
    ap.add_argument("--quality", default="mid", choices=["low", "mid", "high"],
                    help="STFT quality/speed tradeoff")

    # Optional simple overrides (only if you need them)
    ap.add_argument("--low-cut", type=float, default=0.0, help="override low_cut Hz (0=use preset)")
    ap.add_argument("--gain-low", type=float, default=1.0, help="gain for low passthrough")
    ap.add_argument("--gain-remap", type=float, default=1.0, help="gain for remap component")
    ap.add_argument("--sr", type=int, default=0, help="override sample rate (0=use preset)")

    args = ap.parse_args()

    if args.csv:
        # Batch mode: CSV drives (filename, preset) per row
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = Path(args.csv)
        if not csv_path.is_file():
            raise SystemExit(f"CSV not found: {csv_path}")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "filename" not in reader.fieldnames or "preset" not in reader.fieldnames:
                raise SystemExit("CSV must have header: filename,preset")
            rows = list(reader)

        for row in rows:
            filename = (row.get("filename") or "").strip()
            preset_name = (row.get("preset") or "generic").strip().lower()
            if not filename:
                continue
            if preset_name not in PRESETS:
                print(f"Skip {filename}: unknown preset '{preset_name}', use one of {sorted(PRESETS.keys())}")
                continue
            in_path = input_dir / filename
            if not in_path.is_file():
                print(f"Skip {filename}: file not found in {input_dir}")
                continue
            stem = Path(filename).stem
            out_path = output_dir / stem / f"{stem}-haptic.wav"
            convert_one(
                in_path=str(in_path),
                out_path=str(out_path),
                preset_name=preset_name,
                quality=args.quality,
                strength=float(args.strength),
                gain_low=float(args.gain_low),
                gain_remap=float(args.gain_remap),
                sr_override=int(args.sr) if args.sr else 0,
                low_cut_override=float(args.low_cut) if args.low_cut else 0.0,
            )
        return

    # No explicit file/CSV: convert every MuCap input WAV with the music preset.
    if not args.in_path:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        wav_files = sorted(input_dir.rglob("*.wav"))
        if not wav_files:
            raise SystemExit(f"No WAV files found: {input_dir}")
        for in_path in wav_files:
            stem = in_path.stem
            out_path = output_dir / stem / f"{stem}-haptic.wav"
            convert_one(
                in_path=str(in_path),
                out_path=str(out_path),
                preset_name=args.preset,
                quality=args.quality,
                strength=float(args.strength),
                gain_low=float(args.gain_low),
                gain_remap=float(args.gain_remap),
                sr_override=int(args.sr) if args.sr else 0,
                low_cut_override=float(args.low_cut) if args.low_cut else 0.0,
            )
        return

    # Single-file mode
    input_path = Path(args.in_path)
    output_path = (
        Path(args.out)
        if args.out
        else Path(args.output_dir) / input_path.stem / f"{input_path.stem}-haptic.wav"
    )
    p = PRESETS[args.preset]
    p = apply_quality(p, args.quality)
    if args.sr and args.sr > 0:
        p.sr = int(args.sr)
    if args.low_cut and args.low_cut > 0:
        p.low_cut = float(args.low_cut)

    y, sr = load_audio_mono(str(input_path), p.sr)
    out, info = render_hybrid(
        y_raw=y,
        sr=sr,
        p=p,
        strength=float(args.strength),
        gain_low=float(args.gain_low),
        gain_remap=float(args.gain_remap),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, out, sr)

    msg = (
        f"Written: {output_path}\n"
        f"preset={args.preset}, quality={args.quality}, sr={sr}, low_cut={p.low_cut}Hz, "
        f"ncar={p.ncar}, tactile=[{p.fmin}-{p.fmax}]Hz, src=[{p.src_flo}-{p.src_fhi}]Hz, "
        f"strength={args.strength}\n"
    )
    if "low_strength" in info:
        msg += f"auto: low_strength={info['low_strength']:.3f}, auto_ratio={info['auto_ratio']:.3f}\n"
    print(msg)


if __name__ == "__main__":
    main()
