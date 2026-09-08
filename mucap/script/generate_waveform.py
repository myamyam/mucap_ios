#!/usr/bin/env python3
"""Render time-domain waveform and frequency spectrogram plots to a PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


MAX_FREQUENCY_HZ = 22_050


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate waveform and spectrogram plots from an audio file."
    )
    parser.add_argument("audio", type=Path, help="Input audio file, such as cello_2.wav.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path "
            "(default: <audio-directory>/waveform/<audio_stem>_waveform.png)."
        ),
    )
    parser.add_argument("--width", type=float, default=14.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=8.0, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=160, help="Output resolution in DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if args.width <= 0 or args.height <= 0 or args.dpi <= 0:
        raise ValueError("Width, height, and DPI must be positive.")

    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")

    channel_count = samples.shape[1]
    waveform = samples.mean(axis=1) if channel_count > 1 else samples[:, 0]
    times = np.arange(samples.shape[0], dtype=np.float64) / sample_rate
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else audio_path.parent / "waveform" / f"{audio_path.stem}_waveform.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, (waveform_axis, frequency_axis) = plt.subplots(
        2,
        1,
        figsize=(args.width, args.height),
        sharex=True,
        layout="constrained",
    )

    waveform_axis.plot(times, waveform, linewidth=0.55, alpha=0.85)
    waveform_axis.set_xlabel("Time (seconds)")
    waveform_axis.set_ylabel("Amplitude")
    waveform_axis.set_ylim(-1.0, 1.0)
    waveform_axis.tick_params(axis="x", labelbottom=True)

    _, _, _, spectrogram_image = frequency_axis.specgram(
        waveform,
        NFFT=2048,
        Fs=sample_rate,
        noverlap=1536,
        mode="magnitude",
        scale="dB",
        cmap="magma",
    )
    frequency_axis.set_xlabel("Time (seconds)")
    frequency_axis.set_ylabel("Frequency (Hz)")
    frequency_axis.set_ylim(0.0, MAX_FREQUENCY_HZ)

    duration = samples.shape[0] / sample_rate
    waveform_axis.set_xlim(0.0, duration)
    frequency_axis.set_xlim(0.0, duration)
    colorbar = figure.colorbar(
        spectrogram_image,
        ax=frequency_axis,
        pad=0.015,
    )
    colorbar.set_label("Magnitude (dB)")

    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)

    print(f"Input: {audio_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Input channels: {channel_count}")
    print("Plotted signal: channel mean" if channel_count > 1 else "Plotted signal: mono channel")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Waveform and spectrogram: {output_path}")


if __name__ == "__main__":
    main()
