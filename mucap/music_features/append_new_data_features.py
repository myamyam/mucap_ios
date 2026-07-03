"""Append audio features from an arbitrary WAV folder to music_features.csv.

This is the resumable counterpart of extract_dataset.py for files that do not
follow MuCap's ``*.novocal_N.wav`` naming convention.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_features.feature_extractor_db import AudioFeatureExtraction


CSV_COLUMNS = [
    "file_name",
    "genre",
    "dataset",
    "sample_rate",
    "length",
    "volume",
    "pitch",
    "tempo",
    "attack",
    "envelope",
    "onset_strength",
    "resonance",
    "notes",
    "beat_frames",
    "top5_instruments",
    "file",
]


def discover_wavs(input_dir: Path) -> list[Path]:
    return sorted(
        (path.resolve() for path in input_dir.rglob("*.wav")),
        key=lambda path: str(path).lower(),
    )


def load_existing_names(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    frame = pd.read_csv(csv_path, usecols=["file_name"])
    return set(frame["file_name"].dropna().astype(str))


def csv_file_path(wav_path: Path) -> str:
    feature_dir = Path(__file__).resolve().parent
    return os.path.relpath(wav_path, feature_dir).replace("\\", "/")


def extract_row(
    wav_path: Path,
    csv_path: Path,
    dataset_name: str,
    analyzer: AudioFeatureExtraction,
) -> dict:
    y, sr = analyzer.load_audio(str(wav_path))
    if y is None or len(y) == 0:
        raise RuntimeError("Audio is empty or could not be loaded")

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
    )
    attack = "sharp" if float(np.max(onset_env)) > 0.5 else "soft"
    envelope = "transient" if attack == "sharp" else "sustained"
    onset_strength = float(np.mean(onset_env))
    length = float(librosa.get_duration(y=y, sr=sr))
    resonance = "long" if length > 3.0 else "short"
    notes = analyzer.extract_notes(y, sr, instrument_hint=wav_path.parent.name)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    return {
        "file_name": wav_path.name,
        "genre": wav_path.parent.name,
        "dataset": dataset_name,
        "sample_rate": int(sr),
        "length": length,
        "volume": analyzer.volume(y),
        "pitch": analyzer.pitch(y, sr),
        "tempo": tempo,
        "attack": attack,
        "envelope": envelope,
        "onset_strength": onset_strength,
        "resonance": resonance,
        "notes": notes,
        "beat_frames": beat_frames,
        "top5_instruments": analyzer.predict_instrument_essentia(str(wav_path)),
        "file": csv_file_path(wav_path),
    }


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=CSV_COLUMNS).to_csv(
        csv_path,
        mode="a",
        index=False,
        header=not csv_path.exists() or csv_path.stat().st_size == 0,
    )


def write_error(error_log: Path, wav_path: Path, exc: Exception) -> None:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "file": str(wav_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with error_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Append recursively discovered WAV features to MuCap's CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "new_data",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=project_root / "mucap" / "music_features.csv",
    )
    parser.add_argument("--dataset-name", default="new_data")
    parser.add_argument(
        "--error-log",
        type=Path,
        default=project_root / "mucap" / "logs" / "new_data_feature_errors.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=2.0,
        help="Pause after each file to reduce sustained CPU load.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    csv_path = args.csv_path.resolve()
    error_log = args.error_log.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    wavs = discover_wavs(input_dir)
    existing_names = load_existing_names(csv_path)
    name_counts = Counter(path.name for path in wavs)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicate_names:
        preview = ", ".join(duplicate_names[:5])
        raise RuntimeError(
            f"Duplicate WAV file names are not safe for file_name matching: {preview}"
        )

    pending = [path for path in wavs if path.name not in existing_names]
    if args.limit is not None:
        pending = pending[: max(0, args.limit)]

    print(f"input_dir={input_dir}")
    print(f"csv_path={csv_path}")
    print(f"wav_count={len(wavs)}")
    print(f"already_in_csv={len(wavs) - len([p for p in wavs if p.name not in existing_names])}")
    print(f"pending={len(pending)}")
    print(f"dataset_name={args.dataset_name}")
    if args.dry_run:
        for path in pending[:10]:
            print(f"would_process={path.parent.name}/{path.name}")
        print("dry_run=True")
        return 0

    analyzer = AudioFeatureExtraction()
    succeeded = 0
    failed = 0
    started_at = time.monotonic()
    for index, wav_path in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] extracting={wav_path}")
        file_started_at = time.monotonic()
        try:
            row = extract_row(wav_path, csv_path, args.dataset_name, analyzer)
            append_row(csv_path, row)
            existing_names.add(wav_path.name)
            succeeded += 1
            print(
                f"[{index}/{len(pending)}] written={wav_path.name} "
                f"notes={len(row['notes'])}"
            )
        except Exception as exc:
            failed += 1
            write_error(error_log, wav_path, exc)
            print(
                f"[{index}/{len(pending)}] failed={wav_path.name}: {exc}",
                file=sys.stderr,
            )
        finally:
            elapsed = time.monotonic() - started_at
            average = elapsed / index
            remaining = average * (len(pending) - index)
            file_elapsed = time.monotonic() - file_started_at
            print(
                f"[{index}/{len(pending)}] progress={index / len(pending) * 100:.1f}% "
                f"file_seconds={file_elapsed:.1f} "
                f"success={succeeded} failed={failed} "
                f"eta_hours={remaining / 3600:.2f}",
                flush=True,
            )
            if args.cooldown_seconds > 0 and index < len(pending):
                time.sleep(args.cooldown_seconds)

    print(f"succeeded={succeeded}")
    print(f"failed={failed}")
    print(f"csv_written={csv_path}")
    if failed:
        print(f"error_log={error_log}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
