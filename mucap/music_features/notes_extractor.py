"""Extract only file_name and notes for existing MuCap audio rows.

This script intentionally leaves every other feature untouched. By default it
uses mucap/music_features.csv as the source list, resolves each audio path, and
writes a two-column CSV that can later replace only the notes column.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_features.feature_extractor import AudioFeatureExtraction


CSV_COLUMNS = ["file_name", "notes"]


def read_source_rows(source_csv: Path) -> list[dict]:
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def discover_wavs(input_dirs: list[Path]) -> list[Path]:
    wavs = []
    for input_dir in input_dirs:
        wavs.extend(path.resolve() for path in input_dir.rglob("*.wav"))
    return sorted(wavs, key=lambda path: str(path).lower())


def build_name_index(wavs: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for wav_path in wavs:
        index.setdefault(wav_path.name, []).append(wav_path)
    return index


def existing_file_names(output_csv: Path) -> set[str]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()
    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["file_name"] for row in reader if row.get("file_name")}


def resolve_audio_path(
    row: dict,
    feature_dir: Path,
    project_root: Path,
    name_index: dict[str, list[Path]],
) -> Path:
    file_name = row.get("file_name", "")
    raw_file = row.get("file", "")
    candidates = []

    if raw_file:
        raw_path = Path(raw_file)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend([
                feature_dir / raw_path,
                project_root / raw_path,
                project_root / "mucap" / raw_path,
            ])

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    matches = name_index.get(file_name, [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Audio file not found for file_name={file_name}")

    preview = ", ".join(str(path) for path in matches[:5])
    raise RuntimeError(
        f"Ambiguous file_name={file_name}; matched {len(matches)} files: {preview}"
    )


def instrument_hint_for(row: dict, wav_path: Path) -> str:
    parts = [
        row.get("genre", ""),
        row.get("dataset", ""),
        wav_path.parent.name,
        wav_path.name,
    ]
    return " ".join(str(part) for part in parts if part)


def append_row(output_csv: Path, file_name: str, notes: list[dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "file_name": file_name,
            "notes": str(notes),
        })


def write_error(error_log: Path, row: dict, exc: Exception) -> None:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "file_name": row.get("file_name"),
        "file": row.get("file"),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with error_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_notes(notes: list[dict]) -> list[str]:
    warnings = []
    if not notes:
        return ["empty_notes"]

    for index, note in enumerate(notes):
        for key in ("onset_time", "pitch", "duration"):
            value = note.get(key)
            if value is None:
                warnings.append(f"note_{index}_missing_{key}")
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                warnings.append(f"note_{index}_non_numeric_{key}")
                continue
            if not math.isfinite(numeric_value):
                warnings.append(f"note_{index}_non_finite_{key}")
            elif key == "duration" and numeric_value <= 0:
                warnings.append(f"note_{index}_non_positive_duration")
    return warnings


def write_warning(warning_log: Path, row: dict, wav_path: Path, warnings: list[str]) -> None:
    warning_log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "file_name": row.get("file_name"),
        "file": row.get("file"),
        "resolved_path": str(wav_path),
        "warnings": warnings,
    }
    with warning_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_rows_from_input_dirs(input_dirs: list[Path]) -> list[dict]:
    wavs = discover_wavs(input_dirs)
    name_counts = Counter(path.name for path in wavs)
    duplicates = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise RuntimeError(
            "Duplicate WAV file names are not safe for file_name-only output: "
            f"{preview}"
        )
    return [{"file_name": path.name, "file": str(path)} for path in wavs]


def main() -> int:
    feature_dir = Path(__file__).resolve().parent
    project_root = feature_dir.parents[1]
    default_source_csv = project_root / "mucap" / "music_features.csv"
    default_dataset_dir = project_root / "mucap" / "dataset"

    parser = argparse.ArgumentParser(
        description="Extract only file_name and notes into a replacement CSV."
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=default_source_csv,
        help="CSV that provides the target row list. Default: mucap/music_features.csv",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=None,
        help="Optional WAV directory mode. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "mucap" / "notes_features.csv",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=project_root / "mucap" / "logs" / "notes_extractor_errors.jsonl",
    )
    parser.add_argument(
        "--warning-log",
        type=Path,
        default=project_root / "mucap" / "logs" / "notes_extractor_warnings.jsonl",
    )
    parser.add_argument(
        "--include-dataset",
        action="append",
        default=None,
        help="Only process rows whose dataset value matches. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=None,
        help="Skip rows whose dataset value matches. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.5,
        help="Pause after each file to reduce sustained CPU heat.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dirs = [path.resolve() for path in args.input_dir or [default_dataset_dir]]
    wavs = discover_wavs(input_dirs)
    name_index = build_name_index(wavs)

    if args.input_dir:
        rows = build_rows_from_input_dirs(input_dirs)
    else:
        source_csv = args.source_csv.resolve()
        if not source_csv.exists():
            raise FileNotFoundError(f"Source CSV not found: {source_csv}")
        rows = read_source_rows(source_csv)

    if args.include_dataset:
        include = set(args.include_dataset)
        rows = [row for row in rows if row.get("dataset") in include]

    if args.exclude_dataset:
        exclude = set(args.exclude_dataset)
        rows = [row for row in rows if row.get("dataset") not in exclude]

    if args.resume:
        done = existing_file_names(args.output_csv.resolve())
        rows = [row for row in rows if row.get("file_name") not in done]

    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    print(f"rows={len(rows)}")
    print(f"wav_search_dirs={'; '.join(str(path) for path in input_dirs)}")
    print(f"output_csv={args.output_csv.resolve()}")
    if args.dry_run:
        for row in rows[:10]:
            wav_path = resolve_audio_path(row, feature_dir, project_root, name_index)
            print(f"would_process={row.get('file_name')} -> {wav_path}")
        print("dry_run=True")
        return 0

    analyzer = AudioFeatureExtraction()
    succeeded = 0
    failed = 0
    started_at = time.monotonic()

    for index, row in enumerate(rows, start=1):
        file_name = row.get("file_name")
        try:
            wav_path = resolve_audio_path(row, feature_dir, project_root, name_index)
            y, sr = analyzer.load_audio(str(wav_path))
            notes = analyzer.extract_notes(
                y,
                sr,
                instrument_hint=instrument_hint_for(row, wav_path),
            )
            warnings = validate_notes(notes)
            if warnings:
                write_warning(args.warning_log.resolve(), row, wav_path, warnings)
            append_row(args.output_csv.resolve(), file_name, notes)
            succeeded += 1
            print(
                f"[{index}/{len(rows)}] written={file_name} notes={len(notes)}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            write_error(args.error_log.resolve(), row, exc)
            print(
                f"[{index}/{len(rows)}] failed={file_name}: {exc}",
                file=sys.stderr,
                flush=True,
            )

        elapsed = time.monotonic() - started_at
        average = elapsed / index
        remaining = average * (len(rows) - index)
        print(
            f"[{index}/{len(rows)}] progress={index / len(rows) * 100:.1f}% "
            f"success={succeeded} failed={failed} eta_hours={remaining / 3600:.2f}",
            flush=True,
        )
        if args.cooldown_seconds > 0 and index < len(rows):
            time.sleep(args.cooldown_seconds)

    print(f"succeeded={succeeded}")
    print(f"failed={failed}")
    print(f"csv_written={args.output_csv.resolve()}")
    if failed:
        print(f"error_log={args.error_log.resolve()}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
