import argparse
import glob
import math
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FFMPEG_BIN = ROOT / "tools" / "ffmpeg" / "bin"
if FFMPEG_BIN.exists():
    os.add_dll_directory(str(FFMPEG_BIN))


def tool_exe(name: str) -> str:
    exe_name = f"{name}.exe"
    bundled = FFMPEG_BIN / exe_name
    if bundled.exists():
        return str(bundled)

    found = shutil.which(name)
    if found:
        return found

    candidates: list[str] = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.extend(
            glob.glob(
                str(
                    Path(local_appdata)
                    / "Microsoft"
                    / "WinGet"
                    / "Packages"
                    / "Gyan.FFmpeg*"
                    / "ffmpeg-*"
                    / "bin"
                    / exe_name
                )
            )
        )
    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if root:
            candidates.extend(glob.glob(str(Path(root) / "GyanDLS" / "ffmpeg" / "bin" / exe_name)))
            candidates.extend(glob.glob(str(Path(root) / "ffmpeg" / "bin" / exe_name)))

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return name


def resolve_input(path: str) -> Path:
    video = Path(path)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise FileNotFoundError(f"Input video does not exist: {video}")
    return video


def default_output_dir(video: Path, interval: float) -> Path:
    label = str(interval).replace(".", "_")
    return ROOT / "outputs" / "video_assets" / video.stem / f"frames_{label}s"


def format_seconds(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def video_duration_seconds(video: Path) -> float:
    command = [
        tool_exe("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def timestamps(duration: float, start: float, end: float | None, interval: float) -> list[float]:
    stop = duration if end is None else min(end, duration)
    if start < 0:
        raise ValueError("--start must be >= 0")
    if interval <= 0:
        raise ValueError("--interval must be > 0")
    if stop <= start:
        raise ValueError("End time must be greater than start time")

    count = math.ceil((stop - start) / interval)
    return [start + i * interval for i in range(count) if start + i * interval < stop]


def extract_one_frame(input_video: Path, output_path: Path, timestamp: float, overwrite: bool) -> None:
    command = [
        tool_exe("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_video),
        "-ss",
        f"{timestamp:.3f}",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def extract_frames(
    input_video: Path,
    output_dir: Path,
    interval: float,
    start: float,
    end: float | None,
    prefix: str,
    overwrite: bool,
) -> None:
    duration = video_duration_seconds(input_video)
    frame_times = timestamps(duration, start, end, interval)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, timestamp in enumerate(frame_times, start=1):
        timestamp_label = format_seconds(timestamp)
        output_path = output_dir / f"{prefix}_{index:04d}_{timestamp_label}s.png"
        extract_one_frame(input_video, output_path, timestamp, overwrite)
        print(f"Saved frame {index:04d}: {output_path}")

    print(f"Done. Saved {len(frame_times)} frames to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract video frames at a fixed interval, defaulting to every 0.2 seconds."
    )
    parser.add_argument("--input", required=True, help="Input .mp4 video path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output frame directory. Defaults to outputs/video_assets/<video>/frames_<interval>s.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Frame interval in seconds.",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds.")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output file prefix. Defaults to the input video filename stem.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files.")
    args = parser.parse_args()

    input_video = resolve_input(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(input_video, args.interval)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    prefix = args.prefix or input_video.stem

    extract_frames(
        input_video=input_video,
        output_dir=output_dir,
        interval=args.interval,
        start=args.start,
        end=args.end,
        prefix=prefix,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
