import argparse
import os
import shutil
import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def require_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        return ffmpeg_path

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        winget_packages = (
            Path(local_app_data)
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        )
        for candidate in winget_packages.glob("ffmpeg-*/*/ffmpeg.exe"):
            return str(candidate)
        for candidate in winget_packages.glob("ffmpeg-*/bin/ffmpeg.exe"):
            return str(candidate)

    raise RuntimeError(
        "ffmpeg was not found. Install ffmpeg and make sure the ffmpeg command "
        "is available in PowerShell. If it was installed with winget, close this "
        "terminal and open a new one, then run: ffmpeg -version"
    )


def separate_audio(input_path, output_path=None, sample_rate=16000, mono=True):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = require_ffmpeg()

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ar",
        str(sample_rate),
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    cmd.extend(["-acodec", "pcm_s16le", str(output_path)])

    subprocess.run(cmd, check=True)
    return output_path


def iter_video_files(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    for path in sorted(input_path.iterdir()):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def parse_args():
    parser = argparse.ArgumentParser(description="Extract wav audio from video files.")
    parser.add_argument("input", help="Input video file or directory, e.g. outputs/clip1.mp4")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output wav file or directory. If omitted, saves next to the input video.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate.")
    parser.add_argument("--stereo", action="store_true", help="Keep stereo instead of mono.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_arg = Path(args.output) if args.output else None

    for video_path in iter_video_files(input_path):
        if output_arg is None:
            wav_path = video_path.with_suffix(".wav")
        elif input_path.is_dir():
            wav_path = output_arg / f"{video_path.stem}.wav"
        else:
            wav_path = output_arg

        saved_path = separate_audio(
            input_path=video_path,
            output_path=wav_path,
            sample_rate=args.sample_rate,
            mono=not args.stereo,
        )
        print(f"Saved wav audio: {saved_path}")


if __name__ == "__main__":
    main()
