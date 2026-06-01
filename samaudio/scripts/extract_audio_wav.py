import argparse
import glob
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FFMPEG_BIN = ROOT / "tools" / "ffmpeg" / "bin"
if FFMPEG_BIN.exists():
    os.add_dll_directory(str(FFMPEG_BIN))


def find_tool(name: str) -> str:
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


def ffmpeg_exe() -> str:
    return find_tool("ffmpeg")


def resolve_input(path: str) -> Path:
    video = Path(path)
    if not video.is_absolute():
        video = ROOT / video
    if not video.exists():
        raise FileNotFoundError(f"Input video does not exist: {video}")
    return video


def default_output_path(video: Path) -> Path:
    return ROOT / "outputs" / "video_assets" / video.stem / "audio" / f"{video.stem}.wav"


def extract_audio(
    input_video: Path,
    output_wav: Path,
    sample_rate: int,
    channels: int,
    overwrite: bool,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a WAV audio file from an MP4 video."
    )
    parser.add_argument("--input", required=True, help="Input .mp4 video path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .wav path. Defaults to outputs/video_assets/<video>/audio/<video>.wav.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Output sample rate. SAM-Audio uses 48000 Hz.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=[1, 2],
        help="Output channel count.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file.")
    args = parser.parse_args()

    input_video = resolve_input(args.input)
    output_wav = Path(args.output) if args.output else default_output_path(input_video)
    if not output_wav.is_absolute():
        output_wav = ROOT / output_wav

    extract_audio(
        input_video=input_video,
        output_wav=output_wav,
        sample_rate=args.sample_rate,
        channels=args.channels,
        overwrite=args.overwrite,
    )
    print(f"Saved WAV: {output_wav}")


if __name__ == "__main__":
    main()
