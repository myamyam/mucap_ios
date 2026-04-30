import argparse
import os
import shutil
import subprocess
from pathlib import Path


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


def clip_video(input_path, output_path, start_time, end_time=None, duration=None, reencode=False):
    """
    Clip a video with ffmpeg.

    Times can be ffmpeg-compatible strings such as "00:00:05", "01:02:03.5",
    or seconds such as "5" and "12.5".
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    if end_time is None and duration is None:
        raise ValueError("Either end_time or duration must be provided.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = require_ffmpeg()

    cmd = [ffmpeg_path, "-y", "-ss", str(start_time), "-i", str(input_path)]

    if duration is not None:
        cmd.extend(["-t", str(duration)])
    else:
        cmd.extend(["-to", str(end_time)])

    if reencode:
        cmd.extend(["-c:v", "libx264", "-c:a", "aac"])
    else:
        cmd.extend(["-c", "copy"])

    cmd.append(str(output_path))

    subprocess.run(cmd, check=True)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Clip an mp4 video by time using ffmpeg.")
    parser.add_argument("input", help="Input video path, e.g. data/orchestra.mp4")
    parser.add_argument("output", help="Output video path, e.g. clips/orchestra_5s.mp4")
    parser.add_argument("--start", required=True, help='Start time, e.g. "00:00:10" or "10"')
    parser.add_argument("--end", help='End time, e.g. "00:00:15" or "15"')
    parser.add_argument("--duration", help='Clip duration, e.g. "5" or "00:00:05"')
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode for frame-accurate clipping. Slower but more precise.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    clipped_path = clip_video(
        input_path=args.input,
        output_path=args.output,
        start_time=args.start,
        end_time=args.end,
        duration=args.duration,
        reencode=args.reencode,
    )
    print(f"Saved clipped video: {clipped_path}")


if __name__ == "__main__":
    main()
