import os, base64, glob, shutil
from os.path import splitext
from pathlib import Path
from pydub import AudioSegment


def find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return found

    candidates = []
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
                    / "ffmpeg.exe"
                )
            )
        )
    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if root:
            candidates.extend(
                glob.glob(str(Path(root) / "GyanDLS" / "ffmpeg" / "bin" / "ffmpeg.exe"))
            )
            candidates.extend(glob.glob(str(Path(root) / "ffmpeg" / "bin" / "ffmpeg.exe")))

    return next((path for path in candidates if Path(path).exists()), None)


FFMPEG = find_ffmpeg()
if FFMPEG:
    AudioSegment.converter = FFMPEG


def wav_to_mp3(wav_path):
    mp3_path = splitext(wav_path)[0] + ".mp3"
    if not os.path.exists(mp3_path):
        if not AudioSegment.converter:
            raise RuntimeError("ffmpeg was not found. Open a new terminal or add ffmpeg.exe to PATH.")
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate="192k")
    return mp3_path


def encode_audio(path):
    # file exists
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
