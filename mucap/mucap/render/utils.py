import os
import glob
import subprocess

from matplotlib import font_manager as fm
from os.path import dirname

def load_pretendard_fonts(search_dir='fonts'):
    font_files = glob.glob(os.path.join(dirname(__file__), '..', search_dir, "Pretendard-*.otf"))
    if not font_files:
        print("⚠️ Could not find Pretendard fonts.")
        return {}

    props = {}
    for f in font_files:
        name = os.path.splitext(os.path.basename(f))[0].split("-")[1]
        props[name] = fm.FontProperties(fname=f)
    return props


def db_to_weight(db):
    if db < -50:   return "ExtraLight"
    if db < -40:   return "Light"
    if db < -32:   return "Regular"
    if db < -25:   return "Medium"
    if db < -18:   return "SemiBold"
    if db < -10:   return "Bold"
    return "Black"


def add_audio_to_video(video_path, audio_path):
    # Create a temporary output file
    temp_path = video_path.replace(".mp4", "_temp.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-loglevel", "error",
        temp_path

    ]

    try:
        subprocess.run(cmd, check=True)
        os.replace(temp_path, video_path)  # ⬅️ Overwrite the original file
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)