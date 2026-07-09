
import json
from tqdm import tqdm
from os.path import join, abspath, dirname

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from .utils import (
    add_audio_to_video,
    configure_matplotlib_ffmpeg,
    load_pretendard_fonts,
)

def annotation_render(
    wav_path,
    output_path,
    display_text="🎵 Music Visualization 🎵",
    font_dir='fonts',
    fps=30,
    dark_mode=True,
    postfix="static",
    window_size=3.0
):
    configure_matplotlib_ffmpeg()
    # 1. Load audio
    y, sr = librosa.load(wav_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    n_frames = int(np.ceil(duration * fps))

    # 2. Create figure with 3:1 aspect ratio and larger size
    fig, ax = plt.subplots(figsize=(15, 5))  # 3:1 ratio, larger size
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if dark_mode:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        text_color = "white"
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        text_color = "black"

    # 3. Load font & prepare text
    props = load_pretendard_fonts(font_dir)
    font_prop = props.get("Regular")
    font_size = 24 # Much smaller font size

    # Split text into two halves (token level, not word-based)
    tokens = list(display_text)
    mid = len(tokens) // 2
    first_half = ''.join(tokens[:mid])
    second_half = ''.join(tokens[mid:])

    # Determine trigger time using onset
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    trigger_time = onset_times[mid] if len(onset_times) > mid else duration / 2

    # 4. Create single text object
    text_obj = ax.text(0.5, 0.5, first_half,
                       fontproperties=font_prop,
                       fontsize=font_size,
                       ha="center", va="center",
                       color=text_color,
                       transform=ax.transAxes)

    def init():
        ax.set_xlim(0, 1)
        return [text_obj]

    def update(frame):
        t_now = frame / fps
        if second_half and t_now >= trigger_time:
            text_obj.set_text(second_half)
        else:
            text_obj.set_text(first_half)
        return [text_obj]

    # 5. Animate with sliding window
    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)

    # 6. Save animation
    name = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = os.path.join(output_path, f"{name}-{postfix}.mp4")
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='ChatGPT'), bitrate=1800)
    ani.save(out_path, writer=writer)

    # 7. Merge audio
    add_audio_to_video(video_path=out_path, audio_path=wav_path)
