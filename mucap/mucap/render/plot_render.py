import os
from os.path import splitext, basename, join, abspath

import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import make_interp_spline

from mucap.render.utils import db_to_weight, load_pretendard_fonts, add_audio_to_video


def render_plot_video(wav_path,
                 tokens,
                 output_path,
                 text_pop_effect: bool = True,
                 progress_bar: bool = True,
                 hop=0.25,
                 font_dir='fonts',
                 fps=30,
                 window_size=7,
                 postfix=None,
                 dark_mode=True):

    fig, ax = plt.subplots(figsize=(12, 4))
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    ax.set_xlim(0, duration)
    # ax.set_ylim(min(ys) - 30, max(ys) + 30)
    ax.set_xlabel("Time (s)")
    ax.set_title("")             # Remove title
    ax.set_ylabel("")            # Remove y-axis label
    ax.set_yticks([])            # Remove y-axis ticks
    # ax.set_title(basename(wav_path))

    if dark_mode:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.grid(True, color='gray', alpha=0.3)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    else:
        ax.grid(True)

    cursor = ax.axvline(0.0, color='red' if dark_mode else 'red', linewidth=2)  # ✅ Updated

    n_frames = int(np.ceil(duration * fps))

    props = load_pretendard_fonts(font_dir)
    xs, ys, text_objs = [], [], []

    # ② Pre-generate text objects
    for t, tok in tokens:
        s, e = int(t * sr), int((t + hop) * sr)
        seg = y[s:e] if e <= len(y) else y[s:]
        if seg.size == 0: continue

        rms = librosa.feature.rms(y=seg).mean()
        db = librosa.amplitude_to_db([rms])[0]
        f0 = librosa.yin(seg, fmin=50, fmax=800, sr=sr)
        pitch = float(np.nanmedian(f0)) if f0.size else 0
        pitch = 0 if np.isnan(pitch) else pitch

        weight = db_to_weight(db)
        font_prop = props.get(weight, props.get("Regular"))
        size = np.clip(np.log1p(db + 60) * 6, 8, 28)

        txt = ax.text(t, pitch, tok,
                      fontproperties=font_prop,
                      fontsize=size,
                      ha="center", va="center",
                      color="white" if dark_mode else "black")  # ✅ Only change needed here
        xs.append(t)
        ys.append(pitch)
        text_objs.append((t, txt, pitch, size))  # onset, object, base_y, base_size

    if len(xs) >= 3:
        xs_np, ys_np = np.array(xs), np.array(ys)
        x_smooth = np.linspace(xs_np.min(), xs_np.max(), 300)
        spline = make_interp_spline(xs_np, ys_np, k=3)
        y_smooth = spline(x_smooth)

        ax.plot(x_smooth, y_smooth,
                color='lightgray' if dark_mode else 'gray',
                linestyle='--', alpha=0.4, linewidth=2)

    if not progress_bar:
        cursor.set_visible(False)

    def init():
        ax.set_xlim(0, window_size)

        cursor.set_xdata([0.0])
        return [cursor]

    jump_duration = 0.3


    # 4. Per-frame update function
    def update(frame):
        effects = list()

        t_now = frame / fps

        # ⏩ Sliding window movement
        if t_now < window_size / 2:
            ax.set_xlim(0, window_size)
        elif t_now > duration - window_size / 2:
            ax.set_xlim(duration - window_size, duration)
        else:
            ax.set_xlim(t_now - window_size / 2, t_now + window_size / 2)

        if progress_bar:
            cursor.set_xdata([t_now])
            effects.append(cursor)
        if text_pop_effect:
            for onset, txt, base_y, base_size in text_objs:
                dt = t_now - onset
                if 0 <= dt < jump_duration:
                    amp = 10 * np.sin(np.pi * dt / jump_duration)
                    txt.set_y(base_y + amp)
                    txt.set_fontsize(base_size + amp * 0.5)
                else:
                    txt.set_y(base_y)
                    txt.set_fontsize(base_size)
            effects.extend([txt for _, txt, _, _ in text_objs])
        return effects

    # 5. Create animation
    fig.tight_layout()

    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=True
    )

    no_extension = splitext(basename(wav_path))[0]
    out_path = join(output_path, f"{no_extension}{f'-{postfix}' if postfix else ''}.mp4")

    writer = FFMpegWriter(fps=30, metadata=dict(artist='ChatGPT'), bitrate=1800)
    ani.save(out_path, writer=writer)

    # 6. Combine audio
    add_audio_to_video(
        video_path=abspath(out_path),
        audio_path=abspath(wav_path),
    )

if __name__ == "__main__":
    tokens = [(float(i), chr(i + 65)) for i in range(0, 15)]  # A-Z
    wav_path = "../classic_test.wav"
    output_path = "output_videos"
    os.makedirs(output_path, exist_ok=True)

    render_plot_video(
        wav_path=wav_path,
        tokens=tokens,
        output_path=output_path,
        text_pop_effect=False,
        progress_bar=True,
        font_dir="fonts",
        postfix="type1",
        window_size=7,
        dark_mode=True,
    )

    render_plot_video(
        wav_path=wav_path,
        tokens=tokens,
        output_path=output_path,
        text_pop_effect=True,
        progress_bar=False,
        font_dir="fonts",
        postfix="type3",
        window_size=7,
        dark_mode=True,
    )

    render_plot_video(
        wav_path=wav_path,
        tokens=tokens,
        output_path=output_path,
        text_pop_effect=False,
        progress_bar=True,
        font_dir="fonts",
        postfix="type1-light",
        window_size=7,
        dark_mode=False,
    )

    render_plot_video(
        wav_path=wav_path,
        tokens=tokens,
        output_path=output_path,
        text_pop_effect=True,
        progress_bar=False,
        font_dir="fonts",
        postfix="type3-light",
        window_size=7,
        dark_mode=False,
    )