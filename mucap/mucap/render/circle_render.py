import os
from os.path import splitext, basename, join, abspath

import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse
from scipy.interpolate import make_interp_spline

from mucap.render.utils import add_audio_to_video


def render_plot_video(
        wav_path,
        tokens,
        output_path,
        progress_bar: bool = True,
        hop=0.25,
        fps=30,
        window_size=7,
        size_effect: bool = True,
        postfix=None,
        dark_mode=True,
        bouncing_mode=False,
        hide_axes=False,  # ⭐ NEW OPTION
        circle_radius_factor: float = 1.0
):

    fig, ax = plt.subplots(figsize=(12, 4))

    # === Load audio ===
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    ax.set_xlim(0, duration)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title("")


    if dark_mode:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.grid(True, color='gray', alpha=0.3)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    else:
        ax.grid(True)

    # ⭐ Hide axes option: keep axes but make them fully transparent
    if hide_axes:
        # Hide ticks
        ax.tick_params(
            axis='both',
            which='both',
            colors=(0, 0, 0, 0),   # transparent tick color
            labelcolor=(0, 0, 0, 0)  # transparent tick labels
        )

        # Hide grid
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_alpha(0)

        # Hide spines (borders)
        for spine in ax.spines.values():
            spine.set_edgecolor((0, 0, 0, 0))

        # Hide xlabel, ylabel, title
        ax.xaxis.label.set_color((0, 0, 0, 0))
        ax.yaxis.label.set_color((0, 0, 0, 0))
        ax.title.set_color((0, 0, 0, 0))

        # Fully hide the grid that appears automatically in dark_mode
        ax.grid(False)

    cursor = ax.axvline(0.0, color='red', linewidth=2)

    n_frames = int(np.ceil(duration * fps))

    # =========================
    # Calculate pitch + dB
    # =========================
    xs, ys, dbs = [], [], []

    for t, tok in tokens:
        s, e = int(t * sr), int((t + hop) * sr)
        seg = y[s:e] if e <= len(y) else y[s:]
        if seg.size == 0:
            continue

        rms = librosa.feature.rms(y=seg).mean()
        db = float(librosa.amplitude_to_db([rms])[0])
        dbs.append(db)

        f0 = librosa.yin(seg, fmin=50, fmax=800, sr=sr)
        pitch = float(np.nanmedian(f0)) if f0.size else 0
        pitch = 0 if np.isnan(pitch) else pitch

        xs.append(t)
        ys.append(pitch)

    # =========================
    # Draw pitch curve
    # =========================
    if len(xs) >= 3:
        xs_np, ys_np = np.array(xs), np.array(ys)
        x_smooth = np.linspace(xs_np.min(), xs_np.max(), 300)
        spline = make_interp_spline(xs_np, ys_np, k=3)
        y_smooth = spline(x_smooth)

        ax.plot(
            x_smooth, y_smooth,
            color='lightgray' if dark_mode else 'gray',
            linestyle='--', alpha=0.0 if bouncing_mode else 0.3, linewidth=2
        )

    if not progress_bar:
        cursor.set_visible(False)

    # ===============================================================
    # 🔥 Create burst circles + compute local pitch
    # ===============================================================
    circle_effects = []

    default_color = np.array([1.0, 1.0, 1.0]) if dark_mode else np.array([0.0, 0.0, 0.0])

    xs_np = np.array(xs)
    ys_np = np.array(ys)

    for onset, pitch, db in zip(xs, ys, dbs):

        # ⭐ Average pitch within ±0.5 seconds
        neighbor_mask = np.abs(xs_np - onset) <= 0.5
        local_pitch_vals = ys_np[neighbor_mask]

        local_pitch = float(np.mean(local_pitch_vals)) if len(local_pitch_vals) else pitch

        # Create circle
        main_ell = Ellipse(
            (onset, pitch),
            width=0,
            height=0,
            fill=True,
            linewidth=2.0,
            edgecolor=(*default_color, 1.0),
            facecolor=(*default_color, 0.8)
        )
        ax.add_patch(main_ell)

        circle_effects.append({
            "onset": onset,
            "shape": main_ell,
            "color": default_color.copy(),
            "db": db,
            "pitch": pitch,
            "local_pitch": local_pitch
        })

    # ===============================================================
    # Animation update
    # ===============================================================
    circle_duration = 0.5
    trail_duration = 4.0
    base_radius = 0.3 * circle_radius_factor * 0.85
    vertical_stretch = 3.5

    bounce_duration = 1
    bounce_strength = 1.42

    def init():
        ax.set_xlim(0, window_size)
        cursor.set_xdata([0])
        return [cursor]

    def update(frame):
        t_now = frame / fps
        effects = []

        # Sliding window
        if t_now < window_size / 2:
            ax.set_xlim(0, window_size)
        elif t_now > duration - window_size / 2:
            ax.set_xlim(duration - window_size, duration)
        else:
            ax.set_xlim(t_now - window_size / 2, t_now + window_size / 2)

        # Progress bar
        if progress_bar:
            cursor.set_xdata([t_now])
            effects.append(cursor)

        # Floor position
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        y_scale = (y1 - y0) / (x1 - x0)
        floor_y = y0 + (y1 - y0) * 0.05

        # === Fix y-limits based on pitch range ===
        if len(ys) > 0:
            pitch_min, pitch_max = min(ys), max(ys)
        else:
            pitch_min, pitch_max = 0, 500

        margin = (pitch_max - pitch_min) * 0.25
        fixed_ymin = pitch_min - margin
        fixed_ymax = pitch_max + margin

        ax.set_ylim(fixed_ymin, fixed_ymax)

        # ====== Circle animation ======
        for eff in circle_effects:
            onset = eff["onset"]
            shape = eff["shape"]
            color = eff["color"]
            db = eff["db"]
            pitch = eff["pitch"]
            local_pitch = eff["local_pitch"]

            dt = t_now - onset

            # ⭐ Jump style
            if bouncing_mode:
                dt_jump = dt

                if 0 <= dt_jump < bounce_duration:
                    phase = np.pi * dt_jump / bounce_duration
                    amp = local_pitch * bounce_strength
                    bounce_y = floor_y + amp * np.sin(phase)

                elif 0 <= dt_jump < bounce_duration + 0.2:
                    t2 = dt_jump - bounce_duration
                    phase = t2 * 8
                    amp = (local_pitch * bounce_strength * 0.3) * np.exp(-4 * t2)
                    bounce_y = floor_y + amp * (-np.sin(phase))

                else:
                    bounce_y = floor_y
            else:
                bounce_y = pitch

            bounce_y = np.clip(bounce_y, fixed_ymin, fixed_ymax)

            # ---- Radius (size_effect ON/OFF) ----
            if size_effect:
                if 0 <= dt < circle_duration:
                    p = dt / circle_duration
                    db_norm = np.clip((db + 50) / 50, 0, 1)
                    db_scale = 0.4 + 1.5 * db_norm
                    radius = base_radius * np.sin(np.pi * p) * db_scale
                    alpha = 1.0 - p * 0.3

                elif 0 <= dt < trail_duration:
                    fade = 1 - dt / trail_duration
                    db_norm = np.clip((db + 50) / 50, 0, 1)
                    db_scale = 0.4 + 1.5 * db_norm
                    radius = base_radius * 0.6 * fade * db_scale
                    alpha = fade * 0.8

                else:
                    shape.width = 0
                    shape.height = 0
                    shape.set_edgecolor((1, 1, 1, 0))
                    shape.set_facecolor((1, 1, 1, 0))
                    effects.append(shape)
                    continue
            else:
                radius = base_radius * 1.2
                alpha = max(0, 1 - dt / trail_duration)

            # ---- Safe alpha / radius ----
            alpha = float(np.clip(alpha, 0.0, 1.0))
            radius = max(radius, 0.0)

            edge_a = float(np.clip(alpha, 0.0, 1.0))
            face_a = float(np.clip(alpha * 0.6, 0.0, 1.0))

            # ---- Apply center ----
            if bouncing_mode:
                shape.center = (onset, bounce_y)
            else:
                shape.center = (onset, pitch)

            # ---- Apply shape ----
            shape.width = radius
            shape.height = radius * y_scale * vertical_stretch
            shape.set_edgecolor((*color, edge_a))
            shape.set_facecolor((*color, face_a))

            effects.append(shape)

        return effects

    # === Save ===
    fig.tight_layout()

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)

    noext = splitext(basename(wav_path))[0]
    out_path = join(output_path, f"{noext}{f'-{postfix}' if postfix else ''}.mp4")

    writer = FFMpegWriter(fps=fps, bitrate=3000)
    ani.save(out_path, writer=writer)

    add_audio_to_video(video_path=abspath(out_path),
                       audio_path=abspath(wav_path))


# Run directly
if __name__ == "__main__":
    tokens = [(float(i / 2), "") for i in range(0, 30)]
    wav_path = "../classic_test.wav"
    out_path = "output_videos"
    os.makedirs(out_path, exist_ok=True)

    render_plot_video(
        wav_path,
        tokens,
        out_path,
        progress_bar=False,
        postfix="bouncy_mode_demo",
        dark_mode=True,
        bouncing_mode=True,
        size_effect=False,
        hide_axes=True
    )
