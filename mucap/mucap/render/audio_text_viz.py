"""mucap/render/audio_text_viz_.py — map audio features onto caption text visuals.

Placement: this file lives in mucap/mucap/render/ and follows the repo's
conventions:

    * notes parsing delegates to mucap.data_io.parse_notes_field when the
      package is importable (single source of truth for the repr format),
      with a self-contained fallback for standalone use.
    * file_name matching strips the "fixed_" prefix, same as
      caption_skeleton_patch._audio_key.
    * default CSV paths resolve to the project data dir (mucap/), where
      music_features.csv and notes_features.csv live in this workspace.

Integration (mirrors the caption_skeleton_patch style):

    # generate_caption.py — after the LLM captions are parsed:
    from mucap.render.audio_text_viz import visualize_captions
    html_path = visualize_captions(wav_path, captions)

    # generate_caption_batch.py — inside the per-file loop, same call.

CLI:
    python -m mucap.render.audio_text_viz \
        --file-name some_song.wav \
        --captions-json captions.json \
        --out render/out/some_song.html
    (--notes-csv / --features-csv default to the package-dir CSVs)
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import re
import shutil
import sys
from os.path import basename
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:  # repo-internal parser is the source of truth when available
    from mucap.data_io import parse_notes_field as _repo_parse_notes
except ImportError:
    _repo_parse_notes = None

try:
    from mucap.caption_skeleton_patch import (
        _caption_behavior_from_wav,
        _connected_texture_style,
        _is_confident_string_context,
    )
except ImportError:
    _caption_behavior_from_wav = None
    _connected_texture_style = None
    _is_confident_string_context = None

_DATA_DIR = Path(__file__).resolve().parents[2]  # mucap/
DEFAULT_NOTES_CSV = _DATA_DIR / "notes_features.csv"
DEFAULT_FEATURES_CSV = _DATA_DIR / "music_features.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

# ---------------------------------------------------------------------------
# 1. Loading / parsing
# ---------------------------------------------------------------------------

_NP_REPR = re.compile(r"np\.(?:float|int)\d*\(([^()]*)\)")


def _audio_key(path_or_name: str) -> str:
    """Same file_name convention as caption_skeleton_patch."""
    fname = basename(str(path_or_name))
    return fname[len("fixed_"):] if fname.startswith("fixed_") else fname


def _fallback_parse_notes(raw: object) -> list[dict]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [n for n in raw if isinstance(n, dict)]
    cleaned = _NP_REPR.sub(r"\1", str(raw))
    try:
        notes = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return []
    return [n for n in notes if isinstance(n, dict)] if isinstance(notes, list) else []


def parse_notes(raw: object) -> list[dict]:
    """Notes cell -> clean, sorted list of {onset_time, pitch, duration}."""
    notes = _repo_parse_notes(raw) if _repo_parse_notes else _fallback_parse_notes(raw)
    out = []
    for note in notes or []:
        if not isinstance(note, dict):
            continue
        try:
            out.append(
                {
                    "onset_time": float(note["onset_time"]),
                    "pitch": float(note["pitch"]),
                    "duration": float(note["duration"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda n: n["onset_time"])
    return out


def parse_captions(raw: object) -> list[dict]:
    """Normalize caption input to [{"time": float, "text": str}, ...].

    Accepts: a JSON path/string; {"captions": [...]} wrapper; list of dicts
    with flexible keys (time/timestamp/onset/onset_time/start x
    text/caption/token/syllable); or list of (time, text) pairs.
    """
    if isinstance(raw, (str, Path)):
        raw = json.loads(Path(raw).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("captions", raw.get("results", []))
    if not isinstance(raw, list):
        raise ValueError("captions must be a list of {time, text} objects")

    out = []
    for item in raw:
        time_val, text_val = None, None
        if isinstance(item, dict):
            time_val = next(
                (item[k] for k in ("time", "timestamp", "onset", "onset_time", "start")
                 if k in item), None)
            text_val = next(
                (item[k] for k in ("text", "caption", "token", "syllable")
                 if k in item), None)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            a, b = item[0], item[1]
            time_val, text_val = (a, b) if isinstance(b, str) else (b, a)
        if time_val is None or not text_val:
            continue
        try:
            out.append({"time": float(time_val), "text": str(text_val)})
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda c: c["time"])
    return out


def load_song(
    file_name: str,
    notes_csv: Path = DEFAULT_NOTES_CSV,
    features_csv: Path | None = DEFAULT_FEATURES_CSV,
    feature_row: dict | None = None,
) -> tuple[list[dict], dict]:
    """Return (notes, song_row) for one file, keyed by _audio_key(file_name).

    Falls back to music_features.csv's own notes column when the file is
    absent from notes_features.csv (e.g. rows never re-extracted). If the
    runtime pipeline passes an extracted feature row, use its notes as the
    final fallback for new input audio that is not present in either CSV.
    """
    key = _audio_key(file_name)
    notes: list[dict] = []

    if Path(notes_csv).exists():
        notes_df = pd.read_csv(notes_csv, encoding="utf-8-sig")
        match = notes_df.loc[notes_df["file_name"].map(_audio_key) == key]
        if not match.empty:
            notes = parse_notes(match.iloc[0]["notes"])

    song_row: dict = {}
    if features_csv is not None and Path(features_csv).exists():
        feat_df = pd.read_csv(features_csv, encoding="utf-8-sig")
        feat_match = feat_df.loc[feat_df["file_name"].map(_audio_key) == key]
        if not feat_match.empty:
            song_row = feat_match.iloc[0].to_dict()
            if not notes and "notes" in song_row:
                notes = parse_notes(song_row.get("notes"))

    if feature_row:
        if not song_row:
            song_row = dict(feature_row)
        else:
            merged = dict(feature_row)
            merged.update(song_row)
            song_row = merged
        if not notes and "notes" in feature_row:
            notes = parse_notes(feature_row.get("notes"))

    if not notes:
        raise KeyError(f"No parseable notes for file_name={file_name}")
    return notes, song_row


# ---------------------------------------------------------------------------
# 2. Note-level feature engineering
# ---------------------------------------------------------------------------


def pitch_to_semitones(pitch: np.ndarray) -> np.ndarray:
    pitch = np.asarray(pitch, dtype=float)
    valid = pitch[pitch > 0]
    if valid.size and np.median(valid) > 130:
        out = np.full_like(pitch, np.nan)
        mask = pitch > 0
        out[mask] = 12.0 * np.log2(pitch[mask] / 440.0)
        return out
    return pitch


def note_table(notes: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(notes)
    if df.empty:
        return df
    df["pitch_st"] = pitch_to_semitones(df["pitch"].to_numpy())
    onsets = df["onset_time"].to_numpy()
    df["density"] = np.array(
        [np.sum(np.abs(onsets - t) <= 1.0) / 2.0 for t in onsets], dtype=float
    )
    return df


def robust_norm(
    values: np.ndarray, lo: float = 5, hi: float = 95, gamma: float = 1.0
) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, 0.5)
    mask = np.isfinite(v)
    if mask.sum() < 2:
        return out
    p_lo, p_hi = np.percentile(v[mask], [lo, hi])
    if p_hi - p_lo < 1e-9:
        return out
    x = np.clip((v[mask] - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    out[mask] = x**gamma
    return out


# ---------------------------------------------------------------------------
# 3. Mapping config
# ---------------------------------------------------------------------------

CAPTION_MAPPING: dict[str, dict] = {
    "hue": {"feature": "pitch_st", "range": (225.0, 20.0), "gamma": 1.0},
    "lightness": {"feature": "pitch_st", "range": (35.0, 62.0), "gamma": 1.0},
    "size_px": {"feature": "pitch_st", "range": (34.0, 17.0), "gamma": 1.0},
    "opacity": {"feature": "density", "range": (0.55, 1.0), "gamma": 1.0},
}

FREETEXT_MAPPING: dict[str, dict] = {
    "hue": {"feature": "pitch_st", "range": (225.0, 20.0), "gamma": 1.0},
    "lightness": {"feature": "pitch_st", "range": (35.0, 62.0), "gamma": 1.0},
    "size_px": {"feature": "duration", "range": (15.0, 40.0), "gamma": 0.7},
    "opacity": {"feature": "density", "range": (0.40, 1.0), "gamma": 1.0},
}

SUSTAIN_DECAY_FLOOR = 0.45
CAPTION_FONT_SCALE = 0.85
INTERNAL_PITCH_OFFSET_PX = 20.0
INTERNAL_COLOR_NORM_DELTA = 0.25
INTERNAL_OPACITY_NORM_DELTA = 0.25
INTERNAL_OPACITY_RANGE = (1.0, 0.68)

VISUAL_PALETTES: dict[str, dict[str, tuple[float, float] | float]] = {
    "piano-clear": {"hue": (224.0, 198.0), "lightness": (30.0, 70.0), "sat": 76.0},
    "string-like": {"hue": (282.0, 322.0), "lightness": (30.0, 72.0), "sat": 76.0},
    "wind-like": {"hue": (178.0, 164.0), "lightness": (32.0, 72.0), "sat": 74.0},
    "generic-sustain": {"hue": (32.0, 56.0), "lightness": (36.0, 74.0), "sat": 82.0},
}


def song_style(song_row: dict) -> dict:
    style: dict[str, str] = {}
    attack = str(song_row.get("attack", "")).lower()
    if attack == "sharp":
        style["font-weight"] = "700"
    elif attack == "soft":
        style["font-weight"] = "300"

    envelope = str(song_row.get("envelope", "")).lower()
    style["font-family"] = (
        "Georgia, serif" if envelope == "sustained" else "Helvetica, Arial, sans-serif"
    )

    tempo = song_row.get("tempo")
    try:
        tempo_val = float(re.sub(r"[\[\]]", "", str(tempo)).split()[0])
    except (TypeError, ValueError, IndexError):
        tempo_val = None
    if tempo_val:
        spacing = float(np.interp(tempo_val, [60, 180], [0.20, -0.02]))
        style["letter-spacing"] = f"{spacing:.2f}em"
    return style


def apply_note_mapping(df: pd.DataFrame, mapping: dict[str, dict]) -> pd.DataFrame:
    for attr, cfg in mapping.items():
        norm = robust_norm(df[cfg["feature"]].to_numpy(), gamma=cfg["gamma"])
        lo, hi = cfg["range"]
        df[attr] = lo + norm * (hi - lo)
    return df


def _pitch_norm(row: pd.Series, df: pd.DataFrame) -> float:
    if df.empty or "pitch_st" not in df:
        return 0.5
    values = df["pitch_st"].to_numpy(dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return 0.5
    lo, hi = np.percentile(values[mask], [5, 95])
    if hi - lo < 1e-9:
        return 0.5
    try:
        pitch_value = float(row.get("pitch_st", np.nan))
    except (TypeError, ValueError):
        return 0.5
    if not np.isfinite(pitch_value):
        return 0.5
    return float(np.clip((pitch_value - lo) / (hi - lo), 0.0, 1.0))


def _visual_mode_from_context(
    wav_path_or_name: str | Path,
    row: dict,
) -> str:
    behavior = None
    if _caption_behavior_from_wav is not None:
        try:
            behavior = _caption_behavior_from_wav(str(wav_path_or_name), row)
        except Exception:
            behavior = None

    string_mode = False
    if _is_confident_string_context is not None:
        try:
            string_mode = bool(_is_confident_string_context(row))
        except Exception:
            string_mode = False

    texture_style = None
    if _connected_texture_style is not None:
        try:
            texture_style = _connected_texture_style(row, behavior, string_mode)
        except Exception:
            texture_style = None

    top5 = _parse_visual_top5(row.get("top5_instruments", ""))
    bowed_strings = {"violin", "viola", "cello", "doublebass", "strings"}
    wind_names = {
        "flute", "recorder", "clarinet", "oboe", "bassoon", "saxophone",
        "sax", "trumpet", "trombone", "horn", "french horn", "tuba",
    }
    piano_names = {"piano", "keyboard", "electricpiano", "rhodes"}
    voice_names = {"voice", "vocal", "vocals", "singing"}
    guitar_names = {"guitar", "electricguitar", "acousticguitar", "classicalguitar"}

    def best(names: set[str]) -> float:
        return max((score for name, score in top5 if name in names), default=0.0)

    string_score = best(bowed_strings)
    string_sum = sum(score for name, score in top5 if name in bowed_strings)
    string_count = sum(1 for name, _ in top5 if name in bowed_strings)
    wind_score = best(wind_names)
    piano_score = best(piano_names)
    voice_score = best(voice_names)
    guitar_score = best(guitar_names)
    non_string_best = max(piano_score, voice_score, guitar_score, wind_score)
    behavior_mode = behavior.get("mode") if behavior else None

    if texture_style == "string-like":
        return "string-like"

    if texture_style == "wind-like":
        return "wind-like"

    if (
        piano_score >= 0.80
        and behavior_mode in {"clear-attack", "mixed"}
        and piano_score >= max(string_score, wind_score) + 0.15
    ):
        return "piano-clear"

    if behavior_mode == "sustained-texture":
        return "generic-sustain"
    return "piano-clear"


def _parse_visual_top5(raw: object) -> list[tuple[str, float]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(raw, (list, tuple)):
        return []

    out = []
    for index, item in enumerate(raw):
        try:
            if isinstance(item, dict):
                name = item.get("name") or item.get("instrument") or item.get("label")
                score = item.get("score") or item.get("prob") or item.get("probability")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                name, score = item[0], item[1]
            else:
                name, score = item, 1.0 / (index + 1)
            cleaned = re.sub(r"[^a-z0-9]+", "", str(name).lower())
            out.append((cleaned, float(score)))
        except (TypeError, ValueError):
            continue
    return out


def _palette_color(norm: float, visual_mode: str) -> tuple[float, float, float]:
    palette = VISUAL_PALETTES.get(visual_mode, VISUAL_PALETTES["piano-clear"])
    hue_lo, hue_hi = palette["hue"]  # type: ignore[misc]
    light_lo, light_hi = palette["lightness"]  # type: ignore[misc]
    sat = float(palette["sat"])
    hue = float(hue_lo + norm * (hue_hi - hue_lo))
    lightness = float(light_lo + norm * (light_hi - light_lo))
    return hue, sat, lightness


def _color_style(
    row: pd.Series,
    df: pd.DataFrame,
    visual_mode: str,
    base_norm: float | None = None,
) -> str:
    norm = _pitch_norm(row, df)
    if base_norm is not None:
        norm = float(
            np.clip(
                base_norm + np.clip(norm - base_norm, -INTERNAL_COLOR_NORM_DELTA,
                                    INTERNAL_COLOR_NORM_DELTA),
                0.0,
                1.0,
            )
        )
    hue, sat, lightness = _palette_color(norm, visual_mode)
    return f"hsl({hue:.0f} {sat:.0f}% {lightness:.0f}%)"


def _opacity_style(
    row: pd.Series,
    df: pd.DataFrame,
    base_norm: float | None = None,
) -> float:
    norm = _pitch_norm(row, df)
    if base_norm is not None:
        norm = float(
            np.clip(
                base_norm + np.clip(norm - base_norm, -INTERNAL_OPACITY_NORM_DELTA,
                                    INTERNAL_OPACITY_NORM_DELTA),
                0.0,
                1.0,
            )
        )
    lo, hi = INTERNAL_OPACITY_RANGE
    return float(lo + norm * (hi - lo))


# ---------------------------------------------------------------------------
# 4. Alignment
# ---------------------------------------------------------------------------


def note_index_at(t: float, df: pd.DataFrame) -> int:
    onsets = df["onset_time"].to_numpy()
    ends = onsets + df["duration"].to_numpy()
    active = np.where((onsets <= t) & (t < ends))[0]
    if active.size:
        return int(active[0])
    return int(np.argmin(np.abs(onsets - t)))


def assign_notes_to_chars(text: str, df: pd.DataFrame) -> list[int | None]:
    if df.empty:
        return [None] * len(text)
    span = float((df["onset_time"] + df["duration"]).max())
    visible = [i for i, ch in enumerate(text) if not ch.isspace()]
    n = max(len(visible), 1)
    assign: list[int | None] = [None] * len(text)
    for rank, i in enumerate(visible):
        assign[i] = note_index_at((rank / n) * span, df)
    return assign


# ---------------------------------------------------------------------------
# 5. Rendering
# ---------------------------------------------------------------------------

_PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background: #fff; color: #fff; margin: 36px; }}
  .player {{ margin-top: 18px; text-align: center; }}
  audio {{ width: min(720px, 100%); }}
  .stage {{
    position: relative;
    height: 420px;
    border: 1px solid #111;
    background: #fff;
    overflow: hidden;
  }}
  .grid {{
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(to right, rgba(0,0,0,0.18) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(0,0,0,0.08) 1px, transparent 1px);
    background-size: 12.5% 100%, 100% 50%;
  }}
  .cursor {{
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 3px;
    background: red;
    transform: translateX(-1.5px);
    z-index: 4;
  }}
  .caption-token {{
    position: absolute;
    top: 50%;
    transform: translate(-50%, calc(-50% + var(--token-y, 0px)));
    white-space: nowrap;
    z-index: 3;
    {base};
  }}
  .caption-pop {{
    display: inline-block;
    transform: translateY(0) scale(1);
    transition: transform 90ms ease-out;
    transform-origin: center bottom;
  }}
  .caption-token.is-popping .caption-pop {{
    transform: translateY(-18px) scale(1.12);
  }}
  .caption-char {{
    display: inline-block;
  }}
  .time-axis {{
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 34px;
    border-top: 1px solid rgba(0,0,0,0.45);
    color: #000;
    font: 13px/1.2 Arial, sans-serif;
  }}
  .tick {{
    position: absolute;
    bottom: 8px;
    transform: translateX(-50%);
  }}
</style>
</head>
<body>
<div class="stage" data-duration="{duration}">
  <div class="grid"></div>
  <div class="cursor"></div>
  {body}
  <div class="time-axis">{ticks}</div>
</div>
{audio}
<script>
(function () {{
  const audio = document.querySelector("audio");
  const stage = document.querySelector(".stage");
  const cursor = document.querySelector(".cursor");
  const tokens = Array.from(document.querySelectorAll(".caption-token"));
  const ticks = Array.from(document.querySelectorAll(".tick"));
  const fallbackDuration = parseFloat(stage.dataset.duration || "1") || 1;
  const windowSize = 7.0;
  let raf = null;

  function duration() {{
    return audio && Number.isFinite(audio.duration) && audio.duration > 0
      ? audio.duration
      : fallbackDuration;
  }}

  function viewStart(current, dur) {{
    const visible = Math.min(windowSize, dur);
    if (dur <= visible) return 0;
    if (current < visible / 2) return 0;
    if (current > dur - visible / 2) return dur - visible;
    return current - visible / 2;
  }}

  function updateTimeline() {{
    const dur = duration();
    const visible = Math.min(windowSize, dur);
    const current = audio ? audio.currentTime : 0;
    const start = viewStart(current, dur);
    const end = start + visible;
    tokens.forEach((token) => {{
      const t = parseFloat(token.dataset.time || "0") || 0;
      const inside = t >= start - 0.001 && t <= end + 0.001;
      token.style.display = inside ? "inline-block" : "none";
      if (inside) {{
        token.style.left = `${{Math.max(0, Math.min(100, ((t - start) / visible) * 100))}}%`;
      }}
      const popAge = current - t;
      token.classList.toggle("is-popping", popAge >= 0 && popAge <= 0.32);
    }});
    ticks.forEach((tick) => {{
      const t = parseFloat(tick.dataset.time || "0") || 0;
      const inside = t >= start - 0.001 && t <= end + 0.001;
      tick.style.display = inside ? "inline-block" : "none";
      if (inside) {{
        tick.style.left = `${{Math.max(0, Math.min(100, ((t - start) / visible) * 100))}}%`;
      }}
    }});
    cursor.style.left = `${{Math.max(0, Math.min(100, ((current - start) / visible) * 100))}}%`;
  }}

  function loop() {{
    updateTimeline();
    if (audio && !audio.paused && !audio.ended) {{
      raf = requestAnimationFrame(loop);
    }}
  }}

  updateTimeline();
  if (audio) {{
    audio.addEventListener("loadedmetadata", () => {{
      updateTimeline();
    }});
    audio.addEventListener("timeupdate", updateTimeline);
    audio.addEventListener("seeked", updateTimeline);
    audio.addEventListener("play", () => {{
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(loop);
    }});
    audio.addEventListener("pause", () => {{
      if (raf) cancelAnimationFrame(raf);
      updateTimeline();
    }});
    audio.addEventListener("ended", () => {{
      if (raf) cancelAnimationFrame(raf);
      updateTimeline();
    }});
  }}
}})();
</script>
</body>
</html>"""


def _prepare_audio_html(
    audio_path: str | Path | None,
    out_path: Path | None,
    copy_audio: bool = True,
) -> str:
    if not audio_path:
        return ""

    src = Path(audio_path)
    if not src.exists():
        return ""

    audio_src = src
    if copy_audio and out_path is not None:
        dst = Path(out_path).parent / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        audio_src = Path(src.name)

    return (
        '<div class="player">'
        f'<audio controls preload="metadata" src="{html.escape(str(audio_src))}"></audio>'
        "</div>"
    )


def _token_html(token: str, row: pd.Series, df: pd.DataFrame, visual_mode: str) -> str:
    chars = list(token)
    n = len(chars)
    spans = []
    base_norm = _pitch_norm(row, df)
    for j, ch in enumerate(chars):
        if n > 2 and SUSTAIN_DECAY_FLOOR < 1.0:
            fade = 1.0 - (j / (n - 1)) * (1.0 - SUSTAIN_DECAY_FLOOR)
        else:
            fade = 1.0
        _ = fade
        spans.append(
            f'<span style="opacity: 1.00; '
            f'color: {_color_style(row, df, visual_mode, base_norm)};">'
            f"{html.escape(ch)}</span>"
        )
    style = (
        f"color: {_color_style(row, df, visual_mode)};"
        f" font-size: {row['size_px'] * CAPTION_FONT_SCALE:.1f}px;"
    )
    return f'<span class="tok" style="{style}">{"".join(spans)}</span>'


def _pitch_offset_px(row: pd.Series, df: pd.DataFrame, max_offset: float = 115.0) -> float:
    if df.empty or "pitch_st" not in df:
        return 0.0
    values = df["pitch_st"].to_numpy(dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return 0.0
    lo, hi = np.percentile(values[mask], [5, 95])
    if hi - lo < 1e-9:
        return 0.0
    try:
        pitch_value = float(row.get("pitch_st", np.nan))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(pitch_value):
        return 0.0
    norm = np.clip((pitch_value - lo) / (hi - lo), 0.0, 1.0)
    return (0.5 - norm) * 2.0 * max_offset


def _row_at_time(t: float, df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.iloc[note_index_at(t, df)]


def _caption_end_time(index: int, captions: list[dict], full_duration: float) -> float:
    start = float(captions[index]["time"])
    if index + 1 < len(captions):
        end = float(captions[index + 1]["time"])
    else:
        end = full_duration
    return max(end, start + 0.05)


def _timeline_token_html(
    token: str,
    row: pd.Series,
    timestamp: float,
    end_time: float,
    df: pd.DataFrame,
    visual_mode: str,
) -> str:
    chars = list(token)
    n = len(chars)
    spans = []
    token_offset_y = _pitch_offset_px(row, df)
    base_norm = _pitch_norm(row, df)
    for j, ch in enumerate(chars):
        if n:
            char_time = timestamp + ((j + 0.5) / n) * max(end_time - timestamp, 0.05)
        else:
            char_time = timestamp
        char_row = _row_at_time(char_time, df)
        if char_row is None:
            char_row = row
        if n > 2 and SUSTAIN_DECAY_FLOOR < 1.0:
            fade = 1.0 - (j / (n - 1)) * (1.0 - SUSTAIN_DECAY_FLOOR)
        else:
            fade = 1.0
        _ = fade
        raw_offset_y = _pitch_offset_px(char_row, df)
        offset_y = float(
            np.clip(
                raw_offset_y - token_offset_y,
                -INTERNAL_PITCH_OFFSET_PX,
                INTERNAL_PITCH_OFFSET_PX,
            )
        )
        opacity = _opacity_style(char_row, df, base_norm)
        spans.append(
            f'<span class="caption-char" style="'
            f'opacity: {opacity:.2f}; '
            f'color: {_color_style(char_row, df, visual_mode, base_norm)}; '
            f'transform: translateY({offset_y:.1f}px);">'
            f"{html.escape(ch)}</span>"
        )
    style = (
        f"color: {_color_style(row, df, visual_mode)};"
        f" font-size: {row['size_px'] * CAPTION_FONT_SCALE:.1f}px;"
    )
    return (
        f'<span class="caption-token" data-time="{timestamp:.4f}" '
        f'style="--token-y: {token_offset_y:.1f}px; {style}">'
        f'<span class="caption-pop">{"".join(spans)}</span></span>'
    )


def _duration_from_captions(captions: list[dict], df: pd.DataFrame) -> float:
    values = [float(cap["time"]) for cap in captions if "time" in cap]
    if not df.empty:
        values.append(float((df["onset_time"] + df["duration"]).max()))
    return max(values) if values else 1.0


def _axis_ticks(duration: float) -> str:
    duration = max(float(duration), 1.0)
    step = 1.0 if duration <= 20 else 5.0
    ticks = []
    t = 0.0
    while t <= duration + 1e-6:
        left = min(100.0, max(0.0, (t / duration) * 100.0))
        label = str(int(round(t)))
        ticks.append(
            f'<span class="tick" data-time="{t:.4f}" '
            f'style="left: {left:.3f}%">{label}</span>'
        )
        t += step
    return "".join(ticks)


def render_caption_html(
    captions: list[dict],
    df: pd.DataFrame,
    song_row: dict,
    title: str = "",
    audio_html: str = "",
    visual_mode: str = "piano-clear",
) -> str:
    tokens = []
    parsed_captions = list(captions)
    duration = _duration_from_captions(parsed_captions, df)
    for index, cap in enumerate(parsed_captions):
        idx = note_index_at(cap["time"], df) if not df.empty else None
        if idx is None:
            tokens.append(
                f'<span class="caption-token" data-time="{float(cap["time"]):.4f}">'
                f"{html.escape(cap['text'])}</span>"
            )
        else:
            end_time = _caption_end_time(index, parsed_captions, duration)
            tokens.append(
                _timeline_token_html(
                    cap["text"],
                    df.iloc[idx],
                    cap["time"],
                    end_time,
                    df,
                    visual_mode,
                )
            )
    base = "; ".join(f"{k}: {v}" for k, v in song_style(song_row).items())
    return _PAGE.format(
        title=html.escape(title),
        base=base,
        audio=audio_html,
        body="".join(tokens),
        duration=f"{duration:.4f}",
        ticks=_axis_ticks(duration),
    )


def render_freetext_html(
    text: str,
    df: pd.DataFrame,
    song_row: dict,
    title: str = "",
    audio_html: str = "",
    visual_mode: str = "piano-clear",
) -> str:
    assign = assign_notes_to_chars(text, df)
    spans = []
    for ch, idx in zip(text, assign):
        if ch == "\n":
            spans.append("<br>")
            continue
        escaped = html.escape(ch)
        if idx is None:
            spans.append(f"<span>{escaped}</span>")
            continue
        row = df.iloc[idx]
        style = (
            f"color: {_color_style(row, df, visual_mode)};"
            f" font-size: {row['size_px'] * CAPTION_FONT_SCALE:.1f}px;"
            f" opacity: 1.00;"
        )
        spans.append(f'<span style="{style}">{escaped}</span>')
    base = "; ".join(f"{k}: {v}" for k, v in song_style(song_row).items())
    return _PAGE.format(
        title=html.escape(title),
        base=base,
        audio=audio_html,
        body="".join(spans),
        duration="1.0000",
        ticks=_axis_ticks(1.0),
    )


# ---------------------------------------------------------------------------
# 6. Public API for generate_caption.py / generate_caption_batch.py
# ---------------------------------------------------------------------------


def visualize_captions(
    wav_path_or_name: str,
    captions: object,
    out_path: Path | None = None,
    notes_csv: Path = DEFAULT_NOTES_CSV,
    features_csv: Path | None = DEFAULT_FEATURES_CSV,
    feature_row: dict | None = None,
    audio_path: str | Path | None = None,
    copy_audio: bool = True,
) -> Path:
    """One-call hook: caption output -> styled HTML file. Returns the path.

    ``captions`` accepts whatever generate_caption.py holds at the end of a
    run — a list of dicts, (time, text) pairs, a wrapper dict, or a JSON
    file path (see parse_captions).
    """
    file_name = _audio_key(wav_path_or_name)
    notes, song_row = load_song(file_name, notes_csv, features_csv, feature_row)
    df = apply_note_mapping(note_table(notes), CAPTION_MAPPING)
    context_row = dict(song_row)
    if feature_row:
        context_row.update(feature_row)
    visual_mode = _visual_mode_from_context(audio_path or wav_path_or_name, context_row)

    if out_path is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_OUT_DIR / (Path(file_name).stem + ".html")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_html = _prepare_audio_html(audio_path, out_path, copy_audio)
    doc = render_caption_html(
        parse_captions(captions),
        df,
        song_row,
        title=file_name,
        audio_html=audio_html,
        visual_mode=visual_mode,
    )
    out_path.write_text(doc, encoding="utf-8")
    return out_path


_PATCH_GUIDE = r"""
=== 시각화 연결 가이드 ===

1) generate_caption.py — 캡션 파싱 직후 한 줄:

   from mucap.render.audio_text_viz import visualize_captions
   ...
   captions = <LLM 출력 파싱 결과>          # [{time, text}, ...] 또는 유사 형태
   html_path = visualize_captions(wav_path, captions)

2) generate_caption_batch.py — 파일 루프 안에서 동일 호출.
   출력은 mucap/render/out/<file_name>.html 에 쌓인다.

3) CSV 경로는 mucap/mucap/ 의 notes_features.csv, music_features.csv 를
   자동으로 찾는다. 다른 위치면 notes_csv=/features_csv= 인자로 지정.
"""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes-csv", type=Path, default=DEFAULT_NOTES_CSV)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--file-name", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--captions-json", type=Path)
    group.add_argument("--text", help="Free text (uniform-spread fallback mode)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.captions_json:
        out = visualize_captions(
            args.file_name, args.captions_json,
            out_path=args.out, notes_csv=args.notes_csv,
            features_csv=args.features_csv,
        )
        print(f"mode=caption out={out}")
    else:
        notes, song_row = load_song(args.file_name, args.notes_csv, args.features_csv)
        df = apply_note_mapping(note_table(notes), FREETEXT_MAPPING)
        doc = render_freetext_html(args.text, df, song_row, title=args.file_name)
        out = args.out or Path("out.html")
        out.write_text(doc, encoding="utf-8")
        print(f"mode=freetext chars={len(args.text)} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
