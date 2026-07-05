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
  body {{ background: #fafafa; margin: 48px; }}
  .viz {{ {base}; line-height: 2.1; word-break: keep-all; }}
  .tok {{ white-space: nowrap; margin-right: 0.45em; }}
</style>
</head>
<body>
<div class="viz">{body}</div>
</body>
</html>"""


def _token_html(token: str, row: pd.Series) -> str:
    chars = list(token)
    n = len(chars)
    spans = []
    for j, ch in enumerate(chars):
        if n > 2 and SUSTAIN_DECAY_FLOOR < 1.0:
            fade = 1.0 - (j / (n - 1)) * (1.0 - SUSTAIN_DECAY_FLOOR)
        else:
            fade = 1.0
        spans.append(
            f'<span style="opacity: {row["opacity"] * fade:.2f}">'
            f"{html.escape(ch)}</span>"
        )
    style = (
        f"color: hsl({row['hue']:.0f} 70% {row['lightness']:.0f}%);"
        f" font-size: {row['size_px']:.1f}px;"
    )
    return f'<span class="tok" style="{style}">{"".join(spans)}</span>'


def render_caption_html(
    captions: list[dict], df: pd.DataFrame, song_row: dict, title: str = ""
) -> str:
    tokens = []
    for cap in captions:
        idx = note_index_at(cap["time"], df) if not df.empty else None
        if idx is None:
            tokens.append(f"<span class=\"tok\">{html.escape(cap['text'])}</span>")
        else:
            tokens.append(_token_html(cap["text"], df.iloc[idx]))
    base = "; ".join(f"{k}: {v}" for k, v in song_style(song_row).items())
    return _PAGE.format(title=html.escape(title), base=base, body="".join(tokens))


def render_freetext_html(
    text: str, df: pd.DataFrame, song_row: dict, title: str = ""
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
            f"color: hsl({row['hue']:.0f} 70% {row['lightness']:.0f}%);"
            f" font-size: {row['size_px']:.1f}px;"
            f" opacity: {row['opacity']:.2f};"
        )
        spans.append(f'<span style="{style}">{escaped}</span>')
    base = "; ".join(f"{k}: {v}" for k, v in song_style(song_row).items())
    return _PAGE.format(title=html.escape(title), base=base, body="".join(spans))


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
) -> Path:
    """One-call hook: caption output -> styled HTML file. Returns the path.

    ``captions`` accepts whatever generate_caption.py holds at the end of a
    run — a list of dicts, (time, text) pairs, a wrapper dict, or a JSON
    file path (see parse_captions).
    """
    file_name = _audio_key(wav_path_or_name)
    notes, song_row = load_song(file_name, notes_csv, features_csv, feature_row)
    df = apply_note_mapping(note_table(notes), CAPTION_MAPPING)
    doc = render_caption_html(parse_captions(captions), df, song_row, title=file_name)

    if out_path is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_OUT_DIR / (Path(file_name).stem + ".html")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
