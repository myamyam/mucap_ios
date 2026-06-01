import argparse
import bisect
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*([^\s\[]+)")


@dataclass(frozen=True)
class CaptionToken:
    time_sec: float
    text: str
    end_sec: float


@dataclass(frozen=True)
class MatchRow:
    time_sec: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    is_sounding: bool
    confidence: float
    audio_rms: float


@dataclass(frozen=True)
class CaptionStyle:
    name: str
    fill: tuple[int, int, int]
    stroke_fill: tuple[int, int, int]
    shadow_fill: tuple[int, int, int, int]
    accent_fill: tuple[int, int, int, int]
    font_candidates: tuple[str, ...] = ()
    font_scale: float = 1.0
    stroke_scale: float = 1.0
    shadow_offset: tuple[float, float] = (0.16, 0.18)
    scale_x: float = 1.0
    scale_y: float = 1.0
    shear_x: float = 0.0
    extrusion_fill: tuple[int, int, int, int] | None = None
    extrusion_offset: tuple[float, float] = (0.0, 0.0)
    burst_scale: float = 1.0
    angle: float = 0.0
    accent: str = "none"


DEFAULT_CAPTION_STYLE = CaptionStyle(
    name="default",
    fill=(255, 238, 82),
    stroke_fill=(2, 35, 76),
    shadow_fill=(0, 10, 30, 190),
    accent_fill=(255, 70, 214, 210),
    font_candidates=(r"C:\Windows\Fonts\malgunbd.ttf",),
    font_scale=1.08,
    stroke_scale=1.4,
    shadow_offset=(0.14, 0.18),
    angle=-3.0,
    accent="speed",
)

CAPTION_STYLE_RULES: list[tuple[re.Pattern[str], CaptionStyle]] = [
    (
        re.compile("(\uc73c\uc544\uc545|\uc73c\uc545|\uc544\uc545|\ud06c\uc545|\ud06c\uc559|\uadf8\uc545|\ud5c9|\uc545+)"),
        CaptionStyle(
            name="battle_shout",
            fill=(255, 241, 210),
            stroke_fill=(84, 8, 18),
            shadow_fill=(0, 0, 0, 230),
            accent_fill=(218, 24, 42, 235),
            font_candidates=(
                str(ROOT / "assets/fonts/East_Sea_Dokdo/EastSeaDokdo-Regular.ttf"),
                str(ROOT / "assets/fonts/Kirang_Haerang/KirangHaerang-Regular.ttf"),
                r"C:\Windows\Fonts\NotoSerifKR-VF.ttf",
                r"C:\Windows\Fonts\batang.ttc",
                r"C:\Windows\Fonts\malgunbd.ttf",
            ),
            font_scale=1.38,
            stroke_scale=2.55,
            shadow_offset=(0.30, 0.34),
            scale_x=1.26,
            scale_y=0.92,
            shear_x=-0.18,
            extrusion_fill=(220, 21, 28, 235),
            extrusion_offset=(0.34, 0.22),
            burst_scale=1.65,
            angle=-10.0,
            accent="burst",
        ),
    ),
    (
        re.compile("(\uc288\uc6b1|\uc289|\uc27d|\ucd2c\uc545|\ucc59|\uce74\uc559|\uce89|\uc9f1|\ud0a4\uc789)"),
        CaptionStyle(
            name="slash",
            fill=(235, 248, 255),
            stroke_fill=(5, 22, 42),
            shadow_fill=(75, 0, 9, 220),
            accent_fill=(222, 26, 44, 220),
            font_candidates=(
                str(ROOT / "assets/fonts/Nanum_Brush_Script/NanumBrushScript-Regular.ttf"),
                str(ROOT / "assets/fonts/Nanum_Pen_Script/NanumPenScript-Regular.ttf"),
                r"C:\Windows\Fonts\malgunsl.ttf",
                r"C:\Windows\Fonts\NotoSansKR-VF.ttf",
                r"C:\Windows\Fonts\gulim.ttc",
            ),
            font_scale=1.22,
            stroke_scale=1.65,
            shadow_offset=(0.28, 0.20),
            scale_x=1.32,
            scale_y=0.82,
            shear_x=-0.26,
            extrusion_fill=(18, 18, 22, 220),
            extrusion_offset=(0.20, 0.12),
            burst_scale=1.15,
            angle=-14.0,
            accent="speed",
        ),
    ),
    (
        re.compile("(\ucfe0\uc6c5|\uad11|\uad49|\ucfe1|\ucfe0|\ub4dc\uc775|\ub4dd)"),
        CaptionStyle(
            name="heavy_impact",
            fill=(255, 205, 64),
            stroke_fill=(37, 13, 9),
            shadow_fill=(96, 0, 0, 230),
            accent_fill=(215, 26, 36, 235),
            font_candidates=(
                str(ROOT / "assets/fonts/Bagel_Fat_One/BagelFatOne-Regular.ttf"),
                str(ROOT / "assets/fonts/Dongle/Dongle-Bold.ttf"),
                r"C:\Windows\Fonts\malgunbd.ttf",
                r"C:\Windows\Fonts\NotoSansKR-VF.ttf",
            ),
            font_scale=1.42,
            stroke_scale=2.65,
            shadow_offset=(0.34, 0.34),
            scale_x=1.20,
            scale_y=1.02,
            shear_x=0.10,
            extrusion_fill=(210, 24, 24, 235),
            extrusion_offset=(0.36, 0.28),
            burst_scale=1.85,
            angle=7.0,
            accent="burst",
        ),
    ),
    (
        re.compile("(\ucc0c|\uc9c0\uc789|\ube44\ube59|\ub9c1|\ub9ac\ub9c1|\ub760\ub9ac\ub9c1)"),
        CaptionStyle(
            name="electronic",
            fill=(255, 70, 214),
            stroke_fill=(93, 244, 175),
            shadow_fill=(3, 38, 87, 215),
            accent_fill=(93, 244, 175, 230),
            font_candidates=(
                str(ROOT / "assets/fonts/Gaegu/Gaegu-Bold.ttf"),
                str(ROOT / "assets/fonts/Dongle/Dongle-Regular.ttf"),
                r"C:\Windows\Fonts\gulim.ttc",
                r"C:\Windows\Fonts\HanSantteutDotum-Bold.ttf",
                r"C:\Windows\Fonts\malgunbd.ttf",
            ),
            font_scale=1.18,
            stroke_scale=1.65,
            shadow_offset=(0.22, 0.24),
            scale_x=1.08,
            scale_y=0.94,
            shear_x=-0.08,
            burst_scale=1.05,
            angle=-6.0,
            accent="spark",
        ),
    ),
    (
        re.compile("(\ud31d|\ud0d5|\ud0c1|\ub531|\ud37d|\uc9dd|\ucc29|\ud321|\ud30d)"),
        CaptionStyle(
            name="impact",
            fill=(255, 78, 50),
            stroke_fill=(34, 10, 6),
            shadow_fill=(0, 0, 0, 230),
            accent_fill=(236, 28, 42, 230),
            font_candidates=(
                str(ROOT / "assets/fonts/Kirang_Haerang/KirangHaerang-Regular.ttf"),
                str(ROOT / "assets/fonts/Bagel_Fat_One/BagelFatOne-Regular.ttf"),
                r"C:\Windows\Fonts\NGULIM.TTF",
                r"C:\Windows\Fonts\gulim.ttc",
                r"C:\Windows\Fonts\malgunbd.ttf",
            ),
            font_scale=1.32,
            stroke_scale=2.45,
            shadow_offset=(0.28, 0.30),
            scale_x=1.28,
            scale_y=0.88,
            shear_x=-0.20,
            extrusion_fill=(205, 18, 28, 230),
            extrusion_offset=(0.30, 0.20),
            burst_scale=1.55,
            angle=-8.0,
            accent="burst",
        ),
    ),
    (
        re.compile("(\ub465|\ub450\uc6c5|\uc6c5|\ubd95|\ubd80\uc6c5|\ucfe0\uc6c5)"),
        CaptionStyle(
            name="bass",
            fill=(58, 33, 28),
            stroke_fill=(255, 226, 175),
            shadow_fill=(0, 12, 32, 205),
            accent_fill=(215, 26, 36, 210),
            font_candidates=(
                str(ROOT / "assets/fonts/Song_Myung/SongMyung-Regular.ttf"),
                str(ROOT / "assets/fonts/Diphylleia/Diphylleia-Regular.ttf"),
                r"C:\Windows\Fonts\batang.ttc",
                r"C:\Windows\Fonts\NotoSerifKR-VF.ttf",
                r"C:\Windows\Fonts\malgunbd.ttf",
            ),
            font_scale=1.22,
            stroke_scale=1.7,
            shadow_offset=(0.24, 0.26),
            angle=3.0,
            accent="none",
        ),
    ),
    (
        re.compile("(\ub530|\ud0c0|\ub2e4\ub2e5|\ud30c\ubc15|\ub9ac\ub4ec|\ub9ac\ub9ac)"),
        CaptionStyle(
            name="rhythm",
            fill=(255, 214, 92),
            stroke_fill=(43, 12, 8),
            shadow_fill=(116, 0, 0, 190),
            accent_fill=(230, 28, 42, 210),
            font_candidates=(
                str(ROOT / "assets/fonts/Dongle/Dongle-Bold.ttf"),
                str(ROOT / "assets/fonts/Gaegu/Gaegu-Bold.ttf"),
                r"C:\Windows\Fonts\HanSantteutDotum-Bold.ttf",
                r"C:\Windows\Fonts\malgunbd.ttf",
            ),
            font_scale=1.12,
            stroke_scale=1.45,
            shadow_offset=(0.18, 0.16),
            angle=5.0,
            accent="speed",
        ),
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return default
    return float(str(value))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def object_id_from_mucap_json(path: Path) -> str:
    stem = path.stem
    if "-" in stem:
        return stem.split("-", 1)[0]
    return stem


def parse_caption_tokens(caption: str, default_duration: float, max_duration: float) -> list[CaptionToken]:
    raw_tokens = [
        (float(match.group(1)), match.group(2).strip())
        for match in TOKEN_RE.finditer(caption or "")
        if match.group(2).strip()
    ]
    tokens: list[CaptionToken] = []
    for index, (time_sec, text) in enumerate(raw_tokens):
        next_time = raw_tokens[index + 1][0] if index + 1 < len(raw_tokens) else time_sec + default_duration
        end_sec = min(time_sec + max_duration, max(time_sec + default_duration, next_time))
        tokens.append(CaptionToken(time_sec=time_sec, text=text, end_sec=end_sec))
    return tokens


def load_mucap_captions(
    mucap_output: Path,
    default_duration: float,
    max_duration: float,
) -> dict[str, list[CaptionToken]]:
    if not mucap_output.exists():
        raise FileNotFoundError(f"MuCap output path does not exist: {mucap_output}")

    json_paths = sorted(mucap_output.rglob("*.json")) if mucap_output.is_dir() else [mucap_output]
    captions: dict[str, list[CaptionToken]] = {}
    for path in json_paths:
        data = load_json(path)
        output = data.get("output")
        if not isinstance(output, dict):
            continue
        caption = output.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            continue
        object_id = object_id_from_mucap_json(path)
        tokens = parse_caption_tokens(caption, default_duration, max_duration)
        if tokens:
            captions[object_id] = tokens

    if not captions:
        raise ValueError(f"No MuCap caption JSON files with output.caption found in {mucap_output}")
    return captions


def load_matching_csv(path: Path) -> dict[str, list[MatchRow]]:
    tracks: dict[str, list[MatchRow]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        required = {"object_id", "time_sec", "x_min", "y_min", "x_max", "y_max", "center_x", "center_y"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for row in reader:
            if any(str(row.get(name, "")).strip() == "" for name in required):
                continue
            object_id = str(row["object_id"])
            tracks.setdefault(object_id, []).append(
                MatchRow(
                    time_sec=parse_float(row.get("time_sec")),
                    bbox=(
                        parse_float(row.get("x_min")),
                        parse_float(row.get("y_min")),
                        parse_float(row.get("x_max")),
                        parse_float(row.get("y_max")),
                    ),
                    center=(
                        parse_float(row.get("center_x")),
                        parse_float(row.get("center_y")),
                    ),
                    is_sounding=parse_bool(row.get("is_sounding", "true")),
                    confidence=parse_float(row.get("confidence"), 0.0),
                    audio_rms=parse_float(row.get("audio_rms"), 0.0),
                )
            )

    for object_id in list(tracks):
        tracks[object_id] = sorted(tracks[object_id], key=lambda item: item.time_sec)
    if not tracks:
        raise ValueError(f"No usable rows found in matching CSV: {path}")
    return tracks


def load_scene_cut_times(path: Path) -> list[float]:
    data = load_json(path)
    raw_cuts = data.get("cuts", [])
    if not isinstance(raw_cuts, list):
        raise ValueError(f"scene_cuts.json must contain a list field named 'cuts': {path}")

    cuts: list[float] = []
    for item in raw_cuts:
        if isinstance(item, dict):
            value = item.get("time_sec")
        else:
            value = item
        if value is None:
            continue
        cuts.append(float(value))
    return sorted(set(cut for cut in cuts if cut >= 0.0))


def first_cut_between(scene_cuts: list[float], start_sec: float, end_sec: float) -> float | None:
    if not scene_cuts or end_sec <= start_sec:
        return None
    index = bisect.bisect_right(scene_cuts, start_sec + 1e-6)
    if index < len(scene_cuts) and scene_cuts[index] < end_sec - 1e-6:
        return scene_cuts[index]
    return None


def near_scene_cut(scene_cuts: list[float], time_sec: float, margin_sec: float) -> bool:
    if not scene_cuts or margin_sec <= 0.0:
        return False
    index = bisect.bisect_left(scene_cuts, time_sec)
    candidates = []
    if index < len(scene_cuts):
        candidates.append(scene_cuts[index])
    if index > 0:
        candidates.append(scene_cuts[index - 1])
    return any(abs(time_sec - cut) <= margin_sec for cut in candidates)


def scene_index_for_time(scene_cuts: list[float], time_sec: float) -> int:
    return bisect.bisect_right(scene_cuts, time_sec)


def valid_position_row(
    row: MatchRow,
    frame_width: int,
    frame_height: int,
    min_width: float,
    min_height: float,
    max_area_fraction: float,
    max_height_fraction: float,
) -> bool:
    x1, y1, x2, y2 = row.bbox
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < min_width or box_height < min_height:
        return False
    frame_area = max(1.0, float(frame_width * frame_height))
    if (box_width * box_height) / frame_area > max_area_fraction:
        return False
    if box_height / max(1.0, float(frame_height)) > max_height_fraction:
        return False
    return True


def median_match_row(rows: list[MatchRow], scene_index: int) -> MatchRow:
    bboxes = np.asarray([row.bbox for row in rows], dtype=np.float64)
    centers = np.asarray([row.center for row in rows], dtype=np.float64)
    bbox = tuple(float(value) for value in np.median(bboxes, axis=0))
    center = tuple(float(value) for value in np.median(centers, axis=0))
    return MatchRow(
        time_sec=float(scene_index),
        bbox=bbox,
        center=center,
        is_sounding=True,
        confidence=float(np.median([row.confidence for row in rows])),
        audio_rms=float(np.median([row.audio_rms for row in rows])),
    )


def build_fixed_scene_tracks(
    tracks: dict[str, list[MatchRow]],
    scene_cuts: list[float],
    frame_width: int,
    frame_height: int,
    min_width: float,
    min_height: float,
    max_area_fraction: float,
    max_height_fraction: float,
    min_samples: int,
) -> dict[str, dict[int, MatchRow]]:
    fixed_tracks: dict[str, dict[int, MatchRow]] = {}
    for object_id, rows in tracks.items():
        grouped: dict[int, list[MatchRow]] = {}
        for row in rows:
            if not valid_position_row(
                row,
                frame_width=frame_width,
                frame_height=frame_height,
                min_width=min_width,
                min_height=min_height,
                max_area_fraction=max_area_fraction,
                max_height_fraction=max_height_fraction,
            ):
                continue
            scene_index = scene_index_for_time(scene_cuts, row.time_sec)
            grouped.setdefault(scene_index, []).append(row)

        fixed_tracks[object_id] = {}
        for scene_index, scene_rows in grouped.items():
            if len(scene_rows) >= min_samples:
                fixed_tracks[object_id][scene_index] = median_match_row(scene_rows, scene_index)
    return fixed_tracks


def nearest_row(rows: list[MatchRow], time_sec: float) -> MatchRow:
    times = [row.time_sec for row in rows]
    index = bisect.bisect_left(times, time_sec)
    if index <= 0:
        return rows[0]
    if index >= len(rows):
        return rows[-1]
    before = rows[index - 1]
    after = rows[index]
    return before if abs(time_sec - before.time_sec) <= abs(after.time_sec - time_sec) else after


def interpolate_row(
    rows: list[MatchRow],
    time_sec: float,
    scene_cuts: list[float] | None = None,
) -> MatchRow:
    times = [row.time_sec for row in rows]
    index = bisect.bisect_left(times, time_sec)
    if index <= 0:
        return rows[0]
    if index >= len(rows):
        return rows[-1]

    before = rows[index - 1]
    after = rows[index]
    span = after.time_sec - before.time_sec
    if span <= 1e-9:
        return before

    cut_sec = first_cut_between(scene_cuts or [], before.time_sec, after.time_sec)
    if cut_sec is not None:
        return before if time_sec < cut_sec else after

    alpha = (time_sec - before.time_sec) / span

    def lerp(a: float, b: float) -> float:
        return a + (b - a) * alpha

    return MatchRow(
        time_sec=time_sec,
        bbox=tuple(lerp(a, b) for a, b in zip(before.bbox, after.bbox, strict=True)),
        center=tuple(lerp(a, b) for a, b in zip(before.center, after.center, strict=True)),
        is_sounding=nearest_row(rows, time_sec).is_sounding,
        confidence=nearest_row(rows, time_sec).confidence,
        audio_rms=lerp(before.audio_rms, after.audio_rms),
    )


def position_row_for_time(
    object_id: str,
    rows: list[MatchRow],
    time_sec: float,
    position_mode: str,
    interpolate: bool,
    scene_cuts: list[float],
    fixed_scene_tracks: dict[str, dict[int, MatchRow]],
) -> MatchRow | None:
    if position_mode == "dynamic":
        return (
            interpolate_row(rows, time_sec, scene_cuts=scene_cuts)
            if interpolate
            else nearest_row(rows, time_sec)
        )
    if position_mode == "fixed-by-scene":
        scene_index = scene_index_for_time(scene_cuts, time_sec)
        return fixed_scene_tracks.get(object_id, {}).get(scene_index)
    raise ValueError(f"Unsupported position mode: {position_mode}")


def token_at(tokens: list[CaptionToken], time_sec: float) -> CaptionToken | None:
    starts = [token.time_sec for token in tokens]
    index = bisect.bisect_right(starts, time_sec) - 1
    if index < 0:
        return None
    token = tokens[index]
    if token.time_sec <= time_sec <= token.end_sec:
        return token
    return None


def should_show(row: MatchRow, mode: str, min_rms: float, uncertain_confidence: float) -> bool:
    if mode == "all":
        return True
    if mode == "active":
        return row.is_sounding
    if mode == "lenient":
        return row.is_sounding or row.audio_rms >= min_rms or row.confidence <= uncertain_confidence
    raise ValueError(f"Unsupported show mode: {mode}")


def resolve_font(
    font_path: Path | None,
    font_size: int,
    style: CaptionStyle | None = None,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if font_path is not None:
        candidates.append(font_path)
    if style is not None:
        candidates.extend(Path(candidate) for candidate in style.font_candidates)
    candidates.extend(
        [
            Path(r"C:\Windows\Fonts\malgunbd.ttf"),
            Path(r"C:\Windows\Fonts\malgun.ttf"),
            Path(r"C:\Windows\Fonts\arial.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), font_size)
    return ImageFont.load_default()


def clamp_int(value: float, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        maximum = minimum
    return max(minimum, min(maximum, int(round(value))))


def font_size_for_bbox(
    bbox_height: float,
    mode: str,
    fixed_size: int,
    scale: float,
    min_size: int,
    max_size: int,
) -> int:
    if mode == "fixed":
        return fixed_size
    if mode == "bbox":
        return clamp_int(bbox_height * scale, min_size, max_size)
    raise ValueError(f"Unsupported font size mode: {mode}")


def anchor_fraction_for_bbox(
    bbox_height: float,
    mode: str,
    anchor_fraction: float,
    small_threshold: float,
    small_anchor_fraction: float,
) -> float:
    if mode == "fraction":
        return anchor_fraction
    if mode == "adaptive":
        return small_anchor_fraction if bbox_height <= small_threshold else anchor_fraction
    raise ValueError(f"Unsupported caption anchor mode: {mode}")


def select_caption_style(text: str) -> CaptionStyle:
    compact = re.sub(r"\s+", "", text)
    for pattern, style in CAPTION_STYLE_RULES:
        if pattern.search(compact):
            return style
    return DEFAULT_CAPTION_STYLE


def rgba(color: tuple[int, int, int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if len(color) == 4:
        return color
    return (color[0], color[1], color[2], 255)


def draw_accent_marks(
    draw: ImageDraw.ImageDraw,
    layer_size: tuple[int, int],
    style: CaptionStyle,
    font_size: int,
) -> None:
    if style.accent == "none":
        return
    width, height = layer_size
    color = style.accent_fill
    burst_scale = max(0.4, style.burst_scale)
    line_width = max(2, int(round(font_size * 0.08 * burst_scale)))
    if style.accent == "burst":
        cx = width / 2.0
        cy = height / 2.0
        outer_x = width * 0.48
        outer_y = height * 0.48
        inner_x = outer_x * 0.42
        inner_y = outer_y * 0.42
        points = []
        for index in range(28):
            angle = (math.tau * index / 28.0) - 0.16
            radius_x = outer_x if index % 2 == 0 else inner_x
            radius_y = outer_y if index % 2 == 0 else inner_y
            points.append((cx + math.cos(angle) * radius_x, cy + math.sin(angle) * radius_y))
        shadow_points = [(x + font_size * 0.08, y + font_size * 0.09) for x, y in points]
        accent_rgba = rgba(color)
        draw.polygon(shadow_points, fill=(0, 0, 0, 110))
        draw.polygon(points, fill=(accent_rgba[0], accent_rgba[1], accent_rgba[2], 78))
        draw.line(points + [points[0]], fill=(0, 0, 0, 180), width=max(1, line_width // 2))
        marks = [
            ((font_size * 0.4, font_size * 0.4), (font_size * 1.2 * burst_scale, font_size * 0.05)),
            ((font_size * 0.65, height - font_size * 0.35), (font_size * 0.08, height - font_size * 0.08)),
            ((width - font_size * 0.85, font_size * 0.35), (width - font_size * 0.1, font_size * 0.05)),
            ((width - font_size * 0.45, height - font_size * 0.55), (width - font_size * 0.04, height - font_size * 0.12)),
        ]
    elif style.accent == "spark":
        cx = width - font_size * 0.45
        cy = font_size * 0.45
        arm = font_size * 0.22
        marks = [
            ((cx - arm, cy), (cx + arm, cy)),
            ((cx, cy - arm), (cx, cy + arm)),
            ((font_size * 0.35, height - font_size * 0.35), (font_size * 0.75, height - font_size * 0.75)),
        ]
    else:
        marks = [
            ((font_size * 0.25, font_size * 0.35), (font_size * 0.75, font_size * 0.1)),
            ((font_size * 0.35, font_size * 0.65), (font_size * 0.9, font_size * 0.35)),
            ((width - font_size * 0.85, height - font_size * 0.3), (width - font_size * 0.25, height - font_size * 0.6)),
        ]
    for start, end in marks:
        draw.line(start + end, fill=color, width=line_width)


def transform_caption_layer(layer: Image.Image, style: CaptionStyle) -> Image.Image:
    scale_x = max(0.2, style.scale_x)
    scale_y = max(0.2, style.scale_y)
    if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
        layer = layer.resize(
            (
                max(1, int(round(layer.size[0] * scale_x))),
                max(1, int(round(layer.size[1] * scale_y))),
            ),
            Image.Resampling.BICUBIC,
        )

    shear_x = style.shear_x
    if abs(shear_x) > 0.01:
        width, height = layer.size
        x_offset = max(0.0, -shear_x * height)
        new_width = max(1, int(round(width + abs(shear_x) * height)))
        layer = layer.transform(
            (new_width, height),
            Image.Transform.AFFINE,
            (1, -shear_x, -x_offset, 0, 1, 0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )
    return layer


def draw_typographic_caption(
    image: Image.Image,
    text: str,
    anchor_xy: tuple[float, float],
    font: ImageFont.ImageFont,
    font_size: int,
    style: CaptionStyle,
) -> None:
    scratch = ImageDraw.Draw(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    stroke_width = max(1, int(round(font_size / 18 * style.stroke_scale)))
    shadow_x = int(round(font_size * style.shadow_offset[0]))
    shadow_y = int(round(font_size * style.shadow_offset[1]))
    text_bbox = scratch.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    pad = max(8, int(round(font_size * (0.48 + 0.22 * max(0.0, style.burst_scale - 1.0)))))

    extrude_x = int(round(font_size * style.extrusion_offset[0]))
    extrude_y = int(round(font_size * style.extrusion_offset[1]))
    layer_width = max(1, text_width + pad * 2 + abs(shadow_x) + abs(extrude_x) + stroke_width * 2)
    layer_height = max(1, text_height + pad * 2 + abs(shadow_y) + abs(extrude_y) + stroke_width * 2)
    layer = Image.new("RGBA", (layer_width, layer_height), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    text_x = pad - text_bbox[0]
    text_y = pad - text_bbox[1]

    draw_accent_marks(layer_draw, layer.size, style, font_size)
    if style.extrusion_fill is not None and (abs(extrude_x) > 0 or abs(extrude_y) > 0):
        steps = max(2, int(round(max(abs(extrude_x), abs(extrude_y)) / max(1, font_size * 0.08))))
        for step in range(steps, 0, -1):
            ratio = step / steps
            layer_draw.text(
                (text_x + extrude_x * ratio, text_y + extrude_y * ratio),
                text,
                font=font,
                fill=style.extrusion_fill,
                stroke_width=stroke_width,
                stroke_fill=style.extrusion_fill,
            )
    layer_draw.text(
        (text_x + shadow_x, text_y + shadow_y),
        text,
        font=font,
        fill=style.shadow_fill,
        stroke_width=stroke_width,
        stroke_fill=style.shadow_fill,
    )
    layer_draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=rgba(style.fill),
        stroke_width=stroke_width,
        stroke_fill=rgba(style.stroke_fill),
    )

    layer = transform_caption_layer(layer, style)
    if abs(style.angle) > 0.01:
        layer = layer.rotate(style.angle, resample=Image.Resampling.BICUBIC, expand=True)

    width, height = image.size
    fit_scale = min(1.0, width / max(1, layer.size[0]), height / max(1, layer.size[1]))
    if fit_scale < 1.0:
        layer = layer.resize(
            (
                max(1, int(round(layer.size[0] * fit_scale))),
                max(1, int(round(layer.size[1] * fit_scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    max_left = max(0, width - layer.size[0])
    max_top = max(0, height - layer.size[1])
    left = clamp_int(anchor_xy[0] - layer.size[0] / 2, 0, max_left)
    top = clamp_int(anchor_xy[1] - layer.size[1] / 2, 0, max_top)
    image.paste(layer, (left, top), layer)


def draw_caption(
    frame_bgr: np.ndarray,
    text: str,
    bbox: tuple[float, float, float, float],
    font_path: Path | None,
    font_cache: dict[tuple[str, int], ImageFont.ImageFont],
    font_size_mode: str,
    fixed_font_size: int,
    bbox_font_scale: float,
    min_font_size: int,
    max_font_size: int,
    caption_anchor_mode: str,
    anchor_fraction: float,
    small_bbox_height_threshold: float,
    small_anchor_fraction: float,
    caption_style: str,
) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = bbox
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    anchor_x = x1 + box_width / 2.0
    effective_anchor_fraction = anchor_fraction_for_bbox(
        bbox_height=box_height,
        mode=caption_anchor_mode,
        anchor_fraction=anchor_fraction,
        small_threshold=small_bbox_height_threshold,
        small_anchor_fraction=small_anchor_fraction,
    )
    anchor_y = y1 + box_height * effective_anchor_fraction

    dynamic_font_size = font_size_for_bbox(
        bbox_height=box_height,
        mode=font_size_mode,
        fixed_size=fixed_font_size,
        scale=bbox_font_scale,
        min_size=min_font_size,
        max_size=max_font_size,
    )
    style = select_caption_style(text) if caption_style == "rule" else DEFAULT_CAPTION_STYLE
    dynamic_font_size = clamp_int(
        dynamic_font_size * (style.font_scale if caption_style == "rule" else 1.0),
        min_font_size,
        max(max_font_size, int(round(max_font_size * style.font_scale))),
    )
    font_cache_key = (style.name if caption_style == "rule" else caption_style, dynamic_font_size)
    font = font_cache.get(font_cache_key)
    if font is None:
        font = resolve_font(font_path, dynamic_font_size, style=style)
        font_cache[font_cache_key] = font

    if caption_style in {"rule", "plain"}:
        if caption_style == "plain":
            style = CaptionStyle(
                name="plain",
                fill=(255, 255, 255),
                stroke_fill=(0, 0, 0),
                shadow_fill=(0, 0, 0, 170),
                accent_fill=(255, 255, 255, 0),
                stroke_scale=1.0,
                shadow_offset=(0.08, 0.10),
            )
        draw_typographic_caption(
            image=image,
            text=text,
            anchor_xy=(anchor_x, anchor_y),
            font=font,
            font_size=dynamic_font_size,
            style=style,
        )
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    pad_x = max(4, int(round(dynamic_font_size * 0.32)))
    pad_y = max(3, int(round(dynamic_font_size * 0.18)))
    stroke_width = max(1, int(round(dynamic_font_size / 22)))
    radius = max(3, int(round(dynamic_font_size * 0.18)))
    text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    left = int(round(anchor_x - text_width / 2 - pad_x))
    top = int(round(anchor_y - text_height / 2 - pad_y))
    right = left + text_width + pad_x * 2
    bottom = top + text_height + pad_y * 2

    width, height = image.size
    if left < 0:
        right -= left
        left = 0
    if right > width:
        left -= right - width
        right = width
    if top < 0:
        bottom -= top
        top = 0
    if bottom > height:
        top -= bottom - height
        bottom = height
    left = max(0, left)
    top = max(0, top)

    draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=(0, 0, 0))
    draw.text(
        (left + pad_x, top + pad_y),
        text,
        font=font,
        fill=(255, 255, 255),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0),
    )

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def mux_audio(video_no_audio: Path, source_video: Path, output_path: Path) -> bool:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_no_audio),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return True


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
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


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def resolve_render_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path | None]:
    assets_root = resolve_project_path(args.assets_root)
    input_root = resolve_project_path(args.input_root)

    clip_dir = resolve_project_path(args.clip_dir) if args.clip_dir is not None else None
    clip_id = args.clip_id
    if clip_dir is not None and clip_id is None:
        clip_id = clip_dir.name
    if clip_dir is None and clip_id is not None:
        clip_dir = assets_root / clip_id

    if clip_id is None and args.input_video is not None:
        clip_id = args.input_video.stem
    if clip_dir is None and clip_id is not None:
        clip_dir = assets_root / clip_id

    input_video = (
        resolve_project_path(args.input_video)
        if args.input_video is not None
        else None
    )
    if input_video is None:
        if clip_id is None:
            raise ValueError("Pass --clip-id/--clip-dir or explicit --input-video.")
        input_video = input_root / f"{clip_id}.mp4"

    matching_csv = (
        resolve_project_path(args.matching_csv)
        if args.matching_csv is not None
        else None
    )
    if matching_csv is None:
        if clip_dir is None:
            raise ValueError("Pass --clip-id/--clip-dir or explicit --matching-csv.")
        matching_csv = clip_dir / "matching" / "time_object_audio_coords_full.csv"

    mucap_output = (
        resolve_project_path(args.mucap_output)
        if args.mucap_output is not None
        else None
    )
    if mucap_output is None:
        if clip_dir is None:
            raise ValueError("Pass --clip-id/--clip-dir or explicit --mucap-output.")
        mucap_output = clip_dir / "captions" / "mucap_output"

    output_path = resolve_project_path(args.output) if args.output is not None else None
    if output_path is None:
        if clip_dir is None:
            raise ValueError("Pass --clip-id/--clip-dir or explicit --output.")
        output_path = clip_dir / "final" / "captioned_video.mp4"

    return (
        input_video.resolve(),
        matching_csv.resolve(),
        mucap_output.resolve(),
        output_path.resolve(),
        clip_dir.resolve() if clip_dir is not None else None,
    )


def render_video(args: argparse.Namespace) -> None:
    input_video, matching_csv, mucap_output, output_path, clip_dir = resolve_render_paths(args)

    tracks = load_matching_csv(matching_csv)
    captions = load_mucap_captions(
        mucap_output,
        default_duration=args.token_duration,
        max_duration=args.max_token_duration,
    )
    scene_cuts: list[float] = []
    scene_cuts_path: Path | None = None
    needs_scene_cuts = args.scene_cut_aware or args.position_mode == "fixed-by-scene" or args.scene_cuts is not None
    if needs_scene_cuts:
        scene_cuts_path = (
            resolve_project_path(args.scene_cuts)
            if args.scene_cuts is not None
            else (clip_dir / "scene_cuts.json" if clip_dir is not None else None)
        )
        if scene_cuts_path is None or not scene_cuts_path.exists():
            raise FileNotFoundError(
                "This render mode needs a scene_cuts.json file. "
                "Run scripts/detect_scene_cuts.py first or pass --scene-cuts."
            )
        scene_cuts = load_scene_cut_times(scene_cuts_path)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fixed_scene_tracks: dict[str, dict[int, MatchRow]] = {}
    if args.position_mode == "fixed-by-scene":
        fixed_scene_tracks = build_fixed_scene_tracks(
            tracks=tracks,
            scene_cuts=scene_cuts,
            frame_width=width,
            frame_height=height,
            min_width=args.fixed_min_bbox_width,
            min_height=args.fixed_min_bbox_height,
            max_area_fraction=args.fixed_max_area_fraction,
            max_height_fraction=args.fixed_max_height_fraction,
            min_samples=args.fixed_min_samples,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_cache: dict[tuple[str, int], ImageFont.ImageFont] = {}

    with tempfile.TemporaryDirectory() as temp_name:
        temp_video = Path(temp_name) / "captioned_no_audio.mp4"
        writer = cv2.VideoWriter(
            str(temp_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video writer: {temp_video}")

        frame_index = 0
        drawn_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            time_sec = frame_index / fps
            if near_scene_cut(scene_cuts, time_sec, args.cut_margin):
                writer.write(frame)
                frame_index += 1
                continue
            for object_id, tokens in captions.items():
                rows = tracks.get(object_id)
                if not rows:
                    continue
                token = token_at(tokens, time_sec)
                if token is None:
                    continue
                position_row = position_row_for_time(
                    object_id=object_id,
                    rows=rows,
                    time_sec=time_sec,
                    position_mode=args.position_mode,
                    interpolate=args.interpolate,
                    scene_cuts=scene_cuts,
                    fixed_scene_tracks=fixed_scene_tracks,
                )
                if position_row is None:
                    continue
                decision_row = nearest_row(rows, time_sec)
                if not should_show(decision_row, args.show_mode, args.min_rms, args.uncertain_confidence):
                    continue
                frame = draw_caption(
                    frame,
                    text=token.text,
                    bbox=position_row.bbox,
                    font_path=args.font,
                    font_cache=font_cache,
                    font_size_mode=args.font_size_mode,
                    fixed_font_size=args.font_size,
                    bbox_font_scale=args.bbox_font_scale,
                    min_font_size=args.min_font_size,
                    max_font_size=args.max_font_size,
                    caption_anchor_mode=args.caption_anchor_mode,
                    anchor_fraction=args.anchor_fraction,
                    small_bbox_height_threshold=args.small_bbox_height_threshold,
                    small_anchor_fraction=args.small_anchor_fraction,
                    caption_style=args.caption_style,
                )
                drawn_count += 1

            writer.write(frame)
            frame_index += 1

        cap.release()
        writer.release()

        muxed = False
        if args.keep_audio:
            muxed = mux_audio(temp_video, input_video, output_path)
        if not muxed:
            shutil.copy2(temp_video, output_path)
            if args.keep_audio:
                print("Warning: ffmpeg was not found; saved captioned video without audio.")

    print(f"Input video: {input_video}")
    print(f"Matching CSV: {matching_csv}")
    print(f"MuCap output: {mucap_output}")
    print(f"Position mode: {args.position_mode}")
    if scene_cuts_path is not None:
        print(f"Scene cuts: {scene_cuts_path} ({len(scene_cuts)} cuts)")
    if args.position_mode == "fixed-by-scene":
        fixed_count = sum(len(scene_rows) for scene_rows in fixed_scene_tracks.values())
        print(f"Fixed scene bboxes: {fixed_count}")
    print(f"Frames processed: {frame_index}/{frame_count}")
    print(f"Caption draws: {drawn_count}")
    print(f"Saved video: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render MuCap object captions onto the original video using full matching CSV coordinates."
    )
    parser.add_argument(
        "--clip-id",
        default=None,
        help=(
            "Clip id used to infer input/<clip_id>.mp4, "
            "outputs/video_assets/<clip_id>/matching/time_object_audio_coords_full.csv, "
            "outputs/video_assets/<clip_id>/captions/mucap_output, and final/captioned_video.mp4."
        ),
    )
    parser.add_argument(
        "--clip-dir",
        type=Path,
        default=None,
        help="Clip output directory. Defaults to outputs/video_assets/<clip-id>.",
    )
    parser.add_argument("--assets-root", type=Path, default=Path("outputs/video_assets"))
    parser.add_argument("--input-root", type=Path, default=Path("input"))
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument(
        "--matching-csv",
        type=Path,
        default=None,
        help="Use matching/time_object_audio_coords_full.csv for maximum flexibility.",
    )
    parser.add_argument("--mucap-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--show-mode",
        choices=["active", "lenient", "all"],
        default="all",
        help="active: is_sounding=true only; lenient: active or high RMS/low confidence; all: any matching bbox.",
    )
    parser.add_argument("--min-rms", type=float, default=0.003)
    parser.add_argument("--uncertain-confidence", type=float, default=0.65)
    parser.add_argument("--token-duration", type=float, default=0.45)
    parser.add_argument("--max-token-duration", type=float, default=0.9)
    parser.add_argument("--anchor-fraction", type=float, default=-0.10)
    parser.add_argument(
        "--position-mode",
        choices=["dynamic", "fixed-by-scene"],
        default="dynamic",
        help="dynamic: per-frame/interpolated bbox; fixed-by-scene: median bbox per object per scene.",
    )
    parser.add_argument(
        "--scene-cut-aware",
        action="store_true",
        help="Disable bbox interpolation across detected scene cuts.",
    )
    parser.add_argument(
        "--scene-cuts",
        type=Path,
        default=None,
        help="Path to scene_cuts.json. Default: <clip-dir>/scene_cuts.json when --scene-cut-aware is set.",
    )
    parser.add_argument(
        "--cut-margin",
        type=float,
        default=0.0,
        help="Optional seconds around each scene cut where captions are hidden. Default keeps captions visible.",
    )
    parser.add_argument("--fixed-min-bbox-width", type=float, default=20.0)
    parser.add_argument("--fixed-min-bbox-height", type=float, default=20.0)
    parser.add_argument("--fixed-max-area-fraction", type=float, default=0.45)
    parser.add_argument("--fixed-max-height-fraction", type=float, default=0.90)
    parser.add_argument("--fixed-min-samples", type=int, default=2)
    parser.add_argument("--font", type=Path, default=None)
    parser.add_argument("--font-size", type=int, default=44)
    parser.add_argument(
        "--font-size-mode",
        choices=["fixed", "bbox"],
        default="bbox",
        help="fixed: use --font-size; bbox: scale font size from each object's bbox height.",
    )
    parser.add_argument(
        "--bbox-font-scale",
        type=float,
        default=0.35,
        help="Font size multiplier for bbox-height-based captions.",
    )
    parser.add_argument("--min-font-size", type=int, default=14)
    parser.add_argument("--max-font-size", type=int, default=28)
    parser.add_argument(
        "--caption-style",
        choices=["rule", "plain", "box"],
        default="rule",
        help=(
            "rule: choose typography style from caption text; plain: no background, white text; "
            "box: original black background box."
        ),
    )
    parser.add_argument(
        "--caption-anchor-mode",
        choices=["fraction", "adaptive"],
        default="adaptive",
        help="fraction: always use --anchor-fraction; adaptive: use --small-anchor-fraction for small bboxes.",
    )
    parser.add_argument(
        "--small-bbox-height-threshold",
        type=float,
        default=90.0,
        help="Bbox heights at or below this value use --small-anchor-fraction in adaptive mode.",
    )
    parser.add_argument(
        "--small-anchor-fraction",
        type=float,
        default=-0.10,
        help="Anchor fraction used for small bboxes in adaptive mode. -0.10 places the caption just above the bbox.",
    )
    parser.add_argument("--no-interpolate", dest="interpolate", action="store_false")
    parser.set_defaults(interpolate=True)
    parser.add_argument("--no-audio", dest="keep_audio", action="store_false")
    parser.set_defaults(keep_audio=True)
    return parser.parse_args()


def main() -> None:
    render_video(parse_args())


if __name__ == "__main__":
    main()

