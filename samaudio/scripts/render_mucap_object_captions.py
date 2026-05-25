import argparse
import bisect
import csv
import glob
import json
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


def resolve_font(font_path: Path | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if font_path is not None:
        candidates.append(font_path)
    candidates.extend(
        [
            Path(r"C:\Windows\Fonts\malgun.ttf"),
            Path(r"C:\Windows\Fonts\malgunbd.ttf"),
            Path(r"C:\Windows\Fonts\arial.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), font_size)
    return ImageFont.load_default()


def draw_caption(
    frame_bgr: np.ndarray,
    text: str,
    bbox: tuple[float, float, float, float],
    font: ImageFont.ImageFont,
    anchor_fraction: float,
) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = bbox
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    anchor_x = x1 + box_width / 2.0
    anchor_y = y1 + box_height * anchor_fraction

    pad_x = 14
    pad_y = 8
    text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
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

    draw.rounded_rectangle((left, top, right, bottom), radius=8, fill=(0, 0, 0))
    draw.text(
        (left + pad_x, top + pad_y),
        text,
        font=font,
        fill=(255, 255, 255),
        stroke_width=2,
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
    font = resolve_font(args.font, args.font_size)

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
                    font=font,
                    anchor_fraction=args.anchor_fraction,
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
    parser.add_argument("--anchor-fraction", type=float, default=0.11)
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
    parser.add_argument("--no-interpolate", dest="interpolate", action="store_false")
    parser.set_defaults(interpolate=True)
    parser.add_argument("--no-audio", dest="keep_audio", action="store_false")
    parser.set_defaults(keep_audio=True)
    return parser.parse_args()


def main() -> None:
    render_video(parse_args())


if __name__ == "__main__":
    main()
