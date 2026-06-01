import argparse
import base64
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MODEL = "gpt-image-2"
GPT_IMAGE_SIZES = [(1024, 1024), (1536, 1024), (1024, 1536)]

NAMED_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "orange": (255, 128, 0),
    "purple": (128, 0, 255),
}


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = (base or ROOT) / value
    return value


def frame_index(path: Path) -> int:
    patterns = [
        r"\((\d+)\)\.(?:png|jpg|jpeg|webp)$",
        r"_(\d+)(?:_[0-9.]+s)?\.(?:png|jpg|jpeg|webp)$",
        r"frame[_-]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def image_paths(directory: Path) -> list[Path]:
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: (frame_index(path), path.name))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def parse_color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in NAMED_COLORS:
            return NAMED_COLORS[normalized]
        if re.fullmatch(r"#[0-9a-f]{6}", normalized):
            return tuple(int(normalized[i : i + 2], 16) for i in (1, 3, 5))
        parts = [part.strip() for part in normalized.split(",")]
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return tuple(int(part) for part in parts)
    if isinstance(value, list) and len(value) == 3:
        return tuple(int(part) for part in value)
    raise ValueError(f"Unsupported color: {value!r}")


def color_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def color_rgb_text(rgb: tuple[int, int, int]) -> str:
    return f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]})"


def default_object_plan_path(frames_dir: Path) -> Path:
    clip_dir = frames_dir.parent
    return clip_dir / "detections" / "object_plan.json"


def default_output_dir(frames_dir: Path) -> Path:
    return frames_dir.parent / "detections" / "colored_on_black"


def load_object_plan(plan_path: Path) -> list[dict[str, Any]]:
    data = load_json(plan_path)
    entries = data.get("objects")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Object plan must contain a non-empty objects list: {plan_path}")

    objects = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"objects[{index}] must be an object")
        object_id = entry.get("object_id") or entry.get("id") or entry.get("name")
        instrument = entry.get("instrument") or entry.get("class") or entry.get("label")
        color = entry.get("color")
        if not object_id or not instrument or color is None:
            raise ValueError(
                f"objects[{index}] must include object_id, instrument, and color"
            )
        rgb = parse_color(color)
        objects.append(
            {
                "object_id": str(object_id),
                "instrument": str(instrument),
                "description": str(entry.get("description") or instrument),
                "color": list(rgb),
                "hex": entry.get("hex") or color_hex(rgb),
                "rgb_text": color_rgb_text(rgb),
                "prompt_hint": str(entry.get("prompt_hint") or entry.get("hint") or ""),
            }
        )
    return objects


def build_prompt(objects: list[dict[str, Any]], extra_prompt: str | None) -> str:
    lines = [
        "Create a black-background RGB label map for this video frame.",
        "Use one solid RGB color for each listed sprite.",
        "Fill only the visible pixels of each listed sprite.",
        "Use only the listed RGB colors and pure black RGB(0, 0, 0).",
        "Do not add text, outlines, labels, shadows, or effects.",
        "If a listed sprite is not visible, leave it black.",
        "Sprite RGB color map:",
    ]
    for obj in objects:
        hint = f" Hint: {obj['prompt_hint']}" if obj["prompt_hint"] else ""
        lines.append(
            f"- {obj['object_id']}: {obj['description']} / {obj['instrument']} -> "
            f"{obj['rgb_text']} ({obj['hex']}).{hint}"
        )
    if extra_prompt:
        lines.extend(["Additional instruction:", extra_prompt])
    lines.extend(
        [
            "Keep the same frame size and aspect ratio.",
        ]
    )
    return "\n".join(lines)


def import_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai Python package is not installed. Install it in the active venv "
            "before running GPT image edits, e.g. `pip install openai`."
        ) from exc
    return OpenAI()


def parse_size(value: str) -> tuple[int, int] | None:
    if value == "auto":
        return None
    match = re.fullmatch(r"(\d+)x(\d+)", value)
    if not match:
        raise ValueError(f"Unsupported image size value: {value!r}")
    return int(match.group(1)), int(match.group(2))


def choose_canvas_size(source_size: tuple[int, int], requested_size: str) -> tuple[int, int]:
    parsed = parse_size(requested_size)
    if parsed is not None:
        return parsed

    source_width, source_height = source_size
    source_aspect = source_width / source_height
    return min(
        GPT_IMAGE_SIZES,
        key=lambda size: abs((size[0] / size[1]) - source_aspect),
    )


def fit_box(source_size: tuple[int, int], canvas_size: tuple[int, int]) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    canvas_width, canvas_height = canvas_size
    scale = min(canvas_width / source_width, canvas_height / source_height)
    fitted_width = max(1, int(round(source_width * scale)))
    fitted_height = max(1, int(round(source_height * scale)))
    left = (canvas_width - fitted_width) // 2
    top = (canvas_height - fitted_height) // 2
    return left, top, left + fitted_width, top + fitted_height


def prepare_api_image(
    source_path: Path,
    temp_dir: Path,
    canvas_size: tuple[int, int],
) -> tuple[Path, tuple[int, int], tuple[int, int, int, int]]:
    image = Image.open(source_path).convert("RGB")
    source_size = image.size
    box = fit_box(source_size, canvas_size)
    fitted = image.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", canvas_size, (0, 0, 0))
    canvas.paste(fitted, (box[0], box[1]))

    prepared_path = temp_dir / f"{source_path.stem}_api.png"
    canvas.save(prepared_path)
    return prepared_path, source_size, box


def restore_source_aspect(
    image_bytes: bytes,
    output_path: Path,
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
    content_box: tuple[int, int, int, int],
) -> None:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    crop_box = content_box
    if image.size != canvas_size:
        scale_x = image.size[0] / canvas_size[0]
        scale_y = image.size[1] / canvas_size[1]
        crop_box = (
            int(round(content_box[0] * scale_x)),
            int(round(content_box[1] * scale_y)),
            int(round(content_box[2] * scale_x)),
            int(round(content_box[3] * scale_y)),
        )
    restored = image.crop(crop_box).resize(source_size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    restored.save(output_path)


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        labels_count, labels = cv2.connectedComponents(mask.astype("uint8"), connectivity=8)
        return labels, labels_count

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    label = 0
    for start_y, start_x in zip(*np.where(mask & (labels == 0)), strict=False):
        label += 1
        stack = [(int(start_y), int(start_x))]
        labels[start_y, start_x] = label
        while stack:
            y, x = stack.pop()
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if labels[ny, nx] == 0 and mask[ny, nx]:
                        labels[ny, nx] = label
                        stack.append((ny, nx))
    return labels, label + 1


def solidify_component_colors(
    image: np.ndarray,
    palette: list[tuple[int, int, int]],
    black_tolerance: int,
    min_component_area: int,
    dominance_ratio: float,
) -> np.ndarray:
    non_black = image.max(axis=2) > black_tolerance
    labels, labels_count = component_labels(non_black)
    output = image.copy()
    palette_array = np.asarray(palette, dtype=np.uint8)

    for label in range(1, labels_count):
        component = labels == label
        area = int(component.sum())
        if area < min_component_area:
            output[component] = (0, 0, 0)
            continue
        pixels = image[component]
        counts = [
            int(np.all(pixels == color, axis=1).sum())
            for color in palette_array
        ]
        best_index = int(np.argmax(counts))
        if counts[best_index] <= 0:
            continue
        if counts[best_index] / area < dominance_ratio:
            continue
        output[component] = palette_array[best_index]

    return output


def quantize_mask_image(
    image_path: Path,
    palette: list[tuple[int, int, int]],
    black_tolerance: int,
    color_tolerance: int,
    solidify_components: bool,
    solidify_min_component_area: int,
    solidify_dominance_ratio: float,
) -> None:
    if not palette:
        raise ValueError("Cannot quantize mask image without at least one palette color")

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.int16)
    palette_array = np.asarray(palette, dtype=np.int16)
    non_black = rgb.max(axis=2) > black_tolerance

    distances = np.abs(rgb[:, :, None, :] - palette_array[None, None, :, :]).max(axis=3)
    nearest = distances.argmin(axis=2)
    close_enough = distances.min(axis=2) <= color_tolerance
    quantized = palette_array[nearest].astype(np.uint8)
    quantized[~(non_black & close_enough)] = (0, 0, 0)

    if solidify_components:
        quantized = solidify_component_colors(
            quantized,
            palette,
            black_tolerance,
            solidify_min_component_area,
            solidify_dominance_ratio,
        )

    Image.fromarray(quantized, mode="RGB").save(image_path)


def decode_image_response(result: Any) -> bytes:
    if not getattr(result, "data", None):
        raise RuntimeError("OpenAI image edit response did not contain image data")
    item = result.data[0]
    b64_json = getattr(item, "b64_json", None)
    if b64_json:
        return base64.b64decode(b64_json)
    url = getattr(item, "url", None)
    if url:
        raise RuntimeError(
            "The image edit response returned a URL, but this script expects b64_json. "
            "Set response_format='b64_json' or update the script to download URLs."
        )
    raise RuntimeError("OpenAI image edit response had neither b64_json nor url")


def is_moderation_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "moderation_blocked" in text or "safety system" in text


def write_black_mask_like(frame_path: Path, output_path: Path) -> None:
    image = Image.open(frame_path).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image.size, (0, 0, 0)).save(output_path)


def write_fallback_mask(
    frame_path: Path,
    output_path: Path,
    previous_output_path: Path | None,
    reason: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if previous_output_path is not None and previous_output_path.exists():
        shutil.copyfile(previous_output_path, output_path)
        print(f"  Fallback mask: copied {previous_output_path.name} -> {output_path.name} ({reason})")
        return
    write_black_mask_like(frame_path, output_path)
    print(f"  Fallback mask: wrote black mask {output_path.name} ({reason})")


def edit_frame(
    client: Any,
    model: str,
    frame_path: Path,
    prompt: str,
    output_path: Path,
    size: str,
    quality: str | None,
    previous_output_path: Path | None,
    preserve_source_aspect: bool,
    quantize_palette: list[tuple[int, int, int]] | None,
    quantize_black_tolerance: int,
    quantize_color_tolerance: int,
    solidify_components: bool,
    solidify_min_component_area: int,
    solidify_dominance_ratio: float,
) -> None:
    with tempfile.TemporaryDirectory() as temp_name, ExitStack() as stack:
        temp_dir = Path(temp_name)
        source_size: tuple[int, int] | None = None
        canvas_size: tuple[int, int] | None = None
        content_box: tuple[int, int, int, int] | None = None
        api_size = size
        api_frame_path = frame_path
        api_previous_output_path = previous_output_path

        if preserve_source_aspect:
            original_size = Image.open(frame_path).size
            canvas_size = choose_canvas_size(original_size, size)
            api_size = f"{canvas_size[0]}x{canvas_size[1]}"
            api_frame_path, source_size, content_box = prepare_api_image(
                source_path=frame_path,
                temp_dir=temp_dir,
                canvas_size=canvas_size,
            )
            if previous_output_path is not None and previous_output_path.exists():
                api_previous_output_path, _previous_size, _previous_box = prepare_api_image(
                    source_path=previous_output_path,
                    temp_dir=temp_dir,
                    canvas_size=canvas_size,
                )

        image_file = stack.enter_context(api_frame_path.open("rb"))
        image_input: Any = image_file
        if api_previous_output_path is not None and api_previous_output_path.exists():
            previous_image_file = stack.enter_context(api_previous_output_path.open("rb"))
            image_input = [image_file, previous_image_file]

        kwargs: dict[str, Any] = {
            "model": model,
            "image": image_input,
            "prompt": prompt,
            "size": api_size,
        }
        if not str(model).startswith("gpt-image-"):
            kwargs["response_format"] = "b64_json"
        if quality:
            kwargs["quality"] = quality
        result = client.images.edit(**kwargs)

    image_bytes = decode_image_response(result)
    if (
        preserve_source_aspect
        and source_size is not None
        and canvas_size is not None
        and content_box is not None
    ):
        restore_source_aspect(image_bytes, output_path, source_size, canvas_size, content_box)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

    if quantize_palette is not None:
        quantize_mask_image(
            output_path,
            quantize_palette,
            quantize_black_tolerance,
            quantize_color_tolerance,
            solidify_components,
            solidify_min_component_area,
            solidify_dominance_ratio,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use OpenAI image editing to create one all-objects colored mask image per frame."
    )
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument(
        "--object-plan",
        type=Path,
        default=None,
        help="JSON with objects[]. Defaults to <clip>/detections/object_plan.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <clip>/detections/colored_on_black.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default="auto")
    parser.add_argument("--quality", default=None)
    parser.add_argument(
        "--allow-api-aspect-change",
        action="store_true",
        help="Save raw API output size instead of restoring every output to the source frame size.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--fallback-on-moderation",
        choices=["copy-previous", "black", "error"],
        default="copy-previous",
        help=(
            "When the image edit API blocks a frame, either copy the previous successful "
            "mask, write a black mask, or raise the error."
        ),
    )
    parser.add_argument(
        "--no-previous-output-reference",
        action="store_true",
        help="Disable passing the previous colored output image as a consistency reference.",
    )
    parser.add_argument("--extra-prompt", default=None)
    parser.add_argument(
        "--no-quantize-palette",
        action="store_true",
        help="Disable post-processing that snaps every non-black pixel to the nearest object RGB color.",
    )
    parser.add_argument(
        "--no-solidify-components",
        action="store_true",
        help="Disable post-processing that forces each connected non-black component to one dominant palette color.",
    )
    parser.add_argument(
        "--solidify-min-component-area",
        type=int,
        default=20,
        help="Minimum connected-component area for dominant-color solidification.",
    )
    parser.add_argument(
        "--solidify-dominance-ratio",
        type=float,
        default=0.72,
        help=(
            "Only force a connected component to one color when the dominant palette color "
            "covers at least this fraction of the component. Lower values clean mixed-color "
            "single objects more aggressively; higher values preserve adjacent touching objects."
        ),
    )
    parser.add_argument(
        "--quantize-black-tolerance",
        type=int,
        default=30,
        help="Pixels with all RGB channels at or below this value are forced to pure black during palette quantization.",
    )
    parser.add_argument(
        "--quantize-color-tolerance",
        type=int,
        default=170,
        help=(
            "Only non-black pixels within this max-channel distance from a palette color are snapped "
            "to that color; farther pixels are forced to black."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = resolve_path(args.frames_dir)
    plan_path = resolve_path(args.object_plan) if args.object_plan else default_object_plan_path(frames_dir)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else default_output_dir(frames_dir)
    )

    objects = load_object_plan(plan_path)
    quantize_palette = None if args.no_quantize_palette else [tuple(obj["color"]) for obj in objects]
    prompt = build_prompt(objects, args.extra_prompt)
    frames = image_paths(frames_dir)
    if not frames:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")

    selected = [path for path in frames if frame_index(path) >= args.start_index]
    if args.limit > 0:
        selected = selected[: args.limit]

    print(f"Frames: {len(selected)}")
    print(f"Objects: {len(objects)}")
    print(f"Model: {args.model}")
    print(f"Object plan: {plan_path}")
    print(f"Output: {output_dir}")
    if quantize_palette is not None:
        print(
            "Palette quantization: enabled "
            f"(black_tolerance={args.quantize_black_tolerance}, "
            f"color_tolerance={args.quantize_color_tolerance})"
        )
    else:
        print("Palette quantization: disabled")
    print("Prompt:")
    print(prompt)

    if args.dry_run:
        return

    needs_api = any(
        not (args.skip_existing and (output_dir / frame_path.with_suffix(".png").name).exists())
        for frame_path in selected
    )
    client = import_openai_client() if needs_api else None
    for count, frame_path in enumerate(selected, start=1):
        output_path = output_dir / frame_path.with_suffix(".png").name
        previous_output_path = None
        if not args.no_previous_output_reference:
            current_position = frames.index(frame_path)
            if current_position > 0:
                previous_frame_path = frames[current_position - 1]
                previous_output_path = output_dir / previous_frame_path.with_suffix(".png").name

        if args.skip_existing and output_path.exists():
            print(f"[{count}/{len(selected)}] Skipping existing {output_path}")
            if quantize_palette is not None:
                quantize_mask_image(
                    output_path,
                    quantize_palette,
                    args.quantize_black_tolerance,
                    args.quantize_color_tolerance,
                    not args.no_solidify_components,
                    args.solidify_min_component_area,
                    args.solidify_dominance_ratio,
                )
                print(f"  Quantized existing {output_path.name}")
            continue

        print(f"[{count}/{len(selected)}] Editing {frame_path.name}")
        if previous_output_path is not None and previous_output_path.exists():
            print(f"  Reference: {previous_output_path.name}")
        if client is None:
            raise RuntimeError("OpenAI client was not initialized for a frame that needs API editing.")
        try:
            edit_frame(
                client=client,
                model=args.model,
                frame_path=frame_path,
                prompt=prompt,
                output_path=output_path,
                size=args.size,
                quality=args.quality,
                previous_output_path=previous_output_path,
                preserve_source_aspect=not args.allow_api_aspect_change,
                quantize_palette=quantize_palette,
                quantize_black_tolerance=args.quantize_black_tolerance,
                quantize_color_tolerance=args.quantize_color_tolerance,
                solidify_components=not args.no_solidify_components,
                solidify_min_component_area=args.solidify_min_component_area,
                solidify_dominance_ratio=args.solidify_dominance_ratio,
            )
            print(f"Saved {output_path}")
        except Exception as exc:
            if not is_moderation_error(exc) or args.fallback_on_moderation == "error":
                raise
            if args.fallback_on_moderation == "black":
                write_black_mask_like(frame_path, output_path)
                print(f"  Fallback mask: wrote black mask {output_path.name} (moderation_blocked)")
            else:
                write_fallback_mask(
                    frame_path=frame_path,
                    output_path=output_path,
                    previous_output_path=previous_output_path,
                    reason="moderation_blocked",
                )
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
