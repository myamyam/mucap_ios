import argparse
import base64
import json
import os
import re
import tempfile
import time
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MODEL = "gpt-image-2"
GPT_IMAGE_SIZES = [(1024, 1024), (1536, 1024), (1024, 1536)]

DEFAULT_COLORS = {
    "red": "#ff2d2d",
    "yellow": "#ffdc1e",
    "green": "#23dc50",
    "blue": "#3791ff",
    "magenta": "#dc50ff",
    "orange": "#ff8c2d",
    "cyan": "#28dcdc",
    "purple": "#7b2cff",
    "teal": "#00b894",
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


def default_object_plan_path(frames_dir: Path) -> Path:
    clip_dir = frames_dir.parent
    return clip_dir / "detections" / "object_plan.json"


def default_output_dir(frames_dir: Path) -> Path:
    return frames_dir.parent / "detections" / "colored_on_gray"


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
        if not object_id or not instrument or not color:
            raise ValueError(
                f"objects[{index}] must include object_id, instrument, and color"
            )
        color_name = str(color).lower().strip()
        objects.append(
            {
                "object_id": str(object_id),
                "instrument": str(instrument),
                "description": str(entry.get("description") or instrument),
                "color": color_name,
                "hex": entry.get("hex") or DEFAULT_COLORS.get(color_name, color),
                "prompt_hint": str(entry.get("prompt_hint") or entry.get("hint") or ""),
            }
        )
    return objects


def build_prompt(objects: list[dict[str, Any]], extra_prompt: str | None) -> str:
    lines = [
        "Edit this concert frame for machine-vision masking.",
        "The first input image is the current frame that must be edited.",
        "If a second input image is provided, it is the previous frame's colored output.",
        "Use the second input image only as a temporal consistency reference so the same physical object keeps the same color across frames.",
        "Never copy the previous frame geometry into the current frame; follow the current frame's visible object positions.",
        "Color only the listed visible musical instruments or instrument-playing objects.",
        "Convert every non-target pixel to grayscale, including background, stage, chairs, stands, faces, audience, screen, logo, and unrelated objects.",
        "Preserve the original resolution, composition, lighting, and geometry as much as possible.",
        "Use solid, opaque colors. Do not use gradients, shadows, glow, transparency, labels, outlines, text, or annotations.",
        "If an object is not visible in this frame, leave it grayscale and uncolored.",
        "Object color map:",
    ]
    for obj in objects:
        hint = f" Hint: {obj['prompt_hint']}" if obj["prompt_hint"] else ""
        lines.append(
            f"- {obj['object_id']}: {obj['description']} / {obj['instrument']} -> "
            f"{obj['color']} ({obj['hex']}).{hint}"
        )
    if extra_prompt:
        lines.extend(["Additional instruction:", extra_prompt])
    lines.extend(
        [
            "The input may be placed on a padded API canvas. Only edit the real video frame content.",
            "Keep the real video frame content in the same position and aspect ratio.",
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
        help="Defaults to <clip>/detections/colored_on_gray.",
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
        "--no-previous-output-reference",
        action="store_true",
        help="Disable passing the previous colored output image as a consistency reference.",
    )
    parser.add_argument("--extra-prompt", default=None)
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
    print("Prompt:")
    print(prompt)

    if args.dry_run:
        return

    client = import_openai_client()
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
            continue

        print(f"[{count}/{len(selected)}] Editing {frame_path.name}")
        if previous_output_path is not None and previous_output_path.exists():
            print(f"  Reference: {previous_output_path.name}")
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
        )
        print(f"Saved {output_path}")
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
