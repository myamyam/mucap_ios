import argparse
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name,
    )


def default_output_dir(frames_dir: Path) -> Path:
    return frames_dir.parent / "detections" / "grayscale_reference"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create grayscale reference frames with the same names and size as source frames."
    )
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frames_dir = resolve_path(args.frames_dir)
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(frames_dir)
    paths = image_paths(frames_dir)
    if not paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, path in enumerate(paths, start=1):
        output_path = output_dir / path.with_suffix(".png").name
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(paths)}] Skipping existing {output_path}")
            continue
        image = Image.open(path).convert("L").convert("RGB")
        image.save(output_path)
        written += 1
        print(f"[{index}/{len(paths)}] Saved {output_path}")

    print(f"Done. Saved {written} grayscale frames to {output_dir}")


if __name__ == "__main__":
    main()
