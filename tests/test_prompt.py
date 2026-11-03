from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent))

from multimodal_caption.mobile_pipeline import transcribe


def main():
    result = transcribe((TESTS_DIR / "ttock.wav").read_bytes())
    for cue in result["captions"]:
        print(f'[{cue["time"]:.2f}] {cue["text"]}')


if __name__ == "__main__":
    main()
