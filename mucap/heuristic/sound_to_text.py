import argparse
import sys
from os.path import dirname

if __package__ in (None, ""):
    sys.path.insert(0, dirname(dirname(__file__)))

from heuristic.heuristic_annotation import heuristic_annotation


def main():
    parser = argparse.ArgumentParser(description="Run the Skeleton-based heuristic generator.")
    parser.add_argument("audio")
    args = parser.parse_args()

    details = heuristic_annotation(args.audio, return_details=True)
    print("Generated syllables:", details["text"])
    print(
        f"source={details['feature_source']} instrument={details['instrument']} "
        f"chars={details['char_count']} agreement={details['agreement']} "
        f"estimated={details['is_estimated']}"
    )


if __name__ == "__main__":
    main()
