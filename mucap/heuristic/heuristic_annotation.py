from __future__ import annotations

import sys
from os.path import basename, dirname, join

if __package__ in (None, ""):
    sys.path.insert(0, dirname(dirname(__file__)))

from heuristic.data_io import (
    dominant_instrument,
    feature_row_to_skeleton,
    load_audio_features,
)
from heuristic.sound_to_skeleton_core import (
    ConsensusModel,
    RetrievalExamples,
    sound_to_text,
)


_PROJECT_ROOT = dirname(dirname(__file__))
_FEATURE_CSV_PATH = join(_PROJECT_ROOT, "music_features.csv")
_FEATURE_ROWS = None


def _audio_key(path: str) -> str:
    name = basename(path)
    return name[len("fixed_"):] if name.startswith("fixed_") else name


def _load_feature_rows() -> dict[str, dict]:
    global _FEATURE_ROWS
    if _FEATURE_ROWS is None:
        rows = load_audio_features(_FEATURE_CSV_PATH)
        _FEATURE_ROWS = {_audio_key(name): row for name, row in rows.items()}
    return _FEATURE_ROWS


def _extract_runtime_feature_row(file_path: str) -> dict:
    from music_features.feature_extractor import AudioFeatureExtraction

    extractor = AudioFeatureExtraction()
    row = extractor.extract_features(file_path)
    row["file_name"] = basename(file_path)
    row["top5_instruments"] = extractor.predict_instrument_essentia(file_path)
    return row


def _feature_row_for_audio(file_path: str) -> tuple[dict, str]:
    row = _load_feature_rows().get(_audio_key(file_path))
    if row is not None:
        return row, "csv"
    return _extract_runtime_feature_row(file_path), "runtime"


def heuristic_annotation(file_path: str, return_details: bool = False):
    """Generate a rule-based caption with the Skeleton engine."""
    feature_row, feature_source = _feature_row_for_audio(file_path)
    skeleton = feature_row_to_skeleton(feature_row)
    if skeleton.onset_count == 0:
        raise RuntimeError(
            f"No note onsets were available for Skeleton generation: {file_path}"
        )

    result = sound_to_text(
        skeleton=skeleton,
        instrument=dominant_instrument(feature_row),
        consensus=ConsensusModel(),
        examples=RetrievalExamples(),
    )
    result["feature_source"] = feature_source
    result["instrument"] = dominant_instrument(feature_row)

    return result if return_details else result["text"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a Skeleton-based heuristic caption.")
    parser.add_argument("audio")
    args = parser.parse_args()

    details = heuristic_annotation(args.audio, return_details=True)
    print(details["text"])
    print(
        f"source={details['feature_source']} instrument={details['instrument']} "
        f"chars={details['char_count']} agreement={details['agreement']} "
        f"estimated={details['is_estimated']}"
    )
