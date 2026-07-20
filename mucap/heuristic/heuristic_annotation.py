import librosa
from mucap.tokends2caption import tokens_to_caption
from sound_to_text_logic import (
    extract_audio_features,
    classify_features,
    sound_to_text,
    detect_instrument_type
)

def heuristic_annotation(file_path):

    # 1. Load audio file
    y, sr = librosa.load(file_path, sr=None)

    # 2. Extract features
    features = extract_audio_features(y, sr)

    # 3. Classify features
    labels = classify_features(features)

    # 4. Auto-detect instrument
    instrument = detect_instrument_type(file_path, features)

    # 5. Generate syllables
    tokens = sound_to_text(
        **labels,
        duration=features["duration"],
        instrument=instrument
    )
    caption = tokens_to_caption(tokens)

    return caption

if __name__ == "__main__":
    # Example usage
    file_path = "classic_test.wav"
    tokens = heuristic_annotation(file_path)
    print("🎵 Generated syllables:", tokens)
