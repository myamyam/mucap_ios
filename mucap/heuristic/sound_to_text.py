import librosa

from mucap.generate_caption import visualize
from mucap.video_render import render_video
from sound_to_text_logic import (
    extract_audio_features,
    classify_features,
    sound_to_text,
    detect_instrument_type
)

file_path = "classic_test.wav"
# 1. Load audio file
y, sr = librosa.load(file_path, sr=None)

# 2. Extract features
features = extract_audio_features(y, sr)

# 3. Classify features
labels = classify_features(features)

# 4. Auto-detect instrument
instrument = detect_instrument_type("classic_test.wav", features)

# 5. Generate syllables
tokens = sound_to_text(
    **labels,
    duration=features["duration"],
    instrument=instrument
)

print("🎵 Generated syllables:", tokens)

visualize(file_path, tokens, font_dir='fonts')
render_video(file_path, tokens, hop=0.25, font_dir='fonts', fps=30)
