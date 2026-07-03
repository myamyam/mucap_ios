#%%
from os.path import join, basename, exists
from glob import glob

import librosa
import numpy as np
import regex
from tqdm.notebook import tqdm
import pandas as pd
import tensorflow as tf

print(tf.test.is_built_with_cuda())

from feature_extractor_db import AudioFeatureExtraction

CSV_PATH = "../music_features.csv"


already_done = set()
if exists(CSV_PATH):
    try:
        prev_df = pd.read_csv(CSV_PATH, usecols=["file"])
        already_done = set(prev_df["file"].tolist())
        print(f"Found existing CSV with {len(already_done)} processed files.")
    except Exception as e:
        print(f"⚠️  Could not read existing CSV ({e}). Will recreate it.")
        already_done = set()

files = glob(join("..", "dataset", "**", "*.wav"), recursive=True)
files = [f for f in files
         if regex.match(r".*\.novocal_\d+\.wav$", basename(f))
         and f not in already_done]

print(f"{len(files)} new files to process.")


analyzer = AudioFeatureExtraction()

for i, file in enumerate(tqdm(files, desc="Extracting audio features")):

    # Meta information
    *_, dataset, genre, file_name = file.split("/")

    y, sr = analyzer.load_audio(file)

    # Onset envelope for attack/envelope dynamics
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512
    )
    print(f"Processing {file_name} ({i+1}/{len(files)})...")
    attack = 'sharp' if np.max(onset_env) > 0.5 else 'soft'
    envelope = 'transient' if attack == 'sharp' else 'sustained'
    onset_strength = float(np.mean(onset_env))
    # Resonance by duration
    duration = librosa.get_duration(y=y, sr=sr)
    resonance = 'long' if duration > 3.0 else 'short'
    # Note events
    notes = analyzer.extract_notes(y, sr, instrument_hint=genre)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    if y is None:
        print(f"❌  Failed to load {file}, skipping.")
        continue

    feature_row = {
        "file_name": file_name,
        "genre": genre,
        "dataset": dataset,
        "sample_rate": int(sr),
        "length": len(y) / sr,
        "volume": analyzer.volume(y),
        "pitch": analyzer.pitch(y, sr),
        "tempo": tempo,
        "attack": attack,
        "envelope": envelope,
        "onset_strength": onset_strength,
        "resonance": resonance,
        "notes": notes,
        "beat_frames": beat_frames,
        "top5_instruments": analyzer.predict_instrument_essentia(file),
        "file": file,
    }

    write_header = not exists(CSV_PATH)
    pd.DataFrame([feature_row]).to_csv(
        CSV_PATH,
        mode="a",
        index=False,
        header=write_header,
    )
