import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import urllib.request
import re
from os.path import join, dirname, basename

try:
    from essentia.standard import (
        MonoLoader,
        TensorflowPredictEffnetDiscogs,
        TensorflowPredict2D,
    )
    _HAS_ESSENTIA = True
except ImportError:
    MonoLoader = None
    TensorflowPredictEffnetDiscogs = None
    TensorflowPredict2D = None
    _HAS_ESSENTIA = False

# Configure TensorFlow logging and GPU memory growth
tf.get_logger().setLevel('ERROR')
# GPU detection and memory growth setting
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0].name}")
    except Exception as e:
        print(f"Could not set memory growth: {e}")
else:
    pass
    # print("No GPU found, running on CPU.")

# Define tags for instrument classification
tags = [
    "accordion", "acousticbassguitar", "acousticguitar", "bass", "beat", "bell", "bongo", "brass",
    "cello", "clarinet", "classicalguitar", "computer", "doublebass", "drummachine", "drums",
    "electricguitar", "electricpiano", "flute", "guitar", "harmonica", "harp", "horn", "keyboard",
    "oboe", "orchestra", "organ", "pad", "percussion", "piano", "pipeorgan", "rhodes", "sampler",
    "saxophone", "strings", "synthesizer", "trombone", "trumpet", "viola", "violin", "voice"
]

LOW_VIBRATO_NOTE_HINTS = {
    "words": {
        "flute",
        "clarinet",
        "horn",
        "piano",
        "trumpet",
        "trombone",
        "tuba",
        "bassoon",
    },
    "tokens": {"fl", "cl", "hn", "tpt", "tbn", "tba", "bn"},
}

HIGH_VIBRATO_NOTE_HINTS = {
    "words": {
        "violin",
        "viola",
        "cello",
        "doublebass",
        "oboe",
        "guitar",
        "saxophone",
    },
    "tokens": {"vn", "va", "vc", "db", "ob", "sax"},
}


class AudioFeatureExtraction:
    def __init__(self):
        self.loader = None
        self.embedding_model = None
        self.classifier_model = None
        if not _HAS_ESSENTIA:
            return

        # Essentia models
        self.loader = MonoLoader()
        embedding_filename = join(dirname(__file__), "discogs-effnet-bs64-1.pb")
        classifier_filename = join(dirname(__file__), "mtg_jamendo_instrument-discogs-effnet-1.pb")

        # Check if model files exist, if not use default models
        try:
            self.embedding_model = TensorflowPredictEffnetDiscogs(
                graphFilename=embedding_filename,
                output="PartitionedCall:1"
            )
            self.classifier_model = TensorflowPredict2D(
                graphFilename=classifier_filename
            )
            print("Using custom model files")
        except:
            # Use default models if custom files don't exist
            self.embedding_model = TensorflowPredictEffnetDiscogs()
            self.classifier_model = TensorflowPredict2D()
            print("Using default Essentia models")

    def load_audio(self, file_path):
        """
        Load audio and return time series and sample rate.
        """
        y, sr = librosa.load(file_path, sr=None)
        return y, sr

    def volume(self, y):
        """
        Compute global RMS energy.
        """
        rms_frames = librosa.feature.rms(
            y=y,
            frame_length=2048,
            hop_length=512
        )[0]
        return float(np.sqrt(np.mean(rms_frames ** 2)))

    def pitch(self, y, sr):
        """
        Estimate average F0 using Pyin.
        """
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            frame_length=2048,
            hop_length=512
        )
        return float(np.nanmean(f0))

    def _note_split_settings(self, instrument_hint=None):
        """
        Return note segmentation thresholds from a light instrument hint.
        Low-vibrato instruments keep shorter events; high-vibrato instruments
        require longer stability to avoid splitting vibrato into too many notes.
        """
        hint = "" if instrument_hint is None else str(instrument_hint).lower()
        tokens = set(filter(None, re.split(r"[^a-z0-9]+", hint)))
        compact_hint = re.sub(r"[^a-z0-9]+", "", hint)

        high_vibrato_match = (
            bool(tokens & HIGH_VIBRATO_NOTE_HINTS["tokens"])
            or any(word in compact_hint for word in HIGH_VIBRATO_NOTE_HINTS["words"])
        )
        if high_vibrato_match:
            return 0.12, 5

        low_vibrato_match = (
            bool(tokens & LOW_VIBRATO_NOTE_HINTS["tokens"])
            or any(word in compact_hint for word in LOW_VIBRATO_NOTE_HINTS["words"])
        )
        if low_vibrato_match:
            return 0.08, 3

        return 0.12, 5

    def _note_split_settings_from_audio(self, y, sr, f0, hop_length):
        """
        Runtime-safe note segmentation thresholds from audio only.

        This path intentionally ignores file names and directory names. Shorter,
        more separated voiced regions are treated like clear attacks; continuous
        voiced regions are treated more conservatively to avoid splitting
        vibrato/connected texture into too many notes.
        """
        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0 or f0 is None or len(f0) == 0:
            return 0.12, 5

        voiced = np.isfinite(f0)
        voiced_ratio = float(np.mean(voiced)) if len(voiced) else 0.0
        run_lengths = []
        current = 0
        for is_voiced in voiced:
            if is_voiced:
                current += 1
            elif current:
                run_lengths.append(current)
                current = 0
        if current:
            run_lengths.append(current)

        frame_seconds = hop_length / float(sr)
        median_voiced_run = (
            float(np.median(run_lengths)) * frame_seconds
            if run_lengths else 0.0
        )

        onset_times = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            units="time",
            hop_length=hop_length,
            backtrack=False,
        )
        onset_density = len(onset_times) / duration

        clear_attack_like = (
            voiced_ratio < 0.72
            or median_voiced_run < 0.28
            or (onset_density >= 2.2 and median_voiced_run < 0.45)
        )
        if clear_attack_like:
            return 0.08, 3
        return 0.12, 5

    def _pitch_change_frames(
        self,
        midi_pitch,
        threshold_semitones,
        min_stable_frames,
    ):
        """
        Find frames where the pitch has moved far enough and stayed stable.
        """
        split_frames = []
        reference_pitch = None
        frame = 0

        while frame < len(midi_pitch):
            window = midi_pitch[frame:frame + min_stable_frames]
            stable_values = window[np.isfinite(window)]
            if len(stable_values) < min_stable_frames:
                frame += 1
                continue

            window_pitch = float(np.median(stable_values))
            if reference_pitch is None:
                reference_pitch = window_pitch
                frame += min_stable_frames
                continue

            if abs(window_pitch - reference_pitch) >= threshold_semitones:
                split_frames.append(frame)
                reference_pitch = window_pitch
                frame += min_stable_frames
                continue

            frame += 1

        return split_frames

    def extract_notes(
        self,
        y,
        sr,
        instrument_hint=None,
        pitch_change_threshold_semitones=1.5,
    ):
        """
        Extract note events using stable pitch-change boundaries.
        """
        hop_length = 512
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            frame_length=2048,
            hop_length=hop_length
        )

        duration = librosa.get_duration(y=y, sr=sr)
        if f0 is None or len(f0) == 0 or duration <= 0:
            return []

        if instrument_hint is None:
            min_note_duration, min_stable_frames = self._note_split_settings_from_audio(
                y,
                sr,
                f0,
                hop_length,
            )
        else:
            min_note_duration, min_stable_frames = self._note_split_settings(
                instrument_hint
            )

        voiced_frames = np.where(np.isfinite(f0))[0]
        if len(voiced_frames) == 0:
            return []
        first_note_frame = int(voiced_frames[0])

        midi_pitch = librosa.hz_to_midi(f0)
        pitch_change_frames = self._pitch_change_frames(
            midi_pitch,
            pitch_change_threshold_semitones,
            min_stable_frames,
        )

        boundary_frames = []
        for frame in [first_note_frame] + sorted(set(pitch_change_frames)):
            if frame < first_note_frame:
                continue
            if boundary_frames:
                prev_time = librosa.frames_to_time(
                    boundary_frames[-1],
                    sr=sr,
                    hop_length=hop_length,
                )
                frame_time = librosa.frames_to_time(
                    frame,
                    sr=sr,
                    hop_length=hop_length,
                )
                if frame_time - prev_time < min_note_duration:
                    continue
            boundary_frames.append(frame)

        if not boundary_frames:
            boundary_frames = [first_note_frame]

        end_frame = len(f0)
        if boundary_frames[-1] < end_frame:
            boundary_frames.append(end_frame)

        notes = []
        for start_frame, end_frame in zip(boundary_frames[:-1], boundary_frames[1:]):
            start_time = librosa.frames_to_time(
                start_frame,
                sr=sr,
                hop_length=hop_length,
            )
            end_time = min(
                librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length),
                duration,
            )
            note_duration = float(end_time - start_time)
            if note_duration < min_note_duration:
                continue

            pitch_values = f0[start_frame:end_frame]
            pitch_values = pitch_values[np.isfinite(pitch_values)]
            if len(pitch_values) == 0:
                continue

            notes.append({
                "onset_time": float(start_time),
                "pitch": float(np.median(pitch_values)),
                "duration": note_duration
            })

        return notes

    def extract_features(self, file_path):
        """
        Extract a set of features including note events.
        """
        # Load
        y, sr = self.load_audio(file_path)
        # Volume
        volume = self.volume(y)
        # Pitch
        pitch = self.pitch(y, sr)
        # Onset envelope for attack/envelope dynamics
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=512,
            backtrack=False,
        )
        audio_onsets = [
            float(value)
            for value in librosa.frames_to_time(
                onset_frames,
                sr=sr,
                hop_length=512,
            )
        ]
        attack = 'sharp' if np.max(onset_env) > 0.5 else 'soft'
        envelope = 'transient' if attack == 'sharp' else 'sustained'
        onset_strength = float(np.mean(onset_env))
        # Resonance by duration
        duration = librosa.get_duration(y=y, sr=sr)
        resonance = 'long' if duration > 3.0 else 'short'
        # Note events
        notes = self.extract_notes(y, sr)
        try:
            instrument_predictions = self.predict_instrument_essentia(file_path)
        except Exception as e:
            print(f"Warning: instrument prediction failed; leaving instrument candidates empty: {e}")
            instrument_predictions = []

        top1_instrument = instrument_predictions[0][0] if instrument_predictions else None
        top1_prob = instrument_predictions[0][1] if instrument_predictions else None
        # Compile
        features = {
            "volume": volume,
            "pitch": pitch,
            "attack": attack,
            "envelope": envelope,
            "onset_strength": onset_strength,
            "audio_onsets": audio_onsets,
            "duration": float(duration),
            "resonance": resonance,
            "notes": notes,
            "top5_instruments": instrument_predictions,
            "top1_instrument": top1_instrument,
            "top1_prob": top1_prob
        }
        return features

    def predict_instrument_essentia(self, audio_path):
        """
        Essentia-based instrument classification using pre-trained models.
        """
        if not _HAS_ESSENTIA:
            return self.predict_instrument_librosa_fallback(audio_path)

        try:
            # Load audio using MonoLoader
            loader = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)
            audio = loader()

            # Get embeddings and predictions
            embeddings = self.embedding_model(audio)
            predictions = self.classifier_model(embeddings)
            mean_predictions = np.mean(predictions, axis=0)
            tag_score_pairs = list(zip(tags, mean_predictions))
            top5 = sorted(tag_score_pairs, key=lambda x: x[1], reverse=True)[:5]
            return [(name, float(score)) for name, score in top5]
        except Exception as e:
            print(f"Essentia prediction failed: {e}")
            # Fallback to librosa-based classification
            return self.predict_instrument_librosa_fallback(audio_path)

    def predict_instrument_librosa_fallback(self, audio_path):
        """
        Enhanced Librosa-based instrument classification using comprehensive features.
        """
        y, sr = self.load_audio(audio_path)

        # 1. Spectral Features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

        # 2. MFCC Features (for timbre analysis)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        # 3. Chroma Features (for harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)

        # 4. Temporal Features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

        # 5. Harmonic and Percussive Separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y_harmonic)) + np.sum(np.abs(y_percussive)))

        # 6. Onset Features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_rate = len(onset_frames) / librosa.get_duration(y=y, sr=sr)

        # Calculate comprehensive statistics
        features = {
            'mean_centroid': np.mean(spectral_centroids),
            'std_centroid': np.std(spectral_centroids),
            'mean_rolloff': np.mean(spectral_rolloff),
            'mean_bandwidth': np.mean(spectral_bandwidth),
            'mean_contrast': np.mean(spectral_contrast),
            'mean_flatness': np.mean(spectral_flatness),
            'mean_zcr': np.mean(zero_crossing_rate),
            'std_zcr': np.std(zero_crossing_rate),
            'tempo': tempo,
            'harmonic_ratio': harmonic_ratio,
            'onset_rate': onset_rate,
            'mfcc_1_mean': np.mean(mfccs[1]),  # First MFCC coefficient
            'mfcc_1_std': np.std(mfccs[1]),
            'chroma_std': np.std(chroma),
            'chroma_cqt_std': np.std(chroma_cqt)
        }

        # Enhanced classification using multiple feature combinations
        predictions = []

        # Piano detection (bright, sustained, moderate tempo, low ZCR variance)
        piano_score = self._calculate_piano_score(features)
        predictions.append(("piano", piano_score))

        # Guitar detection (moderate centroid, high contrast, moderate ZCR)
        guitar_score = self._calculate_guitar_score(features)
        predictions.append(("guitar", guitar_score))

        # Violin detection (high centroid, sustained, low ZCR)
        violin_score = self._calculate_violin_score(features)
        predictions.append(("violin", violin_score))

        # Bass detection (low centroid, low bandwidth, high harmonic ratio)
        bass_score = self._calculate_bass_score(features)
        predictions.append(("bass", bass_score))

        # Drums detection (high ZCR, high onset rate, low harmonic ratio)
        drums_score = self._calculate_drums_score(features)
        predictions.append(("drums", drums_score))

        # Voice detection (moderate centroid, moderate ZCR, high MFCC variance)
        voice_score = self._calculate_voice_score(features)
        predictions.append(("voice", voice_score))

        # Flute detection (high centroid, low ZCR, high flatness)
        flute_score = self._calculate_flute_score(features)
        predictions.append(("flute", flute_score))

        # Cello detection (low centroid, moderate bandwidth, high harmonic ratio)
        cello_score = self._calculate_cello_score(features)
        predictions.append(("cello", cello_score))

        # Sort by score and return top 5
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]

    def _calculate_piano_score(self, features):
        """Calculate piano likelihood score."""
        score = 0.0

        # Bright timbre (high centroid)
        if 1500 < features['mean_centroid'] < 3500:
            score += 0.4

        # Sustained notes (low ZCR variance)
        if features['std_zcr'] < 0.05:
            score += 0.3

        # Moderate tempo
        if 80 < features['tempo'] < 160:
            score += 0.2

        # High harmonic content
        if features['harmonic_ratio'] > 0.6:
            score += 0.2

        # Low spectral flatness (tonal rather than noisy)
        if features['mean_flatness'] < 0.1:
            score += 0.1

        return min(score, 0.95)

    def _calculate_guitar_score(self, features):
        """Calculate guitar likelihood score."""
        score = 0.0

        # Moderate centroid
        if 1000 < features['mean_centroid'] < 3000:
            score += 0.3

        # High spectral contrast (string characteristics)
        if features['mean_contrast'] > 5:
            score += 0.3

        # Moderate ZCR (plucked strings)
        if 0.05 < features['mean_zcr'] < 0.15:
            score += 0.2

        # Moderate onset rate
        if 0.5 < features['onset_rate'] < 3.0:
            score += 0.2

        # High harmonic content
        if features['harmonic_ratio'] > 0.5:
            score += 0.1

        return min(score, 0.9)

    def _calculate_violin_score(self, features):
        """Calculate violin likelihood score."""
        score = 0.0

        # High centroid (bright timbre)
        if features['mean_centroid'] > 3000:
            score += 0.4

        # Very low ZCR (sustained bowing)
        if features['mean_zcr'] < 0.03:
            score += 0.3

        # High rolloff
        if features['mean_rolloff'] > 4000:
            score += 0.2

        # High harmonic content
        if features['harmonic_ratio'] > 0.7:
            score += 0.2

        return min(score, 0.9)

    def _calculate_bass_score(self, features):
        """Calculate bass likelihood score."""
        score = 0.0

        # Low centroid
        if features['mean_centroid'] < 1000:
            score += 0.5

        # Low bandwidth
        if features['mean_bandwidth'] < 1500:
            score += 0.3

        # High harmonic content
        if features['harmonic_ratio'] > 0.6:
            score += 0.2

        # Low ZCR variance (sustained)
        if features['std_zcr'] < 0.03:
            score += 0.1

        return min(score, 0.95)

    def _calculate_drums_score(self, features):
        """Calculate drums likelihood score."""
        score = 0.0

        # High ZCR (percussive)
        if features['mean_zcr'] > 0.15:
            score += 0.4

        # High onset rate
        if features['onset_rate'] > 2.0:
            score += 0.3

        # Low harmonic ratio (percussive)
        if features['harmonic_ratio'] < 0.3:
            score += 0.3

        # High spectral flatness (noisy)
        if features['mean_flatness'] > 0.2:
            score += 0.2

        return min(score, 0.9)

    def _calculate_voice_score(self, features):
        """Calculate voice likelihood score."""
        score = 0.0

        # Moderate centroid (human voice range)
        if 800 < features['mean_centroid'] < 2500:
            score += 0.4

        # Moderate ZCR
        if 0.02 < features['mean_zcr'] < 0.08:
            score += 0.3

        # High MFCC variance (expressive)
        if features['mfcc_1_std'] > 2.0:
            score += 0.2

        # Moderate harmonic content
        if 0.4 < features['harmonic_ratio'] < 0.8:
            score += 0.2

        return min(score, 0.85)

    def _calculate_flute_score(self, features):
        """Calculate flute likelihood score."""
        score = 0.0

        # Very high centroid
        if features['mean_centroid'] > 3500:
            score += 0.4

        # Very low ZCR (sustained)
        if features['mean_zcr'] < 0.02:
            score += 0.3

        # High spectral flatness (pure tone)
        if features['mean_flatness'] > 0.3:
            score += 0.2

        # High harmonic content
        if features['harmonic_ratio'] > 0.8:
            score += 0.2

        return min(score, 0.9)

    def _calculate_cello_score(self, features):
        """Calculate cello likelihood score."""
        score = 0.0

        # Low centroid
        if features['mean_centroid'] < 1500:
            score += 0.4

        # Moderate bandwidth
        if 1000 < features['mean_bandwidth'] < 2500:
            score += 0.3

        # High harmonic content
        if features['harmonic_ratio'] > 0.7:
            score += 0.2

        # Low ZCR (sustained)
        if features['mean_zcr'] < 0.05:
            score += 0.2

        return min(score, 0.9)


if __name__ == "__main__":
    extractor = AudioFeatureExtraction()
    audio_path = "classic_test.wav"
    print("Loading audio file...")
    feats = extractor.extract_features(audio_path)
    print("Extracted Features:", feats)
    preds = extractor.predict_instrument_essentia(audio_path)
    print("Instrument Predictions:", preds)
