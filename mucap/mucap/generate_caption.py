import argparse, json, os, re, sys
from functools import lru_cache
from pathlib import Path
import pandas as pd
import _osx_support
from os import path
from os.path import join, basename, splitext, dirname
from dotenv import load_dotenv
from openai import OpenAI
from os.path import dirname, basename

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

try:
    from extract_spectogram import save_spectrogram
except ImportError:
    save_spectrogram = None
from mucap.audio import encode_audio, wav_to_mp3
from mucap.render.plot_render import render_plot_video
from mucap.render.circle_render import render_plot_video as render_circle
from mucap.render.audio_text_viz import visualize_captions
from mucap.render.audio_text_viz_video import visualize_captions_video
from mucap.render.audio_text_bounce_video import visualize_captions_bounce_video
from mucap.render.audio_text_karaoke_video import visualize_captions_karaoke_video
from mucap.render.audio_text_bounce_overlay_video import render_bounce_overlay
from mucap.render.audio_text_karaoke_overlay_video import render_karaoke_overlay
from mucap.render.audio_text_plain_video import visualize_captions_plain_video
from mucap.render.audio_text_plain_overlay_video import render_plain_overlay
from mucap.tokends2caption import tokens_to_caption
try:
    from mucap.video_render import render_video
except ImportError:
    render_video = None
try:
    from mucap.visualize import visualize
except ImportError:
    visualize = None
from mucap.caption_skeleton_patch import augment_feature_str
from music_features.feature_extractor import AudioFeatureExtraction


def select_haptic_preset(wav_path, feature_row):
    """Map MuCap caption behavior/texture classification to a haptic preset."""
    if not feature_row:
        return "generic"

    try:
        from mucap.caption_skeleton_patch import (
            _caption_behavior_from_wav,
            _connected_texture_style,
            _is_confident_string_context,
            _is_percussive_generic_context,
        )

        behavior = _caption_behavior_from_wav(str(wav_path), feature_row)
        if not behavior:
            return "generic"

        behavior_mode = behavior.get("mode")
        if behavior_mode == "clear-attack":
            return "percussive"

        if behavior_mode == "sustained-texture":
            string_mode = bool(_is_confident_string_context(feature_row))
            texture_style = _connected_texture_style(
                feature_row,
                behavior,
                string_mode,
            )
            if (
                texture_style == "generic-sustain"
                and _is_percussive_generic_context(
                    feature_row,
                    behavior,
                    wav_path=str(wav_path),
                )
            ):
                return "percussive"
            return {
                "string-like": "melodic",
                "wind-like": "ambience",
                "generic-sustain": "generic",
            }.get(texture_style, "generic")

        # Unknown modes use the unchanged general preset.
        return "generic"
    except Exception as exc:
        print(f"Warning: haptic preset classification failed; using generic: {exc}")
    return "generic"


def is_clear_attack_behavior(wav_path, feature_row) -> bool:
    """Read the actual caption behavior result without relying on prompt text."""
    if not feature_row:
        return False
    try:
        from mucap.caption_skeleton_patch import _caption_behavior_from_wav

        behavior = _caption_behavior_from_wav(str(wav_path), feature_row)
        return bool(behavior and behavior.get("mode") == "clear-attack")
    except Exception as exc:
        print(f"Warning: clear-attack validation mode unavailable: {exc}")
        return False


def is_percussive_generic_behavior(wav_path, feature_row) -> bool:
    """Identify the percussion-like generic-sustain branch used by the prompt."""
    if not feature_row:
        return False
    try:
        from mucap.caption_skeleton_patch import (
            _caption_behavior_from_wav,
            _connected_texture_style,
            _is_confident_string_context,
            _is_percussive_generic_context,
        )

        behavior = _caption_behavior_from_wav(str(wav_path), feature_row)
        if not behavior or behavior.get("mode") != "sustained-texture":
            return False
        string_mode = bool(_is_confident_string_context(feature_row))
        if _connected_texture_style(feature_row, behavior, string_mode) != "generic-sustain":
            return False
        return _is_percussive_generic_context(
            feature_row,
            behavior,
            wav_path=str(wav_path),
        )
    except Exception as exc:
        print(f"Warning: percussive-generic validation mode unavailable: {exc}")
        return False


def generate_haptic_output(wav_path, output_path, feature_row=None):
    """Create the haptic WAV beside the rendered MuCap videos."""
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from multimodal_caption.hybrid_test import convert_one

    source_wav = Path(wav_path).resolve()
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    haptic_path = output_dir / f"{source_wav.stem}-haptic.wav"
    preset_name = select_haptic_preset(source_wav, feature_row)
    print(f"Haptic classification: preset={preset_name}")

    convert_one(
        in_path=str(source_wav),
        out_path=str(haptic_path),
        preset_name=preset_name,
        quality="mid",
        strength=1.0,
        gain_low=1.0,
        gain_remap=1.0,
        sr_override=0,
        low_cut_override=0.0,
    )
    return haptic_path


def audio_feature_key(path_or_name):
    fname = basename(path_or_name)
    return fname[len("fixed_"):] if fname.startswith("fixed_") else fname


def audio_duration_seconds(wav_path, default=14.0):
    if not wav_path:
        return default
    try:
        import wave
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return default


def extract_features(audio_path, output_json, max_events=64):
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=512)
    peak = float(np.max(rms)) if len(rms) else 0.0
    norm = rms / peak if peak > 0 else rms

    if len(times) and max_events:
        idx = np.linspace(0, len(times) - 1, min(int(max_events), len(times)), dtype=int)
    else:
        idx = []

    payload = {
        "duration": duration,
        "events": [
            {"time": float(times[i]), "energy": float(norm[i])}
            for i in idx
        ],
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _caption_units(caption_text):
    matches = re.findall(r"\[([0-9]+(?:\.[0-9]+)?)\]\s*([^\[]+)", caption_text)
    units = [(float(t), txt.strip()) for t, txt in matches if txt.strip()]
    return units or [(0.0, caption_text.strip())]


def _wrap_text(draw, text, font, max_width):
    lines, current = [], ""
    for ch in text:
        trial = current + ch
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def render_text_from_json_no_moviepy(text, feature_json, audio_path, output_path,
                                     window_size=14):
    import os
    import subprocess
    import tempfile

    import imageio
    import imageio_ffmpeg
    import librosa
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    with open(feature_json, "r", encoding="utf-8") as f:
        feature_data = json.load(f)

    duration = float(feature_data.get("duration") or 0.0)
    if duration <= 0:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))

    events = feature_data.get("events", [])
    units = _caption_units(text)
    width, height, fps = 1280, 720, 24
    font_path = join(dirname(__file__), "fonts", "Pretendard-Bold.otf")
    font = ImageFont.truetype(font_path, 78) if os.path.exists(font_path) else ImageFont.load_default()
    small_font = ImageFont.truetype(font_path, 30) if os.path.exists(font_path) else ImageFont.load_default()

    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        writer = imageio.get_writer(temp_video, fps=fps, codec="libx264", quality=8)
        total_frames = max(1, int(duration * fps))
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            active = [txt for ts, txt in units if ts <= t]
            visible = "".join(active)[-window_size:] if active else units[0][1]

            energy = 0.0
            if events:
                nearest = min(events, key=lambda e: abs(float(e.get("time", 0.0)) - t))
                energy = float(nearest.get("energy", 0.0))
            pulse = int(40 + energy * 120)

            img = Image.new("RGB", (width, height), (8, 10, 16))
            draw = ImageDraw.Draw(img)
            draw.ellipse(
                (width // 2 - 190 - pulse, height // 2 - 190 - pulse,
                 width // 2 + 190 + pulse, height // 2 + 190 + pulse),
                outline=(80, 180, 255),
                width=10,
            )

            lines = _wrap_text(draw, visible, font, width - 220)
            line_h = 92
            y0 = height // 2 - (len(lines) * line_h) // 2
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=font)
                x = (width - (bbox[2] - bbox[0])) // 2
                draw.text((x, y0 + i * line_h), line, font=font, fill=(245, 248, 255))
            draw.text((42, height - 60), f"{t:05.2f}s", font=small_font, fill=(145, 160, 180))
            writer.append_data(np.asarray(img))
        writer.close()

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg, "-y",
            "-i", temp_video,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ], check=True)
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)


def retrieve_rag_context(query_text: str, top_k=2, threshold=0.8):
    """Retrieve relevant context using Chroma + OpenAI embeddings with similarity threshold."""

    persist_directory = join(dirname(__file__), 'rag', 'ragdb_skeleton')

    try:
        vectorstore = Chroma(persist_directory=persist_directory,
                             collection_name="mucap_rag_skeleton",
                             embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
    except BaseException as e:
        print(f"Warning: Chroma vector store failed, using embedding RAG fallback: {e}")
        return retrieve_rag_context_embedding_cache(query_text, persist_directory, top_k=top_k)
    
    # Check if collection exists and has documents
    try:
        collection = vectorstore._collection
        num_docs = collection.count()
        if num_docs == 0:
            print("Warning: No documents found in the vector store.")
            return None
    except Exception as e:
        print(f"Warning: Error checking vector store, using embedding RAG fallback: {e}")
        return retrieve_rag_context_embedding_cache(query_text, persist_directory, top_k=top_k)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": threshold}
    )

    try:
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query_text)
        else:
            docs = retriever.get_relevant_documents(query_text)
    except Exception as e:
        print(f"Warning: Error querying vector store, using embedding RAG fallback: {e}")
        return retrieve_rag_context_embedding_cache(query_text, persist_directory, top_k=top_k)
    if not docs:
        print(f"Warning: No documents found above similarity threshold {threshold}")
        return None

    output = ''
    for i, doc in enumerate(docs):
        output += f"[Annotation {i+1}] {doc.page_content}\n"

    return output


def retrieve_rag_context_embedding_cache(query_text: str, persist_directory: str, top_k=2):
    """Fallback RAG retrieval using text-embedding-3-large and cached vectors.

    The canonical path is Chroma over mucap_rag_skeleton. If a Windows Chroma
    build cannot open the persisted HNSW index, keep the same ragdb_skeleton
    document store but build a sidecar embedding cache with text-embedding-3-large
    and rank by cosine similarity.
    """
    import hashlib
    import sqlite3

    import numpy as np

    db_path = join(persist_directory, "chroma.sqlite3")
    if not path.exists(db_path):
        print(f"Warning: RAG SQLite DB not found: {db_path}")
        return None

    try:
        with sqlite3.connect(db_path) as con:
            rows = con.execute(
                """
                SELECT m.string_value
                FROM embedding_metadata AS m
                JOIN embeddings AS e ON e.id = m.id
                JOIN segments AS s ON s.id = e.segment_id
                JOIN collections AS c ON c.id = s.collection
                WHERE m.key = 'chroma:document'
                  AND c.name = 'mucap_rag_skeleton'
                  AND m.string_value IS NOT NULL
                ORDER BY e.id
                """
            ).fetchall()
    except Exception as e:
        print(f"Warning: embedding RAG fallback failed to read SQLite: {e}")
        return None

    docs = [doc_text for (doc_text,) in rows]
    if not docs:
        print("Warning: embedding RAG fallback found no documents.")
        return None

    cache_key = hashlib.sha256("\n".join(docs).encode("utf-8")).hexdigest()[:16]
    cache_path = join(persist_directory, f"embedding_cache_{cache_key}.npz")
    embeddings_client = OpenAIEmbeddings(model="text-embedding-3-large")

    if path.exists(cache_path):
        cache = np.load(cache_path)
        doc_embeddings = cache["embeddings"]
    else:
        print("Building RAG embedding cache with text-embedding-3-large...")
        batches = []
        batch_size = 64
        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            batches.extend(embeddings_client.embed_documents(batch))
        doc_embeddings = np.asarray(batches, dtype=np.float32)
        np.savez_compressed(cache_path, embeddings=doc_embeddings)

    query_embedding = np.asarray(
        embeddings_client.embed_query(query_text),
        dtype=np.float32,
    )

    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    denom = np.maximum(doc_norms * query_norm, 1e-12)
    scores = (doc_embeddings @ query_embedding) / denom
    top_indices = np.argsort(scores)[::-1][:top_k]

    output = ""
    for i, index in enumerate(top_indices):
        output += f"[Annotation {i+1}] {docs[int(index)]}\n"
    return output

def interpret_music_features(music_features):
    if not music_features:
        return "3) [Context]: - No music features available; rely only on the audio."

    # Convert strings to numbers
    def safe_float(value, default=0.0):
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default

    volume = safe_float(music_features.get('volume', 0))
    pitch = safe_float(music_features.get('pitch', 0))
    onset_strength = safe_float(music_features.get('onset_strength', 0))
    attack_raw = music_features.get('attack', '')
    envelope_raw = music_features.get(
        'envelope',
        music_features.get('envelop', ''),
    )
    resonance_raw = music_features.get('resonance', '')

    top5_instruments = music_features.get('top5_instruments', [])
    if isinstance(top5_instruments, str):
        # When loaded from CSV, it may be a string (e.g., "['piano', 'drums']")
        import ast
        try:
            top5_instruments = ast.literal_eval(top5_instruments)
        except (ValueError, SyntaxError):
            top5_instruments = []

    instrument_names = [inst[0] if isinstance(inst, (list, tuple)) else inst for inst in top5_instruments]
    instruments_str = ', '.join(instrument_names) if instrument_names else "Unknown"

    # Interpret feature values
    pitch_desc = "high pitch" if pitch > 0.6 else "mid pitch" if pitch > 0.3 else "low pitch"
    attack_text = str(attack_raw).strip().lower()
    if attack_text in {"sharp", "soft"}:
        attack_desc = f"{attack_text} attack"
    else:
        attack_value = safe_float(attack_raw)
        attack_desc = (
            "strong attack" if attack_value > 0.6
            else "moderate attack" if attack_value > 0.3
            else "soft attack"
        )
    volume_desc = "loud" if volume > 0.6 else "medium volume" if volume > 0.3 else "quiet"
    resonance_text = str(resonance_raw).strip().lower()
    if resonance_text in {"long", "short"}:
        resonance_desc = f"{resonance_text} resonance"
    else:
        resonance_value = safe_float(resonance_raw)
        resonance_desc = "long resonance" if resonance_value > 0.6 else "short resonance"
    envelope_text = str(envelope_raw).strip().lower() or "unknown"

    return (
        f"3) [Context]:\n"
        f"- Instruments used: {instruments_str}\n"
        f"- Interpreted audio traits: {volume_desc}, {pitch_desc}, {attack_desc}, {resonance_desc}\n"
        f"- Raw values: volume: {volume:.2f}, pitch: {pitch:.2f}, "
        f"attack: {attack_text or attack_raw}, envelope: {envelope_text}, "
        f"onset_strength: {onset_strength:.2f}, resonance: {resonance_text or resonance_raw}"
    )

def build_prompt_rhythm(audio_b64: str, design_prompt=None, music_features=None,
                        use_rag=None, wav_path=None):

    duration = audio_duration_seconds(wav_path)
    end_timestamp = max(0.0, duration - 0.2)

    if music_features is not None:
        feature_str = interpret_music_features(music_features)
    else:
        feature_str = "3) [Context]: - No music features available; rely only on the audio."

    if wav_path:
        feature_str = augment_feature_str(
            feature_str,
            wav_path,
            runtime_features=music_features,
        )

    if use_rag:
        # print("🔍 Retrieving extra context from vector DB...")
        rag_context = retrieve_rag_context(feature_str, threshold=0.1)  # Lowered threshold to 0.1

        if rag_context is not None:
            feature_str += (
                "\n4) [Onomatopoeia annotations of similar audio] "
                "(use only as reference and adapt to the provided music): "
                "The instruments and annotations in these retrieved examples belong "
                "to reference clips, not necessarily to the current audio. Do not "
                "treat retrieved instrument names as the instrument identity or style "
                "mode of the current audio; use the current audio analysis above for "
                "style decisions. "
                f"{rag_context}\n"
            )

    text = (
        "You are an audio subtitle writer. Listen to the audio and create sound-mimetic "
        "captions using Hangul Korean onomatopoeia only.\n"
        f"Audio duration: {duration:.2f} seconds. End timestamps must be between "
        f"0.00 and {end_timestamp:.2f} seconds.\n"
        f"{feature_str}\n"
        "Output rules are strict:\n"
        "- Return exactly one line and nothing else.\n"
        "- Every caption unit must match this pattern: [seconds] HangulText\n"
        "- Square brackets around every timestamp are mandatory. Format-only example: "
        "[0.00] 가 [0.45] 나. The placeholder syllables 가 and 나 demonstrate syntax "
        "only and are not sound-word vocabulary examples.\n"
        "- Separate caption units with spaces and keep every timestamp immediately "
        "before its Hangul caption token.\n"
        "- Use only Hangul Korean onomatopoeia for musical instrument sounds.\n"
        "- Choose instrumental syllables from the audible pitch, register, duration, "
        "loudness, attack, resonance, articulation, and rests. Do not use human cries, "
        "speech-like exclamations, impact-only effects, or electronic beeps unless they "
        "are genuinely produced as part of the instrument sound.\n"
        "- Do not write explanations, greetings, markdown, English words, romanized Korean, "
        "Japanese, Chinese characters, Hanja, adjectives, or extra symbols.\n"
        "- Do not say you will analyze the audio. Directly output the timestamped caption line.\n"
        "- Use as many units as needed to match audible rhythm and sound events.\n"
        "- If the context includes a Rapid onset hint, follow it with repeated close-timestamp Hangul units for clearly separated attacks.\n"
        "- Repetition is allowed only when the notes are genuinely uniform. Do not let "
        "one-syllable tokens dominate a passage whose notes have audible sustain or "
        "resonance. When pitch, register, duration, loudness, attack, resonance, "
        "articulation, or rests change, reflect those changes in the consonant, vowel, "
        "and token length instead of repeating one token or alternating a fixed pair. "
        "Do not add variety unsupported by the audio.\n"
        "- If unsure, still output a short valid caption line with timestamps.\n"
        f"Optional style guide: {design_prompt or 'No guide'}"
        # "6) [Reference rules]\n"
        # "- For loud and short sounds, use sharp transient syllables like 'ttak', 'tchak', 'jjang'"
        # "- For low, sustained tones, use forms like 'doong', 'woong', 'boong'"
        # "- For rapid repetition, repeat consonants ('tta-tta', 'ta-ka-ta-ka', 'ppa-ba-ba')"
        # "- For airy high notes, include an h sound such as 'hwi', 'hi-ing', 'shi-ing'"
        # "- Represent beat divisions with spaces and emphasize using doubled consonants ('ttan', 'jjan', 'ppam')"
    )

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "input_audio",
                 "input_audio": {"data": audio_b64, "format": "mp3"}}
            ],
        }
    ]


def llm_api(model, messages, temp=0.7):
    client = OpenAI()
    r = client.chat.completions.create(
        model=model, messages=messages, temperature=temp,
        max_tokens=2048, top_p=1, frequency_penalty=0, presence_penalty=0
    )

    return r.choices[0].message.content.strip()


# ─────────────────────── Overall execution flow ─────────────────────── #
def parse_caption(line: str):
    token_regex = re.compile(r"\[(\d+\.?\d*)\]\s*([^\s]+)")
    return [(float(t), tok) for t, tok in token_regex.findall(line)]


def _is_hangul_token(token: str) -> bool:
    return bool(re.fullmatch(r"[가-힣]+", token))


@lru_cache(maxsize=128)
def _audio_activity_profile(wav_path: str):
    """Cache onset times and RMS activity used by mid-gap validation."""
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    hop_length = 512
    onset_envelope = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,
    )
    onset_times = tuple(
        float(value)
        for value in librosa.frames_to_time(
            onset_frames,
            sr=sr,
            hop_length=hop_length,
        )
    )
    onset_strengths = tuple(
        float(onset_envelope[int(frame)])
        for frame in onset_frames
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=hop_length,
    )[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length,
    )
    peak = float(np.max(rms)) if len(rms) else 0.0
    activity_threshold = peak * 0.06
    return (
        onset_times,
        onset_strengths,
        tuple(float(value) for value in rms_times),
        tuple(float(value) for value in rms),
        activity_threshold,
        peak,
    )


def active_mid_gap_validation_error(
    tokens,
    wav_path: str,
    min_caption_gap: float = 4.0,
    min_onsets: int = 3,
    min_active_ratio: float = 0.35,
) -> str | None:
    """Reject long caption gaps only when the underlying audio remains active."""
    ordered = sorted(tokens, key=lambda item: item[0])
    if len(ordered) < 2:
        return None

    try:
        onset_times, _, rms_times, rms_values, activity_threshold, _ = (
            _audio_activity_profile(str(Path(wav_path).resolve()))
        )
    except Exception as exc:
        print(f"Warning: mid-gap audio validation unavailable: {exc}")
        return None

    for (left_time, _), (right_time, _) in zip(ordered, ordered[1:]):
        gap = right_time - left_time
        if gap < min_caption_gap:
            continue

        # Ignore the immediate edges because the caption at either boundary can
        # legitimately describe a short attack or resonance around that timestamp.
        inner_start = left_time + 0.5
        inner_end = right_time - 0.5
        if inner_end <= inner_start:
            continue

        onset_count = sum(
            1 for onset in onset_times
            if inner_start <= onset <= inner_end
        )
        if onset_count < min_onsets:
            continue

        interval_rms = [
            value for timestamp, value in zip(rms_times, rms_values)
            if inner_start <= timestamp <= inner_end
        ]
        if not interval_rms or activity_threshold <= 0:
            continue

        active_ratio = (
            sum(value >= activity_threshold for value in interval_rms)
            / len(interval_rms)
        )
        if active_ratio >= min_active_ratio:
            return (
                f"caption skips active audio from {left_time:.2f}s to "
                f"{right_time:.2f}s ({gap:.2f}s gap, {onset_count} onsets, "
                f"{active_ratio:.0%} active)"
            )

    return None


def near_silent_caption_times(
    tokens,
    wav_path: str,
    silent_onset_tolerance: float = 0.25,
    silent_db_threshold: float = -32.0,
    extreme_silence_db_threshold: float = -40.0,
) -> list[float]:
    """Return captions in extreme silence or low energy without a nearby onset."""
    import numpy as np

    try:
        (
            onset_times,
            onset_strengths,
            rms_times,
            rms_values,
            _,
            peak_rms,
        ) = _audio_activity_profile(str(Path(wav_path).resolve()))
    except Exception as exc:
        print(f"Warning: onset/RMS caption validation unavailable: {exc}")
        return []

    if peak_rms <= 0 or not rms_values:
        return []

    silent_caption_times = []
    for timestamp, _ in tokens:
        local_rms = [
            value for time, value in zip(rms_times, rms_values)
            if timestamp - 0.12 <= time <= timestamp + 0.35
        ]
        if not local_rms:
            continue
        local_peak = max(local_rms)
        local_db = 20.0 * np.log10(max(local_peak, 1e-12) / peak_rms)
        nearest_onset = (
            min(abs(timestamp - onset) for onset in onset_times)
            if onset_times else float("inf")
        )
        is_extreme_silence = local_db <= extreme_silence_db_threshold
        is_low_energy_without_onset = (
            local_db <= silent_db_threshold
            and nearest_onset > silent_onset_tolerance
        )
        if is_extreme_silence or is_low_energy_without_onset:
            silent_caption_times.append(timestamp)

    return silent_caption_times


def remove_near_silent_captions(caption: str, wav_path: str) -> str:
    """Remove only captions confidently placed in near-silent audio."""
    tokens = parse_caption(caption)
    if not tokens:
        return caption

    silent_times = near_silent_caption_times(tokens, wav_path)
    if not silent_times:
        return caption

    silent_keys = {round(timestamp, 6) for timestamp in silent_times}
    kept = [
        (timestamp, token)
        for timestamp, token in tokens
        if round(timestamp, 6) not in silent_keys
    ]
    removed_text = ", ".join(f"{timestamp:.2f}s" for timestamp in silent_times)
    print(f"Removed near-silent captions at: {removed_text}")
    return " ".join(
        f"[{timestamp:.2f}] {token}"
        for timestamp, token in kept
    )


def onset_rms_validation_error(
    tokens,
    wav_path: str,
    clear_attack_mode: bool,
    percussive_generic_mode: bool = False,
    caption_onset_tolerance: float = 0.20,
    min_clear_attack_coverage: float = 0.75,
    max_consecutive_missed_gap: float = 0.75,
    max_percussive_active_gap: float = 1.25,
) -> str | None:
    """Cross-check captions against audible onsets and local RMS energy."""
    import numpy as np

    silent_caption_times = near_silent_caption_times(tokens, wav_path)
    if silent_caption_times:
        times = ", ".join(f"{value:.2f}s" for value in silent_caption_times[:8])
        return (
            "captions are placed in near-silent audio without a nearby onset at "
            f"{times}"
        )

    try:
        (
            onset_times,
            onset_strengths,
            rms_times,
            rms_values,
            activity_threshold,
            _,
        ) = _audio_activity_profile(
            str(Path(wav_path).resolve())
        )
    except Exception as exc:
        print(f"Warning: onset caption validation unavailable: {exc}")
        return None

    if (
        not (clear_attack_mode or percussive_generic_mode)
        or not onset_times
        or not onset_strengths
    ):
        return None

    strength_threshold = float(np.percentile(onset_strengths, 60))
    strong_onsets = [
        onset for onset, strength in zip(onset_times, onset_strengths)
        if strength >= strength_threshold
    ]
    if not strong_onsets:
        return None

    caption_times = [timestamp for timestamp, _ in tokens]
    missed = [
        onset for onset in strong_onsets
        if not any(
            abs(onset - caption_time) <= caption_onset_tolerance
            for caption_time in caption_times
        )
    ]
    coverage = 1.0 - (len(missed) / len(strong_onsets))
    missed_set = set(missed)

    if percussive_generic_mode:
        consecutive_three_missed = next(
            (
                group
                for group in zip(
                    strong_onsets,
                    strong_onsets[1:],
                    strong_onsets[2:],
                )
                if all(onset in missed_set for onset in group)
            ),
            None,
        )
        if consecutive_three_missed:
            times = ", ".join(
                f"{onset:.2f}s" for onset in consecutive_three_missed
            )
            return (
                "percussive-generic captions miss three consecutive strong "
                f"attacks at {times}; place captions within "
                f"{caption_onset_tolerance:.2f}s of the audible attacks"
            )

        duration = audio_duration_seconds(wav_path)
        boundaries = [0.0] + sorted(caption_times) + [duration]
        for left, right in zip(boundaries, boundaries[1:]):
            if right - left < max_percussive_active_gap:
                continue
            inner_start = left + caption_onset_tolerance
            inner_end = right - caption_onset_tolerance
            strong_inside = [
                onset for onset in strong_onsets
                if inner_start <= onset <= inner_end
            ]
            if len(strong_inside) < 3:
                continue
            interval_rms = [
                value for timestamp, value in zip(rms_times, rms_values)
                if inner_start <= timestamp <= inner_end
            ]
            active_ratio = (
                sum(value >= activity_threshold for value in interval_rms)
                / len(interval_rms)
                if interval_rms else 0.0
            )
            if active_ratio >= 0.35:
                return (
                    "percussive-generic captions leave active strikes without "
                    f"a caption from {left:.2f}s to {right:.2f}s "
                    f"({right - left:.2f}s gap, {len(strong_inside)} strong "
                    f"onsets, {active_ratio:.0%} active)"
                )
        return None

    consecutive_missed = next(
        (
            (left, right)
            for left, right in zip(strong_onsets, strong_onsets[1:])
            if (
                left in missed_set
                and right in missed_set
                and right - left <= max_consecutive_missed_gap
            )
        ),
        None,
    )
    if coverage < min_clear_attack_coverage or consecutive_missed:
        times = ", ".join(f"{value:.2f}s" for value in missed[:8])
        consecutive_text = ""
        if consecutive_missed:
            consecutive_text = (
                f"; consecutive strong attacks are both missing at "
                f"{consecutive_missed[0]:.2f}s and {consecutive_missed[1]:.2f}s"
            )
        return (
            f"clear-attack caption coverage is only {coverage:.0%}; strong audible "
            f"attacks lack nearby captions around {times}{consecutive_text}"
        )

    return None


def caption_validation_error(
    caption: str,
    wav_path: str,
    clear_attack_mode: bool = False,
    percussive_generic_mode: bool = False,
) -> str | None:
    if "..." in caption or "…" in caption:
        return "ellipsis/omission marker is not allowed"
    if re.search(r"\[\d+\.?\d*\]\s*$", caption):
        return "final timestamp has no Hangul caption token"

    tokens = parse_caption(caption)
    if not tokens:
        return "no timestamped caption tokens"

    first_time = min(t for t, _ in tokens)
    last_time = max(t for t, _ in tokens)
    duration = audio_duration_seconds(wav_path)

    if first_time > 0.35:
        return "caption omits the beginning"
    if duration > 4.0 and last_time < duration * 0.95:
        return "caption ends too early"

    seen_times = set()
    for t, _ in tokens:
        key = round(t, 2)
        if key in seen_times:
            return "duplicate timestamp found"
        seen_times.add(key)

    bad_tokens = [tok for _, tok in tokens if not _is_hangul_token(tok)]
    if bad_tokens:
        return f"non-Hangul token found: {bad_tokens[0]}"

    mid_gap_error = active_mid_gap_validation_error(tokens, wav_path)
    if mid_gap_error:
        return mid_gap_error

    onset_rms_error = onset_rms_validation_error(
        tokens,
        wav_path,
        clear_attack_mode=clear_attack_mode,
        percussive_generic_mode=percussive_generic_mode,
    )
    if onset_rms_error:
        return onset_rms_error

    return None


def append_caption_retry_instruction(messages, reason: str, wav_path: str) -> None:
    correction = (
        "\n\nCorrection required. The previous caption was invalid because: "
        f"{reason}. Regenerate from scratch. Do not summarize or use ellipses. "
        "Every timestamp must be enclosed in square brackets, exactly like "
        "[0.00] 가 [0.45] 나. Use only Hangul syllables inside caption tokens. "
        "Keep the caption natural and follow the audible rhythm from beginning to end. "
        "Do not leave a multi-second timestamp gap when audible attacks and active "
        "instrument sound continue inside that interval; add captions near the "
        "meaningful audible changes. Long gaps are allowed only for genuine silence "
        "or one genuinely unchanged sustained sound."
    )
    for content in messages[0]["content"]:
        if content.get("type") == "text":
            content["text"] += correction
            return


def sanitize_caption(caption: str, wav_path: str) -> str:
    tokens = parse_caption(caption)
    if not tokens:
        loose_regex = re.compile(r"\[?(\d+\.?\d*)\]?\s*([가-힣]+)")
        tokens = [
            (float(timestamp), token)
            for timestamp, token in loose_regex.findall(caption)
        ]
    duration = audio_duration_seconds(wav_path)
    end_limit = max(0.0, duration - 0.2)

    if not tokens:
        return "[0.00] 딩"

    cleaned = []
    prev_t = -1.0
    for t, tok in sorted(tokens, key=lambda item: item[0]):
        if not _is_hangul_token(tok):
            tok = "딩"
        t = max(0.0, min(float(t), end_limit))
        if t <= prev_t:
            adjusted = prev_t + 0.05
            if adjusted > end_limit:
                continue
            t = adjusted
        cleaned.append((t, tok))
        prev_t = t

    cleaned.sort(key=lambda item: item[0])
    return " ".join(f"[{t:.2f}] {tok}" for t, tok in cleaned)


def generate_valid_caption(
    model,
    messages,
    wav_path: str,
    max_attempts: int = 3,
    clear_attack_mode: bool | None = None,
    percussive_generic_mode: bool = False,
) -> str:
    caption = ""
    if clear_attack_mode is None:
        prompt_text = " ".join(
            str(content.get("text", ""))
            for content in messages[0].get("content", [])
            if content.get("type") == "text"
        )
        clear_attack_mode = "Behavior mode: clear-attack" in prompt_text
    for attempt in range(max_attempts):
        caption = llm_api(model, messages, temp=0.5 if attempt else 0.7)
        caption = remove_near_silent_captions(caption, wav_path)
        reason = caption_validation_error(
            caption,
            wav_path,
            clear_attack_mode=clear_attack_mode,
            percussive_generic_mode=percussive_generic_mode,
        )
        if reason is None:
            return caption
        append_caption_retry_instruction(messages, reason, wav_path)
    return sanitize_caption(caption, wav_path)


def caption_ends_too_early(caption: str, wav_path: str,
                           min_coverage: float = 0.95) -> bool:
    tokens = parse_caption(caption)
    if not tokens:
        return False

    duration = audio_duration_seconds(wav_path)
    if duration <= 4.0:
        return False

    last_time = max(t for t, _ in tokens)
    return last_time < duration * min_coverage


def generate_caption_with_end_retry(model, messages, wav_path: str) -> str:
    caption = llm_api(model, messages)
    if caption_ends_too_early(caption, wav_path):
        append_caption_retry_instruction(messages, "caption ends too early", wav_path)
        caption = llm_api(model, messages, temp=0.5)
    return caption


def annotate_and_show(wav_path, use_design_prompt=True,
                      model="gpt-audio-1.5",
                      use_audio_feature=True,
                      use_rag=True,
                      output_path=None,
                      postfix=None,
                      verbose=False,
                      caption_only=False):

    if output_path is None:
        output_path = dirname(wav_path)
    os.makedirs(output_path, exist_ok=True)

    mp3_path = wav_to_mp3(wav_path)
    audio_b64 = encode_audio(mp3_path)

    if use_design_prompt:
        design_guide_path = join(dirname(__file__), 'prompt', 'design_guide.txt')
        with open(design_guide_path, "r", encoding="utf-8") as f:
            design_txt = f.read().strip()
    else:
        design_txt = None

    if use_audio_feature:
        # lo
        feature_path = join(dirname(__file__), '..', 'music_features.csv')
        audio_feature_df = pd.read_csv(feature_path)
        wav_filename = audio_feature_key(wav_path)

        is_audio_feature_available = wav_filename in audio_feature_df['file_name'].values

        if is_audio_feature_available:
            print(f"Using pre-extracted audio features for {wav_filename}")
            audio_feature_df = audio_feature_df[audio_feature_df['file_name'] == wav_filename].iloc[0]
            audio_feature_df = audio_feature_df.to_dict()
            features = {
                'volume': audio_feature_df['volume'] if 'volume' in audio_feature_df else None,
                'pitch': audio_feature_df['pitch'] if 'pitch' in audio_feature_df else None,
                'attack': audio_feature_df['attack'] if 'attack' in audio_feature_df else None,
                'envelope': audio_feature_df['envelope'] if 'envelope' in audio_feature_df else None,
                'onset_strength': audio_feature_df['onset_strength'] if 'onset_strength' in audio_feature_df else None,
                'audio_onsets': audio_feature_df['audio_onsets'] if 'audio_onsets' in audio_feature_df else [],
                'duration': audio_feature_df['duration'] if 'duration' in audio_feature_df else None,
                'resonance': audio_feature_df['resonance'] if 'resonance' in audio_feature_df else None,
                'notes': audio_feature_df['notes'] if 'notes' in audio_feature_df else [],
            }
            instrument_predictions = audio_feature_df['top5_instruments'] if 'top5_instruments' in audio_feature_df else None
        else:
            print(f"Audio features not found for {wav_filename}, extracted features instead.")
            audio_extractor = AudioFeatureExtraction()

            features = audio_extractor.extract_features(wav_path)
            try:
                instrument_predictions = audio_extractor.predict_instrument_essentia(wav_path)
            except Exception as e:
                print(f"Warning: instrument prediction failed; using piano fallback: {e}")
                instrument_predictions = [("piano", 1.0)]


        music_features = {'volume': features['volume'],
                          'pitch': features['pitch'],
                          'attack': features['attack'],
                          'envelope': features['envelope'],
                          'envelop': features['envelope'],
                          'onset_strength': features['onset_strength'],
                          'audio_onsets': features.get('audio_onsets', []),
                          'duration': features.get('duration'),
                          'resonance': features['resonance'],
                          'top5_instruments': instrument_predictions,
                          'notes': features.get('notes', [])}
    else:
        music_features = None

    messages = build_prompt_rhythm(audio_b64, design_txt, music_features, use_rag, wav_path=wav_path)
    
    # text_only = next(
    # (item['text'] for item in messages[0]['content'] if item['type'] == 'text'), None
    # )
    # print("🔎 Prompt length:", len(text_only.split()) if text_only else "N/A")
    # print("🔍 Prompt preview:\n", text_only[:500], "...")  # Preview only the beginning

    clear_attack_mode = is_clear_attack_behavior(wav_path, music_features)
    percussive_generic_mode = is_percussive_generic_behavior(
        wav_path,
        music_features,
    )
    caption = generate_valid_caption(
        model,
        messages,
        wav_path,
        clear_attack_mode=clear_attack_mode,
        percussive_generic_mode=percussive_generic_mode,
    )

    messages[0]["content"] = [c for c in messages[0]["content"]
                              if c["type"] != "input_audio"]
    out_json = {
        "input": messages,
        "output": {"audio": mp3_path, "caption": caption}
    }

    no_extension = basename(wav_path).rsplit('.', 1)[0]
    with open(join(output_path, f"{no_extension}{f'-{postfix}' if postfix else ''}.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    tokens = parse_caption(caption)
    caption = tokens_to_caption(tokens)

    no_extension = basename(wav_path).rsplit('.', 1)[0]
    with open(join(output_path, f"{no_extension}{f'-{postfix}' if postfix else ''}.txt"), "w", encoding="utf-8") as f:
        f.write(caption)

    if caption_only:
        return

    try:
        html_path = join(
            output_path,
            f"{no_extension}{f'-{postfix}' if postfix else ''}-audio_text_viz.html",
        )
        visualize_captions(
            wav_path,
            tokens,
            out_path=html_path,
            feature_row=music_features,
            audio_path=wav_path,
            copy_audio=True,
        )
        video_path = join(
            output_path,
            f"{no_extension}{f'-{postfix}' if postfix else ''}-audio_text_viz_direct.mp4",
        )
        visualize_captions_video(
            wav_path,
            tokens,
            out_path=video_path,
            feature_row=music_features,
            audio_path=wav_path,
            html_path=html_path,
        )
        if verbose:
            print(f"Saved audio text visualization: {html_path}")
            print(f"Saved audio text visualization video: {video_path}")
    except Exception as e:
        print(f"Warning: audio text visualization failed: {e}")

    try:
        bounce_video_path = join(
            output_path,
            f"{no_extension}{f'-{postfix}' if postfix else ''}-audio_text_bounce.mp4",
        )
        visualize_captions_bounce_video(
            wav_path,
            tokens,
            out_path=bounce_video_path,
            feature_row=music_features,
            audio_path=wav_path,
        )
        bounce_overlay_path = join(
            output_path,
            f"{no_extension}{f'-{postfix}' if postfix else ''}"
            "-audio_text_bounce_overlay.mp4",
        )
        render_bounce_overlay(
            wav_path,
            bounce_video_path,
            bounce_overlay_path,
        )
        if verbose:
            print(f"Saved audio text bounce video: {bounce_video_path}")
    except Exception as e:
        print(f"Warning: audio text bounce visualization failed: {e}")

    try:
        karaoke_video_path = join(
            output_path,
            f"{no_extension}{f'-{postfix}' if postfix else ''}-audio_text_karaoke.mp4",
        )
        visualize_captions_karaoke_video(
            wav_path,
            tokens,
            out_path=karaoke_video_path,
            feature_row=music_features,
            audio_path=wav_path,
        )
        karaoke_overlay_path = join(
            output_path,
            f"{no_extension}{f'-{postfix}' if postfix else ''}"
            "-audio_text_karaoke_overlay.mp4",
        )
        render_karaoke_overlay(
            wav_path,
            karaoke_video_path,
            karaoke_overlay_path,
        )
        if verbose:
            print(f"Saved audio text karaoke video: {karaoke_video_path}")
    except Exception as e:
        print(f"Warning: audio text karaoke visualization failed: {e}")

    try:
        plain_video_path = join(
            output_path,
            f"{no_extension}{f'-{postfix}' if postfix else ''}-audio_text_plain.mp4",
        )
        visualize_captions_plain_video(
            wav_path,
            tokens,
            out_path=plain_video_path,
            feature_row=music_features,
            audio_path=wav_path,
        )
        plain_overlay_path = join(
            output_path,
            f"{no_extension}{f'-{postfix}' if postfix else ''}"
            "-audio_text_plain_overlay.mp4",
        )
        render_plain_overlay(
            wav_path,
            plain_video_path,
            plain_overlay_path,
        )
        if verbose:
            print(f"Saved plain audio text video: {plain_video_path}")
    except Exception as e:
        print(f"Warning: plain audio text visualization failed: {e}")


    # visualize(mp3_path, output_path, tokens, font_dir='fonts', postfix=postfix, verbose=verbose)
    # render_video(mp3_path, tokens, output_path, hop=0.25, font_dir='fonts', fps=30, postfix=postfix, verbose=verbose)
    # render_plot_video(mp3_path, tokens, output_path, hop=0.25, font_dir='fonts', fps=30, postfix="postfix", verbose=verbose)

    # Keep the legacy render implementations available, but do not run them
    # during the standard MuCap V2 pipeline unless explicitly re-enabled here.
    enable_legacy_renders = False
    skip_legacy_renders = not enable_legacy_renders
    if not skip_legacy_renders:
        render_plot_video(
            wav_path=wav_path,
            tokens=tokens,
            output_path=output_path,
            text_pop_effect=False,
            progress_bar=True,
            font_dir="fonts",
            postfix="type1"
        )
        render_plot_video(
            wav_path=wav_path,
            tokens=tokens,
            output_path=output_path,
            text_pop_effect=True,
            progress_bar=False,
            font_dir="fonts",
            postfix="type3"
        )
        render_circle(
            wav_path=wav_path,
            tokens=tokens,
            output_path=output_path,
            progress_bar=False,
            postfix="circle_grad",
            bouncing_mode=False
        )
        render_circle(
            wav_path=wav_path,
            tokens=tokens,
            output_path=output_path,
            progress_bar=False,
            size_effect=False,
            bouncing_mode=True,
            postfix="circle_flat",
            hide_axes=True,
        )
        render_circle(
            wav_path=wav_path,
            tokens=tokens,
            output_path=output_path,
            progress_bar=True,
            size_effect=False,
            bouncing_mode=False,
            postfix="circle_prog",
            circle_radius_factor=0.8
        )
    else:
        print("Skipping disabled type1/type2/type3/type4 and circle renders.")

    base_name = splitext(basename(wav_path))[0]
    feature_json = join(
        dirname(output_path),
        f"{base_name}-{postfix}-features.json"
    )
    
    if not skip_legacy_renders:
        extract_features(
            audio_path=wav_path,
            output_json=feature_json,
            max_events=len(caption)  # Match the number of text characters
        )

        render_text_from_json_no_moviepy(
            text=caption,
            feature_json=feature_json,
            audio_path=wav_path,
            output_path=join(output_path, f"{base_name}-type2.mp4"),
            window_size=14  # letters per screen
        )
    # render_subtitle(
    #     wav_path=wav_path,
    #     tokens=tokens,
    #     output_path=output_path,
    #     text_pop_effect=True,
    #     progress_bar=False,
    #     font_dir="fonts",
    #     postfix="type2",
    #     window_size=7
    # )

    # Add annotation_render video (static text video) with human annotation
    from mucap.render.annotation_render import annotation_render
    
    # Load human annotation data
    annotation_path = join(dirname(__file__), "..", "human_annotation", "human_annotation.json")
    
    # Extract filename from wav_path to match with annotation data
    wav_filename = audio_feature_key(wav_path)
    
    # Find matching human annotation
    human_annotation_text = None
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
            
        for item in annotation_data:
            audio_url = item['data']['audio']
            # Extract filename from URL
            url_filename = audio_feature_key(audio_url.split('/')[-1])
            
            if url_filename == wav_filename:
                # Get the human annotation text
                if item['annotations'] and item['annotations'][0]['result']:
                    for result in item['annotations'][0]['result']:
                        if 'text' in result['value']:
                            human_annotation_text = result['value']['text'][0]
                            break
                break
    except Exception as e:
        print(f"⚠ Warning: Could not load human annotation: {e}")
        human_annotation_text = None
    
    # Use human annotation if available, otherwise use LLM caption
    display_text = human_annotation_text if human_annotation_text else caption
    
    if not skip_legacy_renders:
        annotation_render(
            wav_path=wav_path,
            output_path=output_path,
            display_text=display_text,
            font_dir='fonts',
            fps=1,
            dark_mode=True,
            postfix="type4"
        )

    if save_spectrogram is not None:
        save_spectrogram(mp3_path, output_path, sr=44100, postfix=postfix)
    else:
        print("Warning: extract_spectogram.py not found; skipping spectrogram output.")

    haptic_path = generate_haptic_output(wav_path, output_path, music_features)
    print(f"Saved haptic WAV: {haptic_path}")



# ────────────────────────────── CLI ───────────────────────────── #
if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("yes", "true", "t", "1")

    parser.add_argument("--input_file", required=True, help="Path to the WAV file")
    parser.add_argument("--output_path", default=None, help="Output path (default: same folder as WAV file)")
    parser.add_argument("--design_prompt", default="design_guide.txt")
    parser.add_argument("--use_audio_feature", type=str2bool, default=True, help="Whether to use audio features")
    parser.add_argument("--use_rag", type=str2bool, default=True, help="Whether to use RAG")
    parser.add_argument(
        "--caption_only",
        type=str2bool,
        default=False,
        help="Only write caption JSON/TXT and skip MuCap visualization MP4/PNG rendering.",
    )

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = dirname(args.input_file)

    if not args.input_file.lower().endswith(".wav"):
        raise ValueError("Only WAV files are supported.")

    design_prompt = args.design_prompt if args.design_prompt else None
    use_audio_feature = args.use_audio_feature
    use_rag = args.use_rag

    postfix = f"{'dp' if design_prompt else ''}_{'af' if use_audio_feature else ''}_{'rag' if use_rag else ''}"

    annotate_and_show(args.input_file, args.design_prompt,
                      use_audio_feature=args.use_audio_feature,
                      use_rag=args.use_rag,
                      output_path=args.output_path,
                      postfix=postfix,
                      caption_only=args.caption_only)
