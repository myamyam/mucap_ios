import argparse, json, re
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

    vectorstore = Chroma(persist_directory=persist_directory,
                         collection_name="mucap_rag_skeleton",
                         embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
    
    # Check if collection exists and has documents
    try:
        collection = vectorstore._collection
        num_docs = collection.count()
        if num_docs == 0:
            print("Warning: No documents found in the vector store.")
            return None
    except Exception as e:
        print(f"Warning: Error checking vector store: {e}")
        return None

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": threshold}
    )

    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query_text)
    else:
        docs = retriever.get_relevant_documents(query_text)
    if not docs:
        print(f"Warning: No documents found above similarity threshold {threshold}")
        return None

    output = ''
    for i, doc in enumerate(docs):
        output += f"[Annotation {i+1}] {doc.page_content}\n"

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
    attack = safe_float(music_features.get('attack', 0))
    envelope = safe_float(music_features.get('envelop', 0))
    onset_strength = safe_float(music_features.get('onset_strength', 0))
    resonance = safe_float(music_features.get('resonance', 0))

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
    attack_desc = "strong attack" if attack > 0.6 else "moderate attack" if attack > 0.3 else "soft attack"
    volume_desc = "loud" if volume > 0.6 else "medium volume" if volume > 0.3 else "quiet"
    resonance_desc = "long resonance" if resonance > 0.6 else "short resonance"

    return (
        f"3) [Context]:\n"
        f"- Instruments used: {instruments_str}\n"
        f"- Interpreted audio traits: {volume_desc}, {pitch_desc}, {attack_desc}, {resonance_desc}\n"
        f"- Raw values: volume: {volume:.2f}, pitch: {pitch:.2f}, attack: {attack:.2f}, "
        f"envelope: {envelope:.2f}, onset_strength: {onset_strength:.2f}, resonance: {resonance:.2f}"
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
        "- Use spaces between units, for example: "
        "[0.00] 쿠 [0.45] 탁 [0.80] 두둥 [1.25] 찌잉 [1.70] 팡\n"
        "- Use Hangul Korean onomatopoeia such as: "
        "찌잉, 땅, 뚜웅, 삽비빅, 따라라, 짜라, 쿠, 탁, 띠리링, "
        "으악, 크악, 챙, 슈욱, 퍽, 쿠웅.\n"
        "- For music, prefer syllables that follow pitch, sustain, attack, and rests: "
        "딩, 띵, 또로롱, 따란, 찌잉, 두둥, 쿵, 탁, 사락, 휘잉.\n"
        "- Do not write explanations, greetings, markdown, English words, romanized Korean, "
        "Japanese, Chinese characters, Hanja, adjectives, or extra symbols.\n"
        "- Do not say you will analyze the audio. Directly output the timestamped caption line.\n"
        "- Use as many units as needed to match audible rhythm and sound events.\n"
        "- If the context includes a Rapid onset hint, follow it with repeated close-timestamp Hangul units for clearly separated attacks.\n"
        "- If the context says the sound is a bowed string, sustained string, vibrato, tremble, or connected bowing texture, that string rule overrides the generic Rapid onset rule: do not caption every dense detected onset. Group the active span into flowing sustained Hangul phrases unless there is a clear new attack, rest, or melodic note. For longer stable sustain, express duration inside the Hangul token by extending vowels or flowing syllable runs, e.g. short: 지잉, 찌잉; medium: 지이잉, 찌이잉, 띠라라, 이이잉; long: 지이이잉, 찌이이이잉, 띠라라라라. Do not cycle examples repeatedly or place them at mechanically even intervals.\n"
        "- Avoid percussive one-syllable repeats like 딩 딩 딩 or 띵 띵 띵 for sustained strings; use them mainly for piano-like or clearly attacked notes.\n"
        "- If repeated notes are truly uniform, repeating the same Hangul syllable is allowed; "
        "when pitch, duration, loudness, attack, resonance, or rests change, reflect those "
        "audible changes with a small related set of Hangul variants such as 딩, 띵, 딩잉, 띠링, 따랑. "
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
        max_tokens=512, top_p=1, frequency_penalty=0, presence_penalty=0
    )

    return r.choices[0].message.content.strip()


# ─────────────────────── Overall execution flow ─────────────────────── #
def parse_caption(line: str):
    token_regex = re.compile(r"\[(\d+\.?\d*)\]\s*([^\s]+)")
    return [(float(t), tok) for t, tok in token_regex.findall(line)]


def _is_hangul_token(token: str) -> bool:
    return bool(re.fullmatch(r"[가-힣]+", token))


def caption_validation_error(caption: str, wav_path: str) -> str | None:
    if "..." in caption or "…" in caption:
        return "ellipsis/omission marker is not allowed"

    tokens = parse_caption(caption)
    if not tokens:
        return "no timestamped caption tokens"

    first_time = min(t for t, _ in tokens)
    last_time = max(t for t, _ in tokens)
    duration = audio_duration_seconds(wav_path)

    if first_time > 0.35:
        return "caption omits the beginning"
    if duration > 4.0 and last_time < duration * 0.75:
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

    return None


def append_caption_retry_instruction(messages, reason: str, wav_path: str) -> None:
    correction = (
        "\n\nCorrection required. The previous caption was invalid because: "
        f"{reason}. Regenerate from scratch. Do not summarize or use ellipses. "
        "Use only Hangul syllables inside caption tokens. Keep the caption natural "
        "and follow the audible rhythm from beginning to end."
    )
    for content in messages[0]["content"]:
        if content.get("type") == "text":
            content["text"] += correction
            return


def sanitize_caption(caption: str, wav_path: str) -> str:
    tokens = parse_caption(caption)
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


def generate_valid_caption(model, messages, wav_path: str, max_attempts: int = 3) -> str:
    caption = ""
    for attempt in range(max_attempts):
        caption = llm_api(model, messages, temp=0.5 if attempt else 0.7)
        reason = caption_validation_error(caption, wav_path)
        if reason is None:
            return caption
        append_caption_retry_instruction(messages, reason, wav_path)
    return sanitize_caption(caption, wav_path)


def caption_ends_too_early(caption: str, wav_path: str,
                           min_coverage: float = 0.75) -> bool:
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
                'resonance': audio_feature_df['resonance'] if 'resonance' in audio_feature_df else None,
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
                          'envelop': features['envelope'],
                          'onset_strength': features['onset_strength'],
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

    caption = generate_caption_with_end_retry(model, messages, wav_path)

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


    # visualize(mp3_path, output_path, tokens, font_dir='fonts', postfix=postfix, verbose=verbose)
    # render_video(mp3_path, tokens, output_path, hop=0.25, font_dir='fonts', fps=30, postfix=postfix, verbose=verbose)
    # render_plot_video(mp3_path, tokens, output_path, hop=0.25, font_dir='fonts', fps=30, postfix="postfix", verbose=verbose)

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

    base_name = splitext(basename(wav_path))[0]
    feature_json = join(
        dirname(output_path),
        f"{base_name}-{postfix}-features.json"
    )
    
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
