import argparse, json, os, re, wave
import pandas as pd
from os.path import join, basename, splitext, dirname
from dotenv import load_dotenv
from openai import OpenAI
from mucap.audio import encode_audio, wav_to_mp3
from mucap.tokends2caption import tokens_to_caption


def retrieve_rag_context(query_text: str, top_k=2, threshold=0.8):
    """Retrieve relevant context using Chroma + OpenAI embeddings with similarity threshold."""
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError(
            "RAG is enabled, but langchain-openai/langchain-chroma are not installed. "
            "Run with --use_rag false for caption-only generation without RAG."
        ) from exc

    persist_directory = join(dirname(__file__), 'rag', 'ragdb')

    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
    
    # Check if collection exists and has documents
    try:
        collection = vectorstore._collection
        num_docs = collection.count()
        if num_docs == 0:
            print("⚠ No documents found in the vector store.")
            return None
    except Exception as e:
        print(f"⚠ Error checking vector store: {e}")
        return None

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": threshold}
    )

    docs = retriever.get_relevant_documents(query_text)
    if not docs:
        print(f"⚠ No documents found above similarity threshold {threshold}")

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

def audio_duration_seconds(wav_path):
    try:
        with wave.open(wav_path, "rb") as audio:
            frames = audio.getnframes()
            sample_rate = audio.getframerate()
            return frames / float(sample_rate)
    except (wave.Error, OSError):
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "Could not read WAV duration with wave, and soundfile is not installed."
            ) from exc
        info = sf.info(wav_path)
        return info.frames / float(info.samplerate)


def prompt_end_second(duration_sec):
    if duration_sec <= 0:
        raise ValueError(f"Audio duration must be positive, got {duration_sec}")
    return max(0.0, duration_sec - min(0.2, duration_sec * 0.1))


def build_prompt_rhythm(audio_b64: str, design_prompt=None, music_features=None,
                        use_rag=None, duration_sec=14.0):

    if music_features is not None:
        feature_str = interpret_music_features(music_features)
    else:
        feature_str = "3) [Context]: - No music features available; rely only on the audio."


    if use_rag:
        # print("🔍 Retrieving extra context from vector DB...")
        rag_context = retrieve_rag_context(feature_str, threshold=0.1)  # Lowered threshold to 0.1

        if rag_context is not None:
            feature_str += (
                "\n4) [Onomatopoeia annotations of similar audio] "
                "(use only as reference and adapt to the provided music): "
                f"{rag_context}\n"
            )

    duration_sec = float(duration_sec)
    end_sec = prompt_end_second(duration_sec)

    text = (
        "1) [Role]: You are a subtitle creator who listens to music and writes Korean vocalizations that closely mimic the sound. Onomatopoeia is allowed, but the goal is to imitate the actual audio with mouth-made syllables.\n"
        f"2) [Objective]: Listen to the {duration_sec:.1f}-second clip and transcribe it with Korean syllabic scat (e.g., jjiing, ttang, ttu-woong, ppi-bi-bi, ttarra, jjara) that best matches the real texture and timing. "
        "Consider rhythm, intensity, tone, and duration, and write as if placing captions on a video so the heard sounds feel natural and vivid. "
        "Onomatopoeia is allowed, but choose syllables that most closely imitate the actual sound rather than abstract repetition.\n"
        f"{feature_str}\n"
        "3) [Format]\n"
        "   - Each unit should follow `[sec] syllable` (e.g., `[0.00] dun-dun`), separated by spaces\n"
        "   - Prohibit numbers, English words, adjectives, Japanese, symbols, or newlines; output only one line\n"
        f"   - Write time-aligned Korean syllabic captions only between 0.00 and {end_sec:.1f} seconds.\n"
        "   - Use as many caption units as needed to match the audible rhythm and sound events.\n"
        "   - Do not force a fixed number of units.\n"
        f"   - If the sound continues throughout the clip, distribute captions across the full duration from the beginning to near {end_sec:.1f}s.\n"
        "   - Do not stop early while audible sound continues.\n"
        "   - Avoid timestamps outside the audio duration.\n"
        "   - Choose consonant and vowel combinations that connect smoothly with preceding words \n"
        "   - Mimic the actual rhythm and sound as closely as possible\n"
        f"4) [Guide] {design_prompt or 'No guide'}"
        "5) [Example] [0.00] dung [0.45] tang [0.80] du-doong [1.25] jjiing [1.70] pang"
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
        max_tokens=256, top_p=1, frequency_penalty=0, presence_penalty=0
    )

    return r.choices[0].message.content.strip()


# ─────────────────────── Overall execution flow ─────────────────────── #
def parse_caption(line: str):
    token_regex = re.compile(r"\[(\d+\.?\d*)\]\s*([^\s]+)")
    return [(float(t), tok) for t, tok in token_regex.findall(line)]


def annotate_and_show(wav_path, use_design_prompt=True,
                      model="gpt-audio",
                      use_audio_feature=True,
                      use_rag=True,
                      output_path=None,
                      postfix=None,
                      verbose=False,
                      caption_only=False):

    duration_sec = audio_duration_seconds(wav_path)
    mp3_path = wav_to_mp3(wav_path)
    audio_b64 = encode_audio(mp3_path)

    if use_design_prompt:
        design_guide_path = join(dirname(__file__), 'prompt', 'design_guide.txt')
        if os.path.exists(design_guide_path):
            with open(design_guide_path, "r", encoding="utf-8") as f:
                design_txt = f.read().strip()
        else:
            print(f"Warning: design guide not found; continuing without it: {design_guide_path}")
            design_txt = None
    else:
        design_txt = None

    if use_audio_feature:
        # lo
        feature_path = join(dirname(__file__), '..', 'music_features', 'music_features.csv')
        audio_feature_df = pd.read_csv(feature_path)
        wav_filename = basename(wav_path)

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
            try:
                from music_features.feature_extractor import AudioFeatureExtraction
            except ImportError as exc:
                raise RuntimeError(
                    "Audio feature extraction is enabled, but its optional dependencies "
                    "(tensorflow/essentia) are not installed. Run with --use_audio_feature false "
                    "for caption-only generation without feature extraction."
                ) from exc
            audio_extractor = AudioFeatureExtraction()

            features = audio_extractor.extract_features(wav_path)
            instrument_predictions = audio_extractor.predict_instrument_essentia(wav_path)


        music_features = {'volume': features['volume'],
                          'pitch': features['pitch'],
                          'attack': features['attack'],
                          'envelop': features['envelope'],
                          'onset_strength': features['onset_strength'],
                          'resonance': features['resonance'],
                          'top5_instruments': instrument_predictions}
    else:
        music_features = None

    messages = build_prompt_rhythm(
        audio_b64,
        design_txt,
        music_features,
        use_rag,
        duration_sec=duration_sec,
    )
    
    # text_only = next(
    # (item['text'] for item in messages[0]['content'] if item['type'] == 'text'), None
    # )
    # print("🔎 Prompt length:", len(text_only.split()) if text_only else "N/A")
    # print("🔍 Prompt preview:\n", text_only[:500], "...")  # Preview only the beginning

    caption = llm_api(model, messages)

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
        return out_json

    from extract_spectogram import save_spectrogram
    from mucap.render.plot_render import render_plot_video
    from mucap.render.circle_render import render_plot_video as render_circle

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
    annotation_path = join(dirname(__file__), "rag", "mucap_annotation_fin.json")
    
    # Extract filename from wav_path to match with annotation data
    wav_filename = basename(wav_path)
    
    # Find matching human annotation
    human_annotation_text = None
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
            
        for item in annotation_data:
            audio_url = item['data']['audio']
            # Extract filename from URL
            url_filename = audio_url.split('/')[-1]
            
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

    save_spectrogram(mp3_path, output_path, sr=44100, postfix=postfix)



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
