import argparse
import base64
import json
import os
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from tqdm import tqdm

from mucap.audio import FFMPEG


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
HUMAN_ANNOTATION_PATH = ROOT / "human_annotation" / "human_annotation.json"
MUSIC_FEATURE_PATH = ROOT / "music_features.csv"


def posix_rel(path):
    return path.relative_to(ROOT).as_posix()


def parse_annotation_target(item):
    audio_url = item["data"]["audio"]
    parts = audio_url.split("/")
    dataset, genre, filename = parts[-3], parts[-2], parts[-1].replace("fixed_", "")
    audio_path = DATASET_DIR / dataset / genre / filename
    json_path = audio_path.with_name(audio_path.name + ".json")
    return {
        "dataset": dataset,
        "genre": genre,
        "file_name": filename,
        "audio_url": audio_url,
        "audio_path": audio_path,
        "json_path": json_path,
    }


def compact_feature_text(row):
    fields = {
        "dataset": row.get("dataset"),
        "genre": row.get("genre"),
        "sample_rate": row.get("sample_rate"),
        "length": row.get("length"),
        "pitch": row.get("pitch"),
        "tempo": row.get("tempo"),
        "attack": row.get("attack"),
        "envelope": row.get("envelope"),
        "onset_strength": row.get("onset_strength"),
        "resonance": row.get("resonance"),
        "notes": row.get("notes"),
        "beat_frames": row.get("beat_frames"),
        "top5_instruments": row.get("top5_instruments"),
    }
    lines = [f"{key}: {value}" for key, value in fields.items() if pd.notna(value)]

    # json2ragdb.ipynb extracts the text between "[Context]" and "volume(dB)".
    # Keep this marker so the generated JSON is compatible with the notebook.
    volume = row.get("volume")
    lines.append(f"volume(dB): {volume}")
    return "\n".join(lines)


def build_prompt(target, feature_row):
    feature_text = compact_feature_text(feature_row)
    return (
        "You are generating a perceptual description for a 14-second instrumental "
        "music segment. This is not the final caption. Describe what is heard so "
        "it can later be used as retrieval context for sound-mimetic captioning.\n\n"
        "[Context]\n"
        f"file_name: {target['file_name']}\n"
        f"dataset: {target['dataset']}\n"
        f"genre: {target['genre']}\n"
        f"{feature_text}\n\n"
        "[Task]\n"
        "Listen to the audio and write a concise auditory description. Mention the "
        "dominant instruments or timbre, rhythm, density, attack, pitch contour, "
        "energy, and any notable transitions. Do not create timestamped captions. "
        "Do not invent lyrics. Return only one plain English description paragraph. "
        "Do not return JSON, markdown, bullet points, labels, keys, or file metadata."
    )


def wav_to_temp_mp3(wav_path):
    if not FFMPEG:
        raise RuntimeError("ffmpeg was not found. Open a new terminal or add ffmpeg.exe to PATH.")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    AudioSegment.from_wav(wav_path).export(tmp.name, format="mp3", bitrate="192k")
    return tmp.name


def encode_file(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def call_openai(client, model, prompt, wav_path, temperature):
    mp3_path = wav_to_temp_mp3(wav_path)
    try:
        audio_b64 = encode_file(mp3_path)
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=350,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "mp3"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()
    finally:
        try:
            os.remove(mp3_path)
        except OSError:
            pass


def write_description_json(target, prompt, annotation, model):
    payload = {
        "input": [
            {
                "type": "text",
                "text": prompt,
            }
        ],
        "output": {
            "audio": posix_rel(target["audio_path"]),
            "annotation": annotation,
        },
        "metadata": {
            "source_audio_url": target["audio_url"],
            "dataset": target["dataset"],
            "genre": target["genre"],
            "file_name": target["file_name"],
            "model": model,
        },
    }

    target["json_path"].parent.mkdir(parents=True, exist_ok=True)
    with open(target["json_path"], "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    failed_path = target["json_path"].with_name(target["json_path"].name + ".failed.txt")
    if failed_path.exists():
        failed_path.unlink()


def build_targets(limit=None, overwrite=False):
    with open(HUMAN_ANNOTATION_PATH, "r", encoding="utf-8") as file:
        human_annotations = json.load(file)

    feature_df = pd.read_csv(MUSIC_FEATURE_PATH, encoding="utf-8")
    feature_by_name = {
        row["file_name"]: row
        for _, row in feature_df.iterrows()
    }

    targets = []
    skipped = {
        "missing_audio": 0,
        "missing_feature": 0,
        "existing_json": 0,
    }

    seen = set()
    for item in human_annotations:
        target = parse_annotation_target(item)
        key = (target["dataset"], target["genre"], target["file_name"])
        if key in seen:
            continue
        seen.add(key)

        if not target["audio_path"].exists():
            skipped["missing_audio"] += 1
            continue
        if target["file_name"] not in feature_by_name:
            skipped["missing_feature"] += 1
            continue
        if target["json_path"].exists() and not overwrite:
            skipped["existing_json"] += 1
            continue

        target["feature_row"] = feature_by_name[target["file_name"]]
        targets.append(target)
        if limit and len(targets) >= limit:
            break

    return targets, skipped, len(human_annotations)


def process_one(target, model, temperature, retry_count, retry_sleep):
    client = OpenAI()
    prompt = build_prompt(target, target["feature_row"])
    last_error = None

    for attempt in range(retry_count + 1):
        try:
            annotation = call_openai(client, model, prompt, target["audio_path"], temperature)
            write_description_json(target, prompt, annotation, model)
            return {"ok": True, "file_name": target["file_name"], "json_path": str(target["json_path"])}
        except Exception as exc:
            last_error = exc
            if attempt < retry_count:
                time.sleep(retry_sleep * (attempt + 1))

    failed_path = target["json_path"].with_name(target["json_path"].name + ".failed.txt")
    with open(failed_path, "w", encoding="utf-8") as file:
        file.write("".join(traceback.format_exception(last_error)))
    return {"ok": False, "file_name": target["file_name"], "error": str(last_error)}


def main():
    parser = argparse.ArgumentParser(
        description="Generate MuCap description JSON files using human_annotation.json as the source of truth."
    )
    parser.add_argument("--model", default="gpt-4o-audio-preview")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    targets, skipped, annotation_count = build_targets(limit=args.limit, overwrite=args.overwrite)

    print(f"human_annotation_count={annotation_count}")
    print(f"targets_to_generate={len(targets)}")
    print(f"skipped={skipped}")
    if targets:
        print(f"first_audio={targets[0]['audio_path']}")
        print(f"first_json={targets[0]['json_path']}")

    if args.dry_run:
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Create .env or set the environment variable first.")

    ok_count = 0
    fail_count = 0
    workers = max(1, args.workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_one,
                target,
                args.model,
                args.temperature,
                args.retry_count,
                args.retry_sleep,
            )
            for target in targets
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result["ok"]:
                ok_count += 1
            else:
                fail_count += 1
                print(f"FAILED {result['file_name']}: {result['error']}")

    print(f"done ok={ok_count} failed={fail_count}")


if __name__ == "__main__":
    main()
