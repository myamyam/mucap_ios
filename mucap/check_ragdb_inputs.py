import json
import os
from os.path import abspath, join

import pandas as pd


ROOT = os.path.dirname(__file__)
DATASET_DIR = join(ROOT, "dataset")
HUMAN_ANNOTATION_PATH = join(ROOT, "human_annotation", "human_annotation.json")
MUSIC_FEATURE_PATH = join(ROOT, "music_features.csv")


def main():
    with open(HUMAN_ANNOTATION_PATH, "r", encoding="utf-8") as file:
        human_annotations = json.load(file)

    music_feature_df = pd.read_csv(MUSIC_FEATURE_PATH, encoding="utf-8")
    feature_by_name = {
        row["file_name"]: row
        for _, row in music_feature_df.iterrows()
    }

    stats = {
        "human_annotation_count": len(human_annotations),
        "feature_rows": len(music_feature_df),
        "ready_docs": 0,
        "missing_audio": 0,
        "missing_feature": 0,
        "missing_description_json": 0,
        "bad_description_json": 0,
        "empty_human_annotation": 0,
    }
    examples = {key: [] for key in stats if key not in ("human_annotation_count", "feature_rows", "ready_docs")}

    for item in human_annotations:
        audio_url = item["data"]["audio"]
        audio_url_arr = audio_url.split("/")
        dataset, genre, filename = audio_url_arr[-3], audio_url_arr[-2], audio_url_arr[-1].replace("fixed_", "")

        audio_path = join(DATASET_DIR, dataset, genre, filename)
        json_path = join(DATASET_DIR, dataset, genre, f"{filename}.json")

        if not os.path.exists(audio_path):
            stats["missing_audio"] += 1
            examples["missing_audio"].append(filename)
            continue

        if filename not in feature_by_name:
            stats["missing_feature"] += 1
            examples["missing_feature"].append(filename)
            continue

        if not os.path.exists(json_path):
            stats["missing_description_json"] += 1
            examples["missing_description_json"].append(filename)
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as file:
                gpt_json = json.load(file)
            context = gpt_json["input"][0]["text"].split("[Context]")[1].split("volume(dB)")[0].strip()
            feeling = gpt_json["output"]["annotation"].strip()
            if not context or not feeling:
                raise ValueError("empty context or annotation")
        except Exception as exc:
            stats["bad_description_json"] += 1
            examples["bad_description_json"].append(f"{filename}: {exc}")
            continue

        text_arr = []
        for row in item["annotations"][0]["result"]:
            value = row.get("value", {})
            if "text" in value:
                text_arr.append(value["text"][0])

        if not " ".join(text_arr).strip():
            stats["empty_human_annotation"] += 1
            examples["empty_human_annotation"].append(filename)
            continue

        stats["ready_docs"] += 1

    print("RAG DB input check")
    for key, value in stats.items():
        print(f"{key}={value}")

    print("\nExamples")
    for key, values in examples.items():
        if values:
            print(f"{key}:")
            for value in values[:30]:
                print(f"  {value}")

    print(f"\nDATASET_DIR={abspath(DATASET_DIR)}")


if __name__ == "__main__":
    main()
