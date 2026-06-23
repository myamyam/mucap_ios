import argparse
import json
import os
import shutil
from os.path import abspath, dirname, join

import pandas as pd
from dotenv import load_dotenv


def build_documents(root_dir, require_nonempty_description=False):
    from langchain_core.documents import Document

    dataset_dir = abspath(join(root_dir, "dataset"))
    human_annotation_path = join(root_dir, "human_annotation", "human_annotation.json")
    music_feature_path = join(root_dir, "music_features.csv")

    with open(human_annotation_path, "r", encoding="utf-8") as file:
        human_annotations = json.load(file)

    music_feature_df = pd.read_csv(music_feature_path, encoding="utf-8")
    feature_names = set(music_feature_df["file_name"].tolist())

    docs = []
    stats = {
        "human_annotation_count": len(human_annotations),
        "ready_docs": 0,
        "missing_description_json": 0,
        "missing_feature": 0,
        "bad_description_json": 0,
        "empty_description": 0,
    }

    for item in human_annotations:
        audio_url = item["data"]["audio"]
        audio_url_arr = audio_url.split("/")
        try:
            dataset, genre, filename = audio_url_arr[-3], audio_url_arr[-2], audio_url_arr[-1].replace("fixed_", "")
        except Exception:
            continue

        json_path = join(dataset_dir, dataset, genre, f"{filename}.json")
        if not os.path.exists(json_path):
            stats["missing_description_json"] += 1
            continue

        if filename not in feature_names:
            stats["missing_feature"] += 1
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as file:
                gpt_json = json.load(file)

            music_feature_txt = gpt_json["input"][0]["text"]
            music_feature_txt = music_feature_txt.split("[Context]")[1].split("volume(dB)")[0].strip()
            music_feeling_txt = gpt_json["output"]["annotation"].strip()
        except Exception:
            stats["bad_description_json"] += 1
            continue

        if not music_feeling_txt:
            stats["empty_description"] += 1
            if require_nonempty_description:
                continue

        text_arr = []
        for row in item["annotations"][0]["result"]:
            value = row.get("value", {})
            if "text" in value:
                text_arr.append(value["text"][0])

        annotation = " ".join(text_arr)
        row_txt = (
            f"Music Feature: {music_feature_txt}\n"
            f"Music Feeling: {music_feeling_txt}\n"
            f"Annotation: {annotation}"
        )

        docs.append(
            Document(
                page_content=row_txt,
                metadata={
                    "source_path": audio_url,
                    "source_type": "audio_caption",
                    "source": filename,
                    "dataset": dataset,
                    "genre": genre,
                },
            )
        )
        stats["ready_docs"] += 1

    return docs, stats


def main():
    parser = argparse.ArgumentParser(description="Build MuCap Chroma RAG DB from generated description JSON files.")
    parser.add_argument("--persist-directory", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-nonempty-description", action="store_true")
    args = parser.parse_args()

    rag_dir = dirname(__file__)
    root_dir = abspath(join(rag_dir, "..", ".."))
    load_dotenv(join(root_dir, ".env"))

    docs, stats = build_documents(
        root_dir=root_dir,
        require_nonempty_description=args.require_nonempty_description,
    )

    for key, value in stats.items():
        print(f"{key}={value}")

    if args.dry_run:
        print("dry_run=True")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Create .env or set the environment variable first.")

    try:
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma
    except ImportError as exc:
        raise RuntimeError(
            "RAG DB dependencies are missing. Install langchain-community, chromadb, openai, and tiktoken."
        ) from exc

    persist_directory = args.persist_directory or join(rag_dir, "ragdb")
    if args.overwrite and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    if not docs:
        raise RuntimeError("No documents are available for RAG DB generation.")

    vectordb = Chroma.from_documents(
        docs[: args.batch_size],
        embeddings,
        persist_directory=persist_directory,
    )
    for start in range(args.batch_size, len(docs), args.batch_size):
        end = min(start + args.batch_size, len(docs))
        print(f"adding_docs={start}-{end} / {len(docs)}")
        vectordb.add_documents(docs[start:end])

    try:
        print(f"ragdb_count={vectordb._collection.count()}")
    except Exception:
        pass
    print(f"ragdb_written={persist_directory}")


if __name__ == "__main__":
    main()
