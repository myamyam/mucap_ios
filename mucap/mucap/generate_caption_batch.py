import argparse
import os
import traceback
from multiprocessing import cpu_count, Pool
from os.path import join, dirname, basename, abspath, splitext
from glob import glob

from dotenv import load_dotenv
from tqdm import tqdm
import re
from mucap.generate_caption import annotate_and_show
import random


def process_file(wav_path, input_dir, output_dir, design_prompt, use_audio_feature, use_rag, caption_only, model):
    relative_dir = dirname(wav_path)
    file_stem = splitext(basename(wav_path))[0]
    dir_path = join(output_dir, relative_dir, file_stem)
    os.makedirs(dir_path, exist_ok=True)

    postfix = f"{'dp' if design_prompt else ''}_{'af' if use_audio_feature else ''}_{'rag' if use_rag else ''}"

    wav_path_abs = join(input_dir, wav_path)

    try:
        annotate_and_show(
            wav_path_abs,
            use_design_prompt=design_prompt,
            use_audio_feature=use_audio_feature,
            use_rag=use_rag,
            output_path=dir_path,
            postfix=postfix,
            verbose=False,
            caption_only=caption_only,
            model=model,
        )
        return None  # success
    except Exception as e:
        traceback.print_exc()
        return f"Warning: Error processing {wav_path}: {e}"


def batch_process(input_dir, output_dir, design_prompt,
                  use_audio_feature=True, use_rag=True,
                  shuffle=False, n=-1, num_workers=None,
                  caption_only=False, model="gpt-audio-1.5"):
    # get with glob
    input_dir = abspath(input_dir)
    output_dir = abspath(output_dir)

    wav_regex = join(input_dir, "**", "*.wav")
    wav_files = glob(wav_regex, recursive=True)

    wav_files = sorted(
        f.replace(input_dir, '')[1:]
        for f in wav_files
    )



    if shuffle:
        random.shuffle(wav_files)
    if n > 0:
        wav_files = wav_files[:n]

    args_list = [
        (wav_path, input_dir, output_dir, design_prompt, use_audio_feature, use_rag, caption_only, model)
        for wav_path in wav_files
    ]

    print(f"Processing {len(args_list)} files with {num_workers or cpu_count()} workers...")

    with Pool(processes=num_workers or cpu_count()) as pool:
        for result in tqdm(pool.starmap(process_file, args_list), total=len(args_list), desc="Processing files"):
            if result:
                print(result)


if __name__ == "__main__":
    load_dotenv()
    random.seed(42)

    parser = argparse.ArgumentParser()

    default_input_dir = join(dirname(__file__), '..', "input")
    default_output_dir = join(dirname(__file__), '..', "output")

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser.add_argument("--input_dir", type=str, default=default_input_dir)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)

    parser.add_argument("--use_design_prompt", type=str2bool, default=True, help="Whether to use the design prompt")
    parser.add_argument("--use_audio_feature", type=str2bool, default=True, help="Whether to use audio features")
    parser.add_argument("--use_rag", type=str2bool, default=True, help="Whether to use RAG")
    parser.add_argument("--model", type=str, default="gpt-audio-1.5", help="OpenAI audio-capable chat model")
    parser.add_argument(
        "--caption_only",
        type=str2bool,
        default=False,
        help="Only write caption JSON/TXT and skip MuCap visualization MP4/PNG rendering.",
    )

    parser.add_argument('--shuffle', action='store_true', help="Shuffle input files randomly.")
    parser.add_argument('--n', type=int, default=-1, help="Maximum number of files to process (-1: all)")
    parser.add_argument('--num_workers', type=int, default=None, help="Number of processes to use (default: CPU count)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        design_prompt=args.use_design_prompt,
        use_audio_feature=args.use_audio_feature,
        use_rag=args.use_rag,
        shuffle=args.shuffle,
        n=args.n,
        num_workers=args.num_workers,
        caption_only=args.caption_only,
        model=args.model

    )
