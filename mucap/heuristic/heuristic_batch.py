import argparse
import os
import traceback
from multiprocessing import cpu_count, Pool
from os.path import join, dirname, basename, abspath
from glob import glob

from dotenv import load_dotenv
from tqdm import tqdm
import re

from heuristic.heuristic_annotation import heuristic_annotation
import random


def process_file(wav_path, input_dir, output_dir):
    file = wav_path.replace(input_dir, '')
    dir_path = join(output_dir, dirname(file))
    os.makedirs(dir_path, exist_ok=True)

    postfix = f"-hr"

    wav_path_abs = join(input_dir, wav_path)
    output_path = join(dir_path, basename(file).replace('.wav', f'{postfix}.txt'))

    try:
        annotation = heuristic_annotation(wav_path_abs)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(annotation)

    except:
        traceback.print_exc()
        return f"⚠️ Error processing {wav_path}: {traceback.format_exc()}"

    return annotation


def batch_process(input_dir, output_dir, shuffle=False, n=-1, num_workers=None):
    # get with glob
    input_dir = abspath(input_dir)
    output_dir = abspath(output_dir)

    wav_regex = join(input_dir, "**", "*.wav")
    wav_files = glob(wav_regex, recursive=True)



    if not wav_files:
        print(f"❌ No .wav files found in {input_dir}. Please check the path.")
        return
    print(f"Found {len(wav_files)} .wav files in {input_dir}")

    # 2. Regex filtering: select only novocal_<number>.wav files
    # pattern = re.compile(r"novocal_\d+\.wav$")
    # filtered_wavs = [f for f in wav_files if pattern.search(basename(f))]
    # if not filtered_wavs:
    #     print("❌ No novocal_*.wav files found in the specified folder.")
    #     return

    # filtered_wavs = [abspath(f) for f in filtered_wavs]
    wav_files = [f.replace(input_dir, '')[1:] for f in wav_files]



    if shuffle:
        random.shuffle(wav_files)
    if n > 0:
        wav_files = wav_files[:n]

    args_list = [
        (wav_path, input_dir, output_dir)
        for wav_path in wav_files
    ]

    print(f"✅ Processing {len(args_list)} files with {num_workers or cpu_count() // 2} workers...")

    with Pool(processes=num_workers or cpu_count()) as pool:
        for result in tqdm(pool.starmap(process_file, args_list), total=len(args_list), desc="Processing files"):
            if result:
                print(result)


if __name__ == "__main__":
    load_dotenv()
    random.seed(42)

    parser = argparse.ArgumentParser()

    default_input_dir = join(dirname(__file__), '..', "dataset")
    default_output_dir = join(dirname(__file__), '..', "output")

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser.add_argument("--input_dir", type=str, default=default_input_dir)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)

    parser.add_argument('--shuffle', action='store_true', help="Shuffle input files randomly.")
    parser.add_argument('--n', type=int, default=-1, help="Maximum number of files to process (-1: all)")
    parser.add_argument('--num_workers', type=int, default=None, help="Number of processes to use (default: CPU count)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        shuffle=args.shuffle,
        n=args.n,
        num_workers = args.num_workers

    )
