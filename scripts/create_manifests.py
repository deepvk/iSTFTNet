import glob
import argparse
import numpy as np

from pathlib import Path


def run(args: argparse.Namespace):
    filenames = glob.glob(f"{args.source_wavs_dir}/*.wav")
    np.random.shuffle(filenames)
    for i, file in enumerate(filenames):
        output_filename = "val.txt" if i < args.val_set_size else "train.txt"
        with open(Path(args.target_manifests_dir).joinpath(output_filename), "a") as f:
            f.write(str(Path(args.source_wavs_dir).joinpath(file)))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_wavs_dir", type=str)
    parser.add_argument("--target_manifests_dir", type=str)
    parser.add_argument("--val_set_size", type=int, default=256)
    args = parser.parse_args()
    run(args)
