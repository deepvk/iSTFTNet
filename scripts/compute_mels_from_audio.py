import argparse
import glob
import os

import torch

from pathlib import Path

from tqdm import tqdm
from scipy.io.wavfile import read

from src.datasets.meldataset import MAX_WAV_VALUE, get_mel_spectrogram
from src.models.stft import TorchSTFT
from src.util.env import AttrDict
from src.util.utils import load_config


def main(args: argparse.Namespace):
    config = load_config(args.config_path)
    os.makedirs(args.output_mel_dirs, exist_ok=True)
    stft = TorchSTFT(**config)
    filelist = glob.glob(f"{args.input_wav_dirs}/*.wav")
    for filename in tqdm(filelist, total=len(filelist)):
        sr, wav = read(filename)
        wav = wav / MAX_WAV_VALUE
        wav = torch.tensor(wav, dtype=torch.float)
        x = get_mel_spectrogram(wav.unsqueeze(0), **config)
        mel_filename = Path(args.output_mel_dirs) / f"{Path(filename).stem}.pt"
        torch.save(x, str(mel_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", help="path to config/config.json")
    parser.add_argument("--input_wav_dirs", default="/app/data/deep_voices_wav")
    parser.add_argument("--output_mel_dirs", default="/app/data/deep_mels")
    args = parser.parse_args()
    main(args)
