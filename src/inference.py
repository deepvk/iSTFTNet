import argparse
import glob
import os

import numpy as np
import onnxruntime
import torch

from pathlib import Path

from loguru import logger
from scipy.io.wavfile import write, read
from tqdm import tqdm

from src.datasets.meldataset import get_mel_spectrogram, MAX_WAV_VALUE
from src.models.modules import Generator
from src.models.stft import TorchSTFT
from src.util.env import AttrDict
from src.util.utils import setup_logger, load_checkpoint, load_config


@torch.no_grad()
def _inference(
    args: argparse.Namespace,
    config: AttrDict,
    device: str,
    mel_spectrograms: list[torch.Tensor],
    filenames: list[str],
    stft: TorchSTFT,
):
    generator = Generator(config).to(device)
    state_dict_g = load_checkpoint(args.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()
    specs, phases = [], []
    for i, mel in enumerate(mel_spectrograms):
        spec, phase = generator(mel)
        _synthesize_write_wav(stft, spec, phase, args.output_dir, filenames[i], config.sampling_rate)
    return specs, phases


@torch.no_grad()
def _onnx_inference(
    args: argparse.Namespace,
    config: AttrDict,
    device: str,
    mel_spectrograms: list[torch.Tensor],
    filenames: list[str],
    stft: TorchSTFT,
):
    specs, phases = [], []
    ort_session = onnxruntime.InferenceSession(args.checkpoint_file, providers=[args.onnx_provider])
    for i, mel in enumerate(mel_spectrograms):
        spec, phase = ort_session.run(None, {"input": mel.detach().cpu().numpy().astype(np.float32)})
        spec = torch.tensor(spec, dtype=torch.float).to(device)
        phase = torch.tensor(phase, dtype=torch.float).to(device)
        _synthesize_write_wav(stft, spec, phase, args.output_dir, filenames[i], config.sampling_rate)
    return specs, phases


def _synthesize_write_wav(stft, spec, phase, output_dir, filename, sr):
    y_g_hat = stft.inverse(spec, phase)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype("int16")
    output_file = Path(output_dir) / f"{Path(filename).stem}_generated.wav"
    write(output_file, sr, audio)
    logger.info(output_file)


def inference(args: argparse.Namespace, config: AttrDict, device: str):
    os.makedirs(args.output_dir, exist_ok=True)
    stft = TorchSTFT(**config).to(device)
    if args.compute_mels:
        filelist = glob.glob(f"{args.input_wavs_dir}/*.wav")
    else:
        filelist = glob.glob(f"{args.input_mels_dir}/*.pt")
    mel_spectrograms = []
    for filename in tqdm(filelist, desc="Generating audios..."):
        if args.compute_mels:
            sr, wav = read(filename)
            wav = wav / MAX_WAV_VALUE
            wav = torch.tensor(wav, dtype=torch.float).to(device)
            x = get_mel_spectrogram(wav.unsqueeze(0), **config)
        else:
            x = torch.load(filename).to(device)
        mel_spectrograms.append(x)
        if args.onnx_inference:
            specs, phases = _onnx_inference(args, config, device, mel_spectrograms, filelist, stft)
        else:
            specs, phases = _inference(args, config, device, mel_spectrograms, filelist, stft)


def main():
    setup_logger()
    logger.info("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", help="path to config/config.json")
    parser.add_argument("--checkpoint_file", default="/app/data/awesome_checkpoints/g_00975000")
    parser.add_argument("--onnx_inference", default=False, help="if True checkpoint file should be .onnx")
    parser.add_argument(
        "--onnx_provider", default="CPUExecutionProvider", help="https://onnxruntime.ai/docs/execution-providers/"
    )
    parser.add_argument("--input_wavs_dir", default="/app/data/deep_voices_wavs")
    parser.add_argument("--input_mels_dir", default="/app/data/deep_voices_mels")
    parser.add_argument("--output_dir", default="/app/data/generated_files")
    parser.add_argument(
        "--compute_mels",
        action=argparse.BooleanOptionalAction,
        help="Pass --no-compute_mels if --input_mels_dir is specified and mels are precomputed.",
    )
    args = parser.parse_args()

    if not args.input_wavs_dir and not args.input_mels_dir:
        logger.error("Mels directory or wav directory to get mels is required.")

    config = load_config(args.config_path)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(args, config, device)


if __name__ == "__main__":
    main()
