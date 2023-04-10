import argparse
import warnings

import torch

from src.models.modules import Generator
from src.util.env import AttrDict
from src.util.utils import load_checkpoint, load_config

warnings.filterwarnings(action="ignore", category=UserWarning)


def convert(args: argparse.Namespace, config: AttrDict, device: str):
    generator = Generator(config).to(device)
    state_dict_g = load_checkpoint(args.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()

    mel_channels = config.num_mels
    # this params are defined for dummy input, will be dynamic for future inference
    batch_size, mel_len = 1, 148
    dummy_input = torch.rand((batch_size, mel_channels, mel_len)).to(device)

    onnx_path = args.converted_model_path
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {"input": {0: "batch_size", 2: "mel_len"}, "output": {0: "batch_size"}}
    generator(dummy_input)

    torch.onnx.export(
        generator, dummy_input, onnx_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--config_path", default="/app/config/config.json")
    parser.add_argument("--converted_model_path", default="/app/istft_vocoder.onnx")
    args = parser.parse_args()
    config = load_config(args.config_path)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    convert(args, config, device)
