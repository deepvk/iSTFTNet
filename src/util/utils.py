import glob
import json
import os

import torch

from sys import stderr
from typing import Optional

from loguru import logger
from torch.nn.utils import weight_norm

from src.util.env import AttrDict


def setup_logger():
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(stderr, format=fmt)


def load_config(config_path: str) -> AttrDict:
    with open(config_path) as f:
        data = f.read()
    json_config = json.loads(data)
    config = AttrDict(json_config)
    return config


def init_weights(m: torch.nn.Module, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m: torch.nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath: str, device: str) -> Optional[dict]:
    logger.info(f"Loading {filepath}")
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    logger.info("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    logger.info(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    logger.info("Complete.")


def scan_checkpoint(cp_dir: str, prefix: str) -> Optional[str]:
    """
    Returns the latest checkpoint from the directory.
    """
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
