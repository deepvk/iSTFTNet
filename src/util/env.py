import os
import shutil

from pathlib import Path

from torch.distributed import init_process_group


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config_path: str, ckpt_path: str):
    """Copies config to the checkpoint directory"""
    config_name = Path(config_path).name
    target_path = Path(ckpt_path).joinpath(config_name)
    if config_path != target_path:
        os.makedirs(ckpt_path, exist_ok=True)
        shutil.copyfile(config_path, Path(ckpt_path).joinpath(config_name))


def configure_env_for_dist_training(config: AttrDict, rank: int):
    os.environ["MASTER_ADDR"] = config.dist_config["dist_addr"]
    os.environ["MASTER_PORT"] = config.dist_config["dist_port"]
    init_process_group(
        config.dist_config["dist_backend"], rank=rank, world_size=config.dist_config["world_size"] * config.num_gpus
    )
