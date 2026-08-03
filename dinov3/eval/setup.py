# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dataclasses import dataclass
from typing import Tuple, TypedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.models import build_model_for_eval


@dataclass
class ModelConfig:
    # Loading a local file
    config_file: str | None = None
    pretrained_weights: str | None = None
    # Loading a DINOv3 or v2 model from torch.hub
    dino_hub: str | None = None
    # Target device for eval; None selects CUDA when available, else CPU
    device: str | None = None


class BaseModelContext(TypedDict):
    """
    An object that contains the context of a model (autocast, description, ...)
    """

    autocast_dtype: torch.dtype  # default could be torch.float
    device: torch.device


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_autocast_device_type(device: str | torch.device | None = None) -> str:
    """Return a torch.autocast-compatible device_type string."""
    resolved = resolve_device(device)
    return "cuda" if resolved.type == "cuda" else "cpu"


def load_model_and_context(model_config: ModelConfig, output_dir: str) -> tuple[torch.nn.Module, BaseModelContext]:
    if model_config.dino_hub is not None:
        assert model_config.pretrained_weights is None and model_config.config_file is None
        if "dinov3" in model_config.dino_hub:
            repo = "dinov3"
        elif "dinov2" in model_config.dino_hub:
            repo = "dinov2"
        else:
            raise ValueError
        model = torch.hub.load(f"facebookresearch/{repo}", model_config.dino_hub)
        base_model_context = BaseModelContext(autocast_dtype=torch.float, device=resolve_device(model_config.device))
    else:
        model, base_model_context = setup_and_build_model(
            config_file=model_config.config_file,
            pretrained_weights=model_config.pretrained_weights,
            output_dir=output_dir,
            device=model_config.device,
        )

    device = base_model_context["device"]
    model.to(device)
    model.eval()
    return model, base_model_context


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.param_dtype
    if teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def setup_and_build_model(
    config_file: str,
    pretrained_weights: str | None = None,
    shard_unsharded_model: bool = False,
    output_dir: str = "",
    opts: list | None = None,
    device: str | torch.device | None = None,
    **ignored_kwargs,
) -> Tuple[nn.Module, BaseModelContext]:
    cudnn.benchmark = True
    del ignored_kwargs
    setup_args = DinoV3SetupArgs(
        config_file=config_file,
        pretrained_weights=pretrained_weights,
        shard_unsharded_model=shard_unsharded_model,
        output_dir=output_dir,
        opts=opts or [],
    )
    config = setup_config(setup_args, strict_cfg=False)
    model = build_model_for_eval(config, setup_args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, BaseModelContext(autocast_dtype=autocast_dtype, device=resolve_device(device))
