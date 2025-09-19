# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import os, sys
import random
import subprocess
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


logger = logging.getLogger("dinov3")


def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int]], List[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int]], num_tokens: List[int]) -> List[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped


def named_replace(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        module = fn(module=module, name=name)
    for child_name_o, child_module in list(module.named_children()):
        child_name = ".".join((name, child_name_o)) if name else child_name_o
        new_child = named_replace(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
        setattr(module, child_name_o, new_child)

    if depth_first and include_root:
        module = fn(module=module, name=name)
    return module


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def fix_random_seeds(seed: int = 31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha() -> str:
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def get_conda_env() -> Tuple[Optional[str], Optional[str]]:
    conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")
    conda_env_path = os.environ.get("CONDA_PREFIX")
    return conda_env_name, conda_env_path


def count_parameters(module: nn.Module) -> int:
    c = 0
    for m in module.parameters():
        c += m.nelement()
    return c


def has_batchnorms(model: nn.Module) -> bool:
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for _, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def merge_checkpoints(dir):
    from accelerate.utils.fsdp_utils import merge_fsdp_weights
    merge_fsdp_weights(
        checkpoint_dir=dir,
        output_path=os.path.join(dir, "consolidated_model"),
        safe_serialization=False,
    )

if __name__ == '__main__':
    merge_checkpoints('/nethome/recpinfo/users/fibz/data/checkpoints/dinov3/ckpt/9999/')

