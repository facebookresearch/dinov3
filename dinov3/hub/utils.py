# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch


_DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"


def _safe_load_state_dict_from_url(url: str, **kwargs):
    # See https://github.com/pytorch/pytorch/releases/tag/v2.1.0 (Misc / #98479)
    if torch.__version__ >= (2, 1):
        local_kwargs = {**kwargs, "weights_only": True}
    else:
        local_kwargs = kwargs
    return torch.hub.load_state_dict_from_url(url, **local_kwargs)
