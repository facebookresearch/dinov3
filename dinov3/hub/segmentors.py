# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum

import torch
from dinov3.eval.segmentation.models import build_segmentation_decoder

from .backbones import (
    dinov3_vit7b16,
    dinov3_vitl16,
    Weights as BackboneWeights,
    convert_path_or_url_to_url,
)
from .utils import _DINOV3_BASE_URL, _safe_load_state_dict_from_url


class SegmentorWeights(Enum):
    ADE20K = "ADE20K"


def _make_dinov3_m2f_segmentor(
    *,
    backbone_name: str = "dinov3_vit7b16",
    pretrained: bool = True,
    segmentor_weights: SegmentorWeights | str = SegmentorWeights.ADE20K,
    backbone_weights: BackboneWeights | str = BackboneWeights.LVD1689M,
    check_hash: bool = False,
    autocast_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    if backbone_name == "dinov3_vit7b16":
        backbone_model = dinov3_vit7b16(pretrained=pretrained, weights=backbone_weights, check_hash=check_hash)
    elif backbone_name == "dinov3_vitl16":
        backbone_model = dinov3_vitl16(pretrained=pretrained, weights=backbone_weights, check_hash=check_hash)
    else:
        raise AssertionError(f"No pretrained segmentation checkpoint available for {backbone_name}")

    hidden_dim = 2048 if "hidden_dim" not in kwargs else kwargs["hidden_dim"]
    segmentor = build_segmentation_decoder(
        backbone_model=backbone_model,
        decoder_type="m2f",
        hidden_dim=hidden_dim,
        autocast_dtype=autocast_dtype,
    )
    if pretrained:
        if type(segmentor_weights) is SegmentorWeights:
            assert segmentor_weights == SegmentorWeights.ADE20K, (
                f"Unsupported weights for segmentor: {segmentor_weights}"
            )
            segmentor_weights_name = segmentor_weights.value.lower()
            hash = kwargs["hash"] if "hash" in kwargs else "bf307cb1"
            model_filename = f"{backbone_name}_{segmentor_weights_name}_m2f_head-{hash}.pth"
            url = os.path.join(_DINOV3_BASE_URL, backbone_name, model_filename)
        else:
            url = convert_path_or_url_to_url(segmentor_weights)
        state_dict = _safe_load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        missing_keys, unexpected_keys = segmentor.load_state_dict(state_dict, strict=False)
        assert len([k for k in missing_keys if "backbone" not in k]) == 0
        assert len(unexpected_keys) == 0

    return segmentor


def dinov3_vit7b16_ms(
    *,
    pretrained: bool = True,
    weights: SegmentorWeights | str = SegmentorWeights.ADE20K,
    backbone_weights: BackboneWeights | str = BackboneWeights.LVD1689M,
    check_hash: bool = False,
    autocast_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):
    return _make_dinov3_m2f_segmentor(
        backbone_name="dinov3_vit7b16",
        pretrained=pretrained,
        segmentor_weights=weights,
        backbone_weights=backbone_weights,
        check_hash=check_hash,
        autocast_dtype=autocast_dtype,
        **kwargs,
    )
