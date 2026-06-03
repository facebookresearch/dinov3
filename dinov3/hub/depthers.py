# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from enum import Enum
from typing import Optional, Tuple

import torch
from dinov3.eval.depth.models import DecoderConfig, make_depther_from_config

from .utils import _DINOV3_BASE_URL, _safe_load_state_dict_from_url
from .backbones import (
    Weights as BackboneWeights,
    dinov3_vitl16,
    dinov3_vit7b16,
    convert_path_or_url_to_url,
)


class DepthWeights(Enum):
    CHMV2 = "CHMV2"
    SYNTHMIX = "SYNTHMIX"


def _get_depth_range(dataset: DepthWeights):
    depth_ranges = {
        DepthWeights.SYNTHMIX: (0.001, 100.0),
    }
    return depth_ranges[dataset]


_DPT_HEAD_CONFIG_DICT = dict(
    use_backbone_norm=True,
    use_batchnorm=True,
    use_cls_token=False,
    n_output_channels=256,
    depth_weights=DepthWeights.SYNTHMIX,
    backbone_weights=BackboneWeights.LVD1689M,
)


def _get_out_layers(backbone_name):
    if "vitl" in backbone_name:
        return [4, 11, 17, 23]
    elif "vit7b" in backbone_name:
        return [9, 19, 29, 39]
    else:
        raise ValueError(f"Unrecognized backbone name {backbone_name}")


def _get_post_process_channels(backbone_name):
    if "vitl" in backbone_name:
        return [1024, 1024, 1024, 1024]
    elif "vit7b" in backbone_name:
        return [2048, 2048, 2048, 2048]


_BACKBONE_DICT = {
    "dinov3_vit7b16": dinov3_vit7b16,
    "dinov3_vitl16": dinov3_vitl16,
}


def _get_depther_config(
    backbone_name: str = "dinov3_vit7b16",
    depth_range: Optional[Tuple[float, float]] = None,
    **kwargs,
):
    out_index = _get_out_layers(backbone_name)
    post_process_channels = _get_post_process_channels(backbone_name)

    depth_range = depth_range or _get_depth_range(DepthWeights(_DPT_HEAD_CONFIG_DICT["depth_weights"]))
    min_depth, max_depth = depth_range
    depther_config = DecoderConfig(
        min_depth=min_depth,
        max_depth=max_depth,
        backbone_out_layers=out_index,
        n_output_channels=_DPT_HEAD_CONFIG_DICT["n_output_channels"],  # type: ignore
        use_backbone_norm=bool(_DPT_HEAD_CONFIG_DICT["use_backbone_norm"]),
        use_batchnorm=bool(_DPT_HEAD_CONFIG_DICT["use_batchnorm"]),
        use_cls_token=bool(_DPT_HEAD_CONFIG_DICT["use_cls_token"]),
        type="dpt",
        # DPTHead args
        head_kwargs=dict(
            channels=512,
            post_process_channels=post_process_channels,
        ),
        **kwargs,
    )
    return depther_config


def _make_dinov3_dpt_depther(
    *,
    backbone_name: str = "dinov3_vit7b16",
    pretrained: bool = True,
    depther_weights: DepthWeights | str = DepthWeights.SYNTHMIX,
    backbone_weights: BackboneWeights | str = BackboneWeights.LVD1689M,
    depth_range: Optional[Tuple[float, float]] = None,
    check_hash: bool = False,
    autocast_dtype: torch.dtype = torch.float32,
    **kwargs,
):
    backbone: torch.nn.Module = _BACKBONE_DICT[backbone_name](
        pretrained=pretrained,
        weights=backbone_weights,
    )

    depther = make_depther_from_config(
        backbone,
        config=_get_depther_config(backbone_name, depth_range),
        autocast_dtype=autocast_dtype,
    )

    if pretrained:
        if isinstance(depther_weights, DepthWeights):
            assert depther_weights == DepthWeights.SYNTHMIX, f"Unsupported depther weights {depther_weights}"
            weights_name = depther_weights.value.lower()
            hash = kwargs["hash"] if "hash" in kwargs else "02040be1"
            url = _DINOV3_BASE_URL + f"/{backbone_name}/{backbone_name}_{weights_name}_dpt_head-{hash}.pth"
        else:
            url = convert_path_or_url_to_url(depther_weights)
        checkpoint = _safe_load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        depther.decoder.load_state_dict(checkpoint, strict=True)
    return depther


def dinov3_vit7b16_dd(
    *,
    pretrained: bool = True,
    weights: DepthWeights | str = DepthWeights.SYNTHMIX,
    backbone_weights: BackboneWeights | str = BackboneWeights.LVD1689M,
    check_hash: bool = False,
    autocast_dtype: torch.dtype = torch.float32,
    **kwargs,
):
    return _make_dinov3_dpt_depther(
        backbone_name="dinov3_vit7b16",
        pretrained=pretrained,
        depther_weights=weights,
        backbone_weights=backbone_weights,
        check_hash=check_hash,
        autocast_dtype=autocast_dtype,
        **kwargs,
    )


def _get_chmv2_config(
    **kwargs,
):
    return DecoderConfig(
        min_depth=0.001,
        max_depth=96.0,
        backbone_out_layers=[5, 11, 17, 23],
        n_output_channels=256,
        use_backbone_norm=True,
        use_batchnorm=False,
        use_cls_token=True,
        type="dpt",
        bins_strategy="chmv2_mixlog",
        norm_strategy="chmv2_mixlog",
        head_kwargs={
            "n_hidden_channels": 128,
            "use_bias": True,
            "projection_after_fusion": False,
        },
        **kwargs,
    )


def dinov3_vitl16_chmv2(
    pretrained: bool = True,
    weights: DepthWeights | str = DepthWeights.CHMV2,
    backbone_weights: BackboneWeights | str = BackboneWeights.SAT493M,
    check_hash: bool = False,
    autocast_dtype: torch.dtype = torch.float32,
    **kwargs,
):
    backbone: torch.nn.Module = dinov3_vitl16(pretrained=pretrained, weights=backbone_weights)

    chmv2_depther = make_depther_from_config(
        backbone,
        config=_get_chmv2_config(),
        autocast_dtype=autocast_dtype,
    )

    if pretrained:
        if isinstance(weights, DepthWeights):
            assert weights == DepthWeights.CHMV2, f"Unsupported chmv2 weights {weights}"
            weights_name = weights.value.lower()
            url = _DINOV3_BASE_URL + f"/chmv2/dinov3_vitl16_{weights_name}_dpt_head-3703d643.pth"
        else:
            url = convert_path_or_url_to_url(weights)
        checkpoint = _safe_load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        chmv2_depther.decoder.load_state_dict(checkpoint, strict=True)
    return chmv2_depther
