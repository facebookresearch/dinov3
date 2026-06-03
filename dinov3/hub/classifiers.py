# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum

import torch
import torch.nn as nn

from .backbones import (
    dinov3_vit7b16,
    Weights as BackboneWeights,
    convert_path_or_url_to_url,
)

from .utils import _DINOV3_BASE_URL, _safe_load_state_dict_from_url


class ClassifierWeights(Enum):
    IMAGENET1K = "IMAGENET1K"


def _make_dinov3_linear_classification_head(
    *,
    backbone_name: str = "dinov3_vit7b16",
    embed_dim: int = 8192,
    pretrained: bool = True,
    classifier_weights: ClassifierWeights | str = ClassifierWeights.IMAGENET1K,
    check_hash: bool = False,
    **kwargs,
):
    linear_head = nn.Linear(embed_dim, 1_000)
    if pretrained:
        if type(classifier_weights) is ClassifierWeights:
            assert classifier_weights == ClassifierWeights.IMAGENET1K, (
                f"Unsupported weights for linear classifier: {classifier_weights}"
            )
            weights_name = classifier_weights.value.lower()
            hash = kwargs["hash"] if "hash" in kwargs else "90d8ed92"
            model_filename = f"{backbone_name}_{weights_name}_linear_head-{hash}.pth"
            url = os.path.join(_DINOV3_BASE_URL, backbone_name, model_filename)
        else:
            url = convert_path_or_url_to_url(classifier_weights)
        state_dict = _safe_load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        linear_head.load_state_dict(state_dict, strict=True)
    return linear_head


class _LinearClassifierWrapper(nn.Module):
    def __init__(self, *, backbone: nn.Module, linear_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head

    def forward(self, x):
        x = self.backbone.forward_features(x)
        cls_token = x["x_norm_clstoken"]
        patch_tokens = x["x_norm_patchtokens"]
        linear_input = torch.cat(
            [
                cls_token,
                patch_tokens.mean(dim=1),
            ],
            dim=1,
        )
        return self.linear_head(linear_input)


def _make_dinov3_linear_classifier(
    *,
    backbone_name: str = "dinov3_vit7b16",
    pretrained: bool = True,
    classifier_weights: ClassifierWeights | str = ClassifierWeights.IMAGENET1K,
    backbone_weights: BackboneWeights | str = BackboneWeights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if backbone_name == "dinov3_vit7b16":
        backbone = dinov3_vit7b16(pretrained=pretrained, weights=backbone_weights, check_hash=check_hash)
    else:
        raise AssertionError(f"Unsupported backbone: {backbone_name}, linear classifiers are provided only for ViT-7b")
    embed_dim = backbone.embed_dim
    linear_head = _make_dinov3_linear_classification_head(
        backbone_name=backbone_name,
        embed_dim=2 * embed_dim,
        pretrained=pretrained,
        classifier_weights=classifier_weights,
        **kwargs,
    )
    return _LinearClassifierWrapper(backbone=backbone, linear_head=linear_head)


def dinov3_vit7b16_lc(
    *,
    pretrained: bool = True,
    weights: ClassifierWeights | str = ClassifierWeights.IMAGENET1K,
    backbone_weights: BackboneWeights | str = BackboneWeights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    """
    Linear classifier  on top of a DINOv3 ViT-7B/16 backbone pretrained on the LVD-1689M dataset and trained on ImageNet-1k.
    """
    return _make_dinov3_linear_classifier(
        backbone_name="dinov3_vit7b16",
        pretrained=pretrained,
        classifier_weights=weights,
        backbone_weights=backbone_weights,
        check_hash=check_hash,
        **kwargs,
    )
