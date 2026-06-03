# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Any, Tuple, Union
from enum import Enum

from torch import nn

from .backbones import dinov3_vitl16, Weights as BackboneWeights, convert_path_or_url_to_url
from .utils import _DINOV3_BASE_URL, _safe_load_state_dict_from_url


class DINOTxtWeights(Enum):
    LVTD2300M = "LVTD2300M"


# returns dinotxt model and tokenizer
def dinov3_vitl16_dinotxt_tet1280d20h24l(
    *,
    pretrained: bool = True,
    weights: Union[DINOTxtWeights, str] = DINOTxtWeights.LVTD2300M,
    backbone_weights: Union[BackboneWeights, str] = BackboneWeights.LVD1689M,
    bpe_path_or_url: str = "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz",
    check_hash: bool = False,
) -> Tuple[nn.Module, Any]:
    from dinov3.eval.text.dinotxt_model import DINOTxt, DINOTxtConfig
    from dinov3.eval.text.text_transformer import TextTransformer
    from dinov3.eval.text.tokenizer import get_tokenizer

    dinotxt_config = DINOTxtConfig(
        embed_dim=2048,
        vision_model_freeze_backbone=True,
        vision_model_train_img_size=224,
        vision_model_use_class_token=True,
        vision_model_use_patch_tokens=True,
        vision_model_num_head_blocks=2,
        vision_model_head_blocks_drop_path=0.3,
        vision_model_use_linear_projection=False,
        vision_model_patch_tokens_pooler_type="mean",
        vision_model_patch_token_layer=1,  # which layer to take patch tokens from
        # 1 - last layer, 2 - second last layer, etc.
        text_model_freeze_backbone=False,
        text_model_num_head_blocks=0,
        text_model_head_blocks_is_causal=False,
        text_model_head_blocks_drop_prob=0.0,
        text_model_tokens_pooler_type="argmax",
        text_model_use_linear_projection=True,
        init_logit_scale=math.log(1 / 0.07),
        init_logit_bias=None,
        freeze_logit_scale=False,
    )
    vision_backbone = dinov3_vitl16(pretrained=pretrained, weights=backbone_weights)
    text_backbone = TextTransformer(
        context_length=77,
        vocab_size=49408,
        dim=1280,
        num_heads=20,
        num_layers=24,
        ffn_ratio=4,
        is_causal=True,
        ls_init_value=None,
        dropout_prob=0.0,
    )
    model = DINOTxt(model_config=dinotxt_config, vision_backbone=vision_backbone, text_backbone=text_backbone)
    if pretrained:
        model.visual_model.backbone = vision_backbone
        model.eval()
        if type(weights) is DINOTxtWeights and weights == DINOTxtWeights.LVTD2300M:
            url = f"{_DINOV3_BASE_URL}/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
        elif type(weights) is DINOTxtWeights and weights != DINOTxtWeights.LVTD2300M:
            raise AssertionError(f"Unsuported weights for DINOTxt: {weights}")
        else:
            url = convert_path_or_url_to_url(weights)
        vision_head_and_text_encoder_state_dict = _safe_load_state_dict_from_url(
            url, map_location="cpu", check_hash=check_hash
        )
        model.load_state_dict(vision_head_and_text_encoder_state_dict, strict=False)
    else:
        model.init_weights()
    return model, get_tokenizer(bpe_path_or_url=bpe_path_or_url)
