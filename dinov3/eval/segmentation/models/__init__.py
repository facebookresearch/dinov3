# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from enum import Enum
from functools import partial

import torch

from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
from dinov3.eval.segmentation.models.heads.linear_head import LinearHead
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead
from dinov3.eval.setup import get_autocast_device_type, resolve_device
from dinov3.eval.utils import ModelWithIntermediateLayers


class BackboneLayersSet(Enum):
    """
    Set of intermediate layers to take from the backbone.
    """

    LAST = "LAST"  # extracting only the last layer
    FOUR_LAST = "FOUR_LAST"  # extracting the four last layers
    FOUR_EVEN_INTERVALS = "FOUR_EVEN_INTERVALS"  # extracting outputs every 1/4 of the total number of blocks


def _get_backbone_out_indices(
    model: torch.nn.Module,
    backbone_out_layers: BackboneLayersSet = BackboneLayersSet.FOUR_EVEN_INTERVALS,
):
    """
    Get indices for output layers of the ViT backbone. For now there are 3 options available:
    BackboneLayersSet.LAST : only extract the last layer, used in segmentation tasks with a bn head.
    BackboneLayersSet.FOUR_EVEN_INTERVALS : extract outputs every 1/4 of the total number of blocks
    Reference outputs in 'FOUR_EVEN_INTERVALS' mode :
    ViT/S (12 blocks): [2, 5, 8, 11]
    ViT/B (12 blocks): [2, 5, 8, 11]
    ViT/L (24 blocks): [5, 11, 17, 23] (classic), [4, 11, 17, 23] (used in the paper)
    ViT/g (40 blocks): [9, 19, 29, 39]
    """
    n_blocks = getattr(model, "n_blocks", 1)
    if backbone_out_layers == BackboneLayersSet.LAST:
        out_indices = [n_blocks - 1]
    elif backbone_out_layers == BackboneLayersSet.FOUR_LAST:
        out_indices = [i for i in range(n_blocks - 4, n_blocks)]
    elif backbone_out_layers == BackboneLayersSet.FOUR_EVEN_INTERVALS:
        # Take indices that were used in the paper (for ViT/L only)
        if n_blocks == 24:
            out_indices = [4, 11, 17, 23]
        else:
            out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    assert all([out_index < n_blocks for out_index in out_indices])
    return out_indices


class FeatureDecoder(torch.nn.Module):
    def __init__(self, segmentation_model: torch.nn.ModuleList, autocast_ctx):
        super().__init__()
        self.segmentation_model = segmentation_model
        self.autocast_ctx = autocast_ctx

    def forward(self, inputs):
        with self.autocast_ctx():
            for module in self.segmentation_model:
                inputs = module.forward(inputs)
        return inputs

    def predict(self, inputs, rescale_to=(512, 512)):
        with torch.inference_mode():
            with self.autocast_ctx():
                out = self.segmentation_model[0](inputs)  # backbone forward
                out = self.segmentation_model[1].predict(out, rescale_to=rescale_to)  # decoder head prediction
        return out


def build_segmentation_decoder(
    backbone_model,
    backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
    decoder_type="linear",
    hidden_dim=2048,
    num_classes=150,
    dropout=0.1,
    autocast_dtype=torch.float32,
    device=None,
):
    backbone_indices_to_use = _get_backbone_out_indices(backbone_model, backbone_out_layers)
    device = resolve_device(device)
    autocast_ctx = partial(
        torch.autocast,
        device_type=get_autocast_device_type(device),
        enabled=True,
        dtype=autocast_dtype,
    )
    if decoder_type == "m2f":
        backbone_model = DINOv3_Adapter(
            backbone_model,
            interaction_indexes=backbone_indices_to_use,
        )
        backbone_model.eval()
        embed_dim = backbone_model.backbone.embed_dim
        patch_size = backbone_model.patch_size
        decoder = Mask2FormerHead(
            input_shape={
                "1": [embed_dim, patch_size * 4, patch_size * 4, 4],
                "2": [embed_dim, patch_size * 2, patch_size * 2, 4],
                "3": [embed_dim, patch_size, patch_size, 4],
                "4": [embed_dim, int(patch_size / 2), int(patch_size / 2), 4],
            },
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            ignore_value=255,
        )
    elif decoder_type == "linear":
        backbone_model = ModelWithIntermediateLayers(
            backbone_model,
            n=backbone_indices_to_use,
            autocast_ctx=autocast_ctx,
            reshape=True,
            return_class_token=False,
        )
        # Important: we freeze the backbone
        backbone_model.requires_grad_(False)
        embed_dim = backbone_model.feature_model.embed_dim
        if isinstance(embed_dim, int):
            if backbone_out_layers in [BackboneLayersSet.FOUR_LAST, BackboneLayersSet.FOUR_EVEN_INTERVALS]:
                embed_dim = [embed_dim] * 4
            else:
                embed_dim = [embed_dim]
        decoder = LinearHead(
            in_channels=embed_dim,
            n_output_channels=num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f'Unsupported decoder "{decoder_type}"')

    segmentation_model = FeatureDecoder(
        torch.nn.ModuleList(
            [
                backbone_model,
                decoder,
            ]
        ),
        autocast_ctx=autocast_ctx,
    )
    return segmentation_model
