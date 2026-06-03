# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum

import torch

from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.eval.detection.models.detr import PostProcess, build_model
from dinov3.eval.detection.models.position_encoding import PositionEncoding

from .backbones import Weights as BackboneWeights, dinov3_vit7b16, dinov3_vitl16plus, convert_path_or_url_to_url
from .utils import _DINOV3_BASE_URL, _safe_load_state_dict_from_url


class DetectionWeights(Enum):
    COCO2017 = "COCO2017"


class DetectorWithProcessor(torch.nn.Module):
    """
    takes as input a list of (3, H, W) normalized image tensors and outputs
    a list of dicts with keys "scores", "labels" and "boxes" (format XYXY)
    """

    def __init__(self, detector, postprocessor):
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor

    def forward(self, samples: list[torch.Tensor]):
        outputs = self.detector(samples)
        sizes_tensor = torch.tensor([sample.shape[1:] for sample in samples], device=samples[0].device)  # N * [3, H, W]
        return self.postprocessor(outputs, target_sizes=sizes_tensor, original_target_sizes=sizes_tensor)


def _make_dinov3_detector(
    *,
    backbone_name: str,
    pretrained: bool = True,
    detector_weights: str | DetectionWeights,
    backbone_weights: str | BackboneWeights,
    check_hash: bool = False,
    **kwargs,
):
    detection_kwargs = dict(
        with_box_refine=True,
        two_stage=True,
        mixed_selection=True,
        look_forward_twice=True,
        k_one2many=6,
        lambda_one2many=1.0,
        num_queries_one2one=1500,
        num_queries_one2many=1500,
        reparam=True,
        position_embedding=PositionEncoding.SINE,
        num_feature_levels=1,
        dec_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        norm_type="pre_norm",
        proposal_feature_levels=4,
        proposal_min_size=50,
        decoder_type="global_rpe_decomp",
        decoder_use_checkpoint=False,
        decoder_rpe_hidden_dim=512,
        decoder_rpe_type="linear",
        layers_to_use=None,
        blocks_to_train=None,
        add_transformer_encoder=True,
        num_encoder_layers=6,
        backbone_use_layernorm=False,
        num_classes=91,  # 91 classes in COCO
        aux_loss=True,
        topk=1500,
        hidden_dim=768,
        nheads=8,
    )
    config = DetectionHeadConfig(**detection_kwargs)
    backbone_class = dict(dinov3_vit7b16=dinov3_vit7b16, dinov3_vitl16plus=dinov3_vitl16plus)[backbone_name]
    n_windows_sqrt = dict(dinov3_vit7b16=3, dinov3_vitl16plus=2)[backbone_name]
    backbone = backbone_class(pretrained=pretrained, weights=backbone_weights, check_hash=check_hash)
    backbone.eval()

    config.n_windows_sqrt = n_windows_sqrt
    config.proposal_in_stride = backbone.patch_size
    config.proposal_tgt_strides = [int(m * backbone.patch_size) for m in (0.5, 1, 2, 4)]

    if config.layers_to_use is None:
        # e.g. [2, 5, 8, 11] for a backbone with 12 blocks, similar to depth evaluation
        config.layers_to_use = [m * backbone.n_blocks // 4 - 1 for m in range(1, 5)]

    detector = build_model(backbone, config)
    if pretrained:
        if type(detector_weights) is DetectionWeights and detector_weights == DetectionWeights.COCO2017:
            assert detector_weights == DetectionWeights.COCO2017, f"Unsupported detector weights {detector_weights}"
            detection_weights_name = detector_weights.value.lower()
            hash = kwargs["hash"] if "hash" in kwargs else "b0235ff7"
            model_filename = f"{backbone_name}_{detection_weights_name}_detr_head-{hash}.pth"
            url = os.path.join(_DINOV3_BASE_URL, backbone_name, model_filename)
        else:
            url = convert_path_or_url_to_url(detector_weights)
        state_dict = _safe_load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)["model"]
        detector.load_state_dict(state_dict, strict=False)
    # Necessary for inference
    detector.num_queries = detector.num_queries_one2one
    detector.transformer.two_stage_num_proposals = detector.num_queries

    postprocessor = PostProcess(config.topk, config.reparam)
    model = DetectorWithProcessor(detector=detector, postprocessor=postprocessor)
    return model


def dinov3_vit7b16_de(
    *,
    pretrained: bool = True,
    weights: DetectionWeights | str = DetectionWeights.COCO2017,
    backbone_weights: BackboneWeights | str = BackboneWeights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    return _make_dinov3_detector(
        backbone_name="dinov3_vit7b16",
        pretrained=pretrained,
        detector_weights=weights,
        backbone_weights=backbone_weights,
        check_hash=check_hash,
        **kwargs,
    )
