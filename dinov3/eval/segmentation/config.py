# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dataclasses import dataclass, field
from enum import Enum
from omegaconf import MISSING
from typing import Any

import torch

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov3.eval.segmentation.models import BackboneLayersSet
from dinov3.eval.setup import ModelConfig


DEFAULT_MEAN = tuple(mean * 255 for mean in IMAGENET_DEFAULT_MEAN)
DEFAULT_STD = tuple(std * 255 for std in IMAGENET_DEFAULT_STD)


class ModelDtype(Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"

    @property
    def autocast_dtype(self):
        return {
            ModelDtype.BFLOAT16: torch.bfloat16,
            ModelDtype.FLOAT32: torch.float32,
        }[self]


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-2
    gradient_clip: float = 35.0


@dataclass
class SchedulerConfig:
    type: str = "WarmupOneCycleLR"
    total_iter: int = 40_000  # Total number of iterations for training
    constructor_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    root: str = MISSING  # Path to the dataset folder
    train: str = ""  # Dataset descriptor, e.g. "ADE20K:split=TRAIN"
    val: str = ""


@dataclass
class DecoderConfig:
    type: str = "m2f"  # Decoder type must be one of [linear, m2f]
    backbone_out_layers: BackboneLayersSet = BackboneLayersSet.LAST
    use_batchnorm: bool = True
    use_cls_token: bool = False
    use_backbone_norm: bool = True  # Uses the backbone's output normalization on all layers
    num_classes: int = 150  # Number of segmentation classes
    hidden_dim: int = 2048  # Hidden dimension, only used for M2F head
    dropout: float = 0.1  # Dropout ratio in the linear head during training


@dataclass
class TrainConfig:
    diceloss_weight: float = 0.0
    celoss_weight: float = 1.0


@dataclass
class TrainTransformConfig:
    img_size: Any = None
    random_img_size_ratio_range: tuple[float] | None = None
    crop_size: tuple[int] | None = None
    flip_prob: float = 0.0


@dataclass
class EvalTransformConfig:
    img_size: Any = None
    tta_ratios: tuple[float] = (1.0,)


@dataclass
class TransformConfig:
    train: TrainTransformConfig = field(default_factory=TrainTransformConfig)
    eval: EvalTransformConfig = field(default_factory=EvalTransformConfig)
    mean: tuple[float] = DEFAULT_MEAN
    std: tuple[float] = DEFAULT_STD


@dataclass
class EvalConfig:
    compute_metric_per_image: bool = False
    reduce_zero_label: bool = True  # For ADE20K, ignores 0 label (=background/unlabeled)
    mode: str = "slide"
    crop_size: int | None = 512
    stride: int | None = 341
    eval_interval: int = 40000
    use_tta: bool = False  # apply test-time augmentation at evaluation time


@dataclass
class SegmentationConfig:
    model: ModelConfig | None = None  # config of the DINOv3 backbone
    bs: int = 2
    n_gpus: int = 8
    num_workers: int = 6  # number of workers to use / GPU
    model_dtype: ModelDtype = ModelDtype.FLOAT32
    seed: int = 100
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    metric_to_save: str = "mIoU"  # Name of the metric to save
    decoder_head: DecoderConfig = field(default_factory=DecoderConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # Additional Parameters
    output_dir: str | None = None
    load_from: str | None = None  # path to .pt checkpoint to resume training from or evaluate from
    # Target device for inference; None selects CUDA when available, else CPU
    device: str | None = None
