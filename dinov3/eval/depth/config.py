# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import MISSING

import torch

from dinov3.eval.setup import ModelConfig


from dinov3.eval.depth.loss import LossType
from dinov3.eval.depth.models import DecoderConfig
from dinov3.eval.depth.transforms import make_depth_train_transforms, make_depth_eval_transforms
from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class Dtype(Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"

    @property
    def autocast_dtype(self):
        return {
            Dtype.BFLOAT16: torch.bfloat16,
            Dtype.FLOAT32: torch.float,
        }[self]


class ResultExtension(Enum):
    JPG = "jpg"
    PNG = "png"
    PTH = "pth"


@dataclass
class ResultConfig:
    save_results: bool = False
    extension: ResultExtension = ResultExtension.JPG
    save_resolution: int | None = (
        None  # if set, the output result image is resized to have its smallest size set to save_resolution
    )
    overlay_alpha: float = 1.0  # if alpha == 1, masks are not overlaid on the original image
    save_separate_files: bool = False  # set to true to save individual files (image, prediction, gt)


@dataclass
class DatasetsConfig:
    root: str = MISSING
    train: str = ""
    val: str = ""
    test: str = ""


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    gradient_clip: float = 35.0


@dataclass
class SchedulerConfig:
    type: str = "WarmupOneCycleLR"
    total_iter: int = 38_400  # Total number of iterations for training
    constructor_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainTransformConfig:
    img_size: Any = None
    random_crop: tuple[int, int] | None = None
    brightness_range: tuple[float, float] = (0.9, 1.1)
    rotation_angle: float = 2.5  # max rotation angle
    fixed_crop: str = "FULL"
    eval_mask: str = "FULL"


@dataclass
class EvalTransformConfig:
    img_size: Any = None
    fixed_crop: str = "FULL"
    eval_mask: str = "FULL"


@dataclass
class TransformConfig:
    train: TrainTransformConfig | None = None
    eval: EvalTransformConfig = field(default_factory=EvalTransformConfig)
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD
    normalization_constant: float = 1000.0


@dataclass
class EvalConfig:
    ignored_value: float = 0.0  # If depth pixels have this value in the dataset, they will be ignored
    align_least_squares: bool = False  # Choose whether to align predictions to ground truth during testing
    min_depth: float = 0.001  # Minimum depth to be evaluated
    max_depth: float = 10.0  # Maximum depth to be evaluated
    use_tta: bool = True  # apply test-time augmentation at evaluation time
    eval_interval: int = 1600  # number of iterations between evaluations


@dataclass
class DepthConfig:
    model: ModelConfig | None = None
    bs: int = 2
    n_gpus: int = 8
    num_workers: int = 2
    seed: int = 321
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    decoder_head: DecoderConfig = field(default_factory=DecoderConfig)
    model_dtype: Dtype | None = None
    losses: dict[LossType, float] | None = None  # For example {SIGLOSS: 1.0, GRADIENT_LOG_LOSS: 0.0}
    transforms: TransformConfig = field(default_factory=TransformConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    metrics: list[str] = field(default_factory=lambda: ["rmse", "abs_rel", "a1"])
    result_config: ResultConfig = field(default_factory=ResultConfig)
    load_from: str | None = None  # path to .pt checkpoint to resume training from
    output_dir: str = ""
    # Target device for inference; None selects CUDA when available, else CPU
    device: str | None = None


def make_depth_train_transforms_from_config(config: DepthConfig):
    assert config.datasets.train is not None
    assert config.transforms.train is not None
    transforms = make_depth_train_transforms(
        img_size=config.transforms.train.img_size,
        normalization_constant=config.transforms.normalization_constant,
        random_crop_size=config.transforms.train.random_crop,
        fixed_crop=config.transforms.train.fixed_crop,
        brightness_range=config.transforms.train.brightness_range,
        rotation_angle=config.transforms.train.rotation_angle,
        mean=config.transforms.mean,
        std=config.transforms.std,
    )
    return transforms


def make_depth_eval_transforms_from_config(config: DepthConfig, split: str = "val"):
    assert split in ["val", "test"]
    transforms = make_depth_eval_transforms(
        img_size=config.transforms.eval.img_size,
        normalization_constant=config.transforms.normalization_constant,
        fixed_crop=config.transforms.eval.fixed_crop,
        tta=config.eval.use_tta if split == "test" else False,
        mean=config.transforms.mean,
        std=config.transforms.std,
    )

    return transforms
