# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import gc
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torchmetrics import Metric

import dinov3.distributed as distributed
from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
from dinov3.eval.accumulators import NoOpAccumulator, ResultsAccumulator
from dinov3.logging import MetricLogger

logger = logging.getLogger("dinov3")


class LossType(Enum):
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(self, samples):
        return nn.functional.normalize(self._model(samples), dim=1, p=2)


class ModelWithMultiScale(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mode: str = "bilinear") -> None:
        super().__init__()
        self._model = model
        self._mode = mode

    def forward(self, samples):
        output = None
        for scale in (1, 0.5**0.5, 0.5):
            if scale == 1:
                resized_samples = samples.clone()
            else:
                resized_samples = nn.functional.interpolate(
                    samples, scale_factor=scale, mode=self._mode, align_corners=False
                )
            scale_output = self._model(resized_samples).clone()
            if output is None:
                output = scale_output
            else:
                output += scale_output
        return output / 3


def wrap_model(
    model: nn.Module,
    *,
    normalize: bool = True,
    multi_scale: bool = False,
) -> nn.Module:
    logger.info("multi-scale: {}".format("enabled" if multi_scale else "disabled"))
    if multi_scale:
        model = ModelWithMultiScale(model)

    logger.info("normalize: {}".format("enabled" if normalize else "disabled"))
    if normalize:
        model = ModelWithNormalize(model)
    return model


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n, autocast_ctx, reshape=False, return_class_token=True):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n = n  # Layer indices (Sequence) or n last layers (int) to take
        self.autocast_ctx = autocast_ctx
        self.reshape = reshape
        self.return_class_token = return_class_token

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images,
                    n=self.n,
                    reshape=self.reshape,
                    return_class_token=self.return_class_token
                )
        return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, Metric],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    accumulate_results: bool = False,
):
    gc.collect()  # Avoids garbage collection errors in DataLoader workers
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    accumulator_class = ResultsAccumulator if accumulate_results else NoOpAccumulator
    accumulators = {k: accumulator_class() for k in postprocessors.keys()}

    # Dataset needs to be wrapped in fairvit.data.adapters.DatasetWithEnumeratedTargets
    for samples, (index, targets), *_ in metric_logger.log_every(data_loader, 10, header):
        samples, targets, index = samples[index >= 0], targets[index >= 0], index[index >= 0]
        if len(index) == 0:
            continue

        outputs = model(samples.to(device))
        index = index.to(device)
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)
            accumulators[k].update(preds=metric_inputs["preds"], target=metric_inputs["target"], index=index)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # accumulator.accumulate() returns None for the NoOpAccumulator
    accumulated_results = {k: accumulator.accumulate() for k, accumulator in accumulators.items()}

    return metric_logger_stats, stats, accumulated_results


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_world_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dataset, batch_size, num_workers, gather_on_cpu=False, device=None):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
    )
    return extract_features_with_dataloader(
        model, data_loader, sample_count, gather_on_cpu=gather_on_cpu, device=device
    )


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False, device=None):
    from dinov3.eval.setup import resolve_device

    device = resolve_device(device)
    gather_device = torch.device("cpu") if gather_on_cpu else device
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.to(device, non_blocking=True)
        labels_rank = labels_rank.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features Shape {features.shape}")
    logger.info(f"Labels Shape {all_labels.shape}")

    return features, all_labels

def save_features_dict(features_dict: Dict[str, torch.Tensor], path: str) -> None:
    logger.info(f'saving features to "{path}"')

    for key, value in features_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor)

    _, ext = os.path.splitext(path)
    if ext == ".pt":
        torch.save(features_dict, path)
    elif ext == ".npy":
        numpy_features_dict = {  # Convert to NumPy arrays (if possible)
            key: value.cpu().numpy() for key, value in features_dict.items()
        }
        np.save(path, numpy_features_dict, allow_pickle=True)
    else:
        raise ValueError(f'Unsupported features dict extension "{ext}"')


def load_features_dict(path: str) -> Dict[str, torch.Tensor]:
    logger.info(f'loading features from "{path}"')

    _, ext = os.path.splitext(path)
    if ext == ".pt":
        features_dict = torch.load(path)
    elif ext == ".npy":
        numpy_features_dict = np.load(path, allow_pickle=True).item()
        features_dict = {key: torch.from_numpy(value) for key, value in numpy_features_dict.items()}
    else:
        raise ValueError(f'Unsupported features dict extension "{ext}"')

    for key, value in features_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor)

    return features_dict


def average_metrics(eval_metrics_dict: dict[Any, dict[str, torch.Tensor]], ignore_keys: List[str] = []):
    """
    Function that computes the average and the std on a metrics dict.
    A linear evaluation dictionary contains "best_classifier",
    so this specific key is removed for computing aggregated metrics.
    """
    output_metrics_dict = {}
    metrics = [metric for metric in eval_metrics_dict[0].keys() if metric not in ignore_keys]
    for metric in metrics:
        stats_tensor = torch.tensor([stat[metric] for stat in eval_metrics_dict.values()])
        output_metrics_dict[metric + "_mean"] = stats_tensor.mean().item()
        output_metrics_dict[metric + "_std"] = torch.std(stats_tensor).item()

    return output_metrics_dict


def save_results(
    preds: torch.Tensor,
    target: torch.Tensor,
    output_dir: str,
    filename_suffix: Optional[str] = None,
) -> None:
    """
    Helper to save predictions from a model and their associated targets, aligned by their index
    """
    filename_suffix = "" if filename_suffix is None else f"_{filename_suffix}"
    preds_filename = f"preds{filename_suffix}.npy"
    target_filename = f"target{filename_suffix}.npy"
    preds_path = os.path.join(output_dir, preds_filename)
    target_path = os.path.join(output_dir, target_filename)
    logger.info(f"Saving to {preds_path}")
    np.save(preds_path, preds.cpu().numpy())
    logger.info(f"Saving to {target_path}")
    np.save(target_path, target.cpu().numpy())
