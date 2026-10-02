# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from collections import namedtuple
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _DepthMetric:
    name: str
    is_lower_better: bool

    @property
    def worst_value(self) -> float:
        return math.inf if self.is_lower_better else -math.inf

    def is_better(self, value1, value2) -> bool:
        sign = 1 if self.is_lower_better else -1
        return sign * value1 < sign * value2


DEPTH_METRICS = (
    _DepthMetric(name="a1", is_lower_better=False),
    _DepthMetric(name="a2", is_lower_better=False),
    _DepthMetric(name="a3", is_lower_better=False),
    _DepthMetric(name="abs_rel", is_lower_better=True),
    _DepthMetric(name="rmse", is_lower_better=True),
    _DepthMetric(name="log_10", is_lower_better=True),
    _DepthMetric(name="rmse_log", is_lower_better=True),
    _DepthMetric(name="silog", is_lower_better=True),
    _DepthMetric(name="sq_rel", is_lower_better=True),
    _DepthMetric(name="mae", is_lower_better=True),
)

DEPTH_METRICS_NAME = [metric.name for metric in DEPTH_METRICS]

_DepthMetricValues = namedtuple("DepthMetricValues", [metric.name for metric in DEPTH_METRICS])  # type: ignore


def calculate_depth_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    list_metrics: list[_DepthMetric] = list(DEPTH_METRICS),
):
    if gt.shape[0] == 0:
        return [torch.nan] * len(DEPTH_METRICS)

    if valid_mask is not None:
        valid_mask = torch.logical_and(valid_mask, gt > 0)

    gt = gt[valid_mask]
    pred = pred[valid_mask]

    metrics_dict = {}

    metric_names = [metric.name for metric in list_metrics]

    thresh = torch.maximum((gt / pred), (pred / gt))
    metrics_dict["a1"] = (thresh < 1.25).float().mean() if "a1" in metric_names else torch.nan
    metrics_dict["a2"] = (thresh < 1.25**2).float().mean() if "a2" in metric_names else torch.nan
    metrics_dict["a3"] = (thresh < 1.25**3).float().mean() if "a3" in metric_names else torch.nan

    error = gt - pred
    sq_error = error**2
    metrics_dict["mae"] = torch.mean(torch.abs(error)) if "mae" in metric_names else torch.nan
    metrics_dict["abs_rel"] = torch.mean(torch.abs(error) / gt) if "abs_rel" in metric_names else torch.nan
    metrics_dict["sq_rel"] = torch.mean(sq_error / gt) if "sq_rel" in metric_names else torch.nan

    metrics_dict["rmse"] = torch.sqrt(sq_error.mean()) if "rmse" in metric_names else torch.nan

    error_log = torch.log(gt) - torch.log(pred)
    sq_error_log = error_log**2
    metrics_dict["rmse_log"] = torch.sqrt(sq_error_log.mean()) if "rmse_log" in metric_names else torch.nan
    if "silog" in metric_names:
        # Center before squaring to avoid cancellation between two large moments.
        # Preserve NaNs from invalid predictions instead of reporting a perfect score.
        metrics_dict["silog"] = torch.sqrt(torch.var(error_log, correction=0)) * 100
    else:
        metrics_dict["silog"] = torch.nan
    metrics_dict["log_10"] = (
        (torch.abs(torch.log10(gt) - torch.log10(pred))).mean() if "log_10" in metric_names else math.inf
    )

    return _DepthMetricValues(**metrics_dict)
