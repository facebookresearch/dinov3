# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import unittest

import torch

from dinov3.train.guided_ssl_meta_arch import GuidedSSLMetaArch
from dinov3.train.metadata_utils import GradientScalingLayer


class GuideGradientNormalizationTest(unittest.TestCase):
    def test_preserves_relative_weights_for_equal_raw_gradients(self):
        features = torch.ones(2, requires_grad=True)
        metrics = {}
        total = GuidedSSLMetaArch._apply_grad_norm_normalization(
            {"first": features[0], "second": features[1]},
            {"first": 0.1, "second": 0.9},
            features,
            metrics,
        )
        (gradient,) = torch.autograd.grad(total, features)
        torch.testing.assert_close(gradient, torch.tensor([0.1, 0.9]))
        torch.testing.assert_close(metrics["grad_norm/target"], torch.tensor(1.0))

    def test_normalizes_unequal_raw_norms_and_keeps_reversed_gradient(self):
        features = torch.ones(2, requires_grad=True)
        reversed_feature = GradientScalingLayer()(features[1], lambda_=-1.0)
        metrics = {}
        total = GuidedSSLMetaArch._apply_grad_norm_normalization(
            {"first": 2 * features[0], "adversarial": 8 * reversed_feature},
            {"first": 0.1, "adversarial": 0.9},
            features,
            metrics,
        )
        (gradient,) = torch.autograd.grad(total, features)
        torch.testing.assert_close(gradient, torch.tensor([0.4, -3.6]))
        torch.testing.assert_close(metrics["grad_norm/first"], torch.tensor(2.0))
        torch.testing.assert_close(metrics["grad_norm/adversarial"], torch.tensor(8.0))
        torch.testing.assert_close(metrics["grad_norm/target"], torch.tensor(4.0))

    def test_excludes_disabled_and_disconnected_guides_from_target(self):
        features = torch.ones(3, requires_grad=True)
        head_parameter = torch.ones((), requires_grad=True)
        metrics = {}
        total = GuidedSSLMetaArch._apply_grad_norm_normalization(
            {
                "first": 2 * features[0],
                "second": 8 * features[1],
                "disabled": 100 * features[2],
                "head_only": head_parameter.square(),
            },
            {"first": 0.1, "second": 0.9, "disabled": 0.0, "head_only": 0.5},
            features,
            metrics,
        )
        gradient, head_gradient = torch.autograd.grad(total, (features, head_parameter))
        torch.testing.assert_close(gradient, torch.tensor([0.4, 3.6, 0.0]))
        torch.testing.assert_close(head_gradient, torch.tensor(1.0))
        torch.testing.assert_close(metrics["grad_norm/target"], torch.tensor(4.0))

    def test_uses_plain_weighted_sum_with_one_active_guide(self):
        features = torch.ones(2, requires_grad=True)
        metrics = {}
        total = GuidedSSLMetaArch._apply_grad_norm_normalization(
            {"active": 2 * features[0], "disabled": features[1]},
            {"active": 0.1, "disabled": 0.0},
            features,
            metrics,
        )
        (gradient,) = torch.autograd.grad(total, features)
        torch.testing.assert_close(gradient, torch.tensor([0.2, 0.0]))
        self.assertNotIn("grad_norm/target", metrics)
