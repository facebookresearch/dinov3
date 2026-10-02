# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

import unittest

import torch

from dinov3.eval.depth.metrics import calculate_depth_metrics


class SILogStabilityTest(unittest.TestCase):
    def test_small_variation_on_large_log_bias(self):
        errors = torch.tensor([30.0, 30.001, 29.999, 30.002], dtype=torch.float32)
        ground_truth = torch.ones_like(errors)
        prediction = torch.exp(-errors)
        actual = calculate_depth_metrics(ground_truth, prediction).silog
        recovered = torch.log(ground_truth.double()) - torch.log(prediction.double())
        expected = ((recovered - recovered.mean()).square().mean().sqrt() * 100).float()
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=1e-5)
        self.assertGreater(actual.item(), 0)

    def test_constant_scale_and_singleton_have_zero_dispersion(self):
        for size in (1, 8):
            with self.subTest(size=size):
                ground_truth = torch.ones(size, dtype=torch.float64)
                actual = calculate_depth_metrics(ground_truth, 7 * ground_truth).silog
                torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_invalid_predictions_do_not_receive_perfect_score(self):
        for prediction in (float("nan"), -1.0, 0.0):
            with self.subTest(prediction=prediction):
                actual = calculate_depth_metrics(torch.ones(3), torch.full((3,), prediction)).silog
                self.assertTrue(torch.isnan(actual))

    def test_all_masked_pixels_remain_undefined(self):
        actual = calculate_depth_metrics(torch.ones(3), torch.ones(3), torch.zeros(3, dtype=torch.bool)).silog
        self.assertTrue(torch.isnan(actual))


if __name__ == "__main__":
    unittest.main()
