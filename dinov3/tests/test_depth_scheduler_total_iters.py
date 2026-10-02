# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

import unittest

import torch

from dinov3.eval.depth.schedulers import build_scheduler


class SchedulerTotalItersTest(unittest.TestCase):
    def test_training_duration_reaches_each_iterative_scheduler(self):
        for scheduler_type in ("ConstantLR", "LinearLR", "PolynomialLR"):
            with self.subTest(scheduler_type=scheduler_type):
                parameter = torch.nn.Parameter(torch.ones(()))
                optimizer = torch.optim.SGD([parameter], lr=1.0)
                scheduler = build_scheduler(scheduler_type, optimizer, 1.0, 17, {})
                self.assertEqual(scheduler.total_iters, 17)

    def test_linear_decay_matches_configured_training_duration(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = build_scheduler("LinearLR", optimizer, 1.0, 10, {"start_factor": 1.0, "end_factor": 0.0})
        for step in range(1, 11):
            optimizer.step()
            scheduler.step()
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1 - step / 10, places=10)

    def test_constructor_configuration_is_not_mutated(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        arguments = {"start_factor": 0.5}
        build_scheduler("LinearLR", optimizer, 1.0, 17, arguments)
        self.assertEqual(arguments, {"start_factor": 0.5})


if __name__ == "__main__":
    unittest.main()
