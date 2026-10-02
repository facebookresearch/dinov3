# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

import unittest

import torch

from dinov3.eval.depth.schedulers import WarmupMultiStepLR


class MultiStepGroupPhaseTest(unittest.TestCase):
    def make_scheduler(self, rates, milestones):
        groups = [{"params": [torch.nn.Parameter(torch.ones(()))], "lr": rate} for rate in rates]
        optimizer = torch.optim.SGD(groups)
        scheduler = WarmupMultiStepLR(optimizer, total_steps=10, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def test_group_rates_preserve_their_ratios(self):
        optimizer, scheduler = self.make_scheduler([1.0, 2.0, 4.0], [0.5, 0.9, 1.0])
        for step in range(11):
            expected_phase = sum(scheduler._step_count >= 10 * point for point in (0.5, 0.9, 1.0))
            for group, initial in zip(optimizer.param_groups, (1.0, 2.0, 4.0)):
                self.assertAlmostEqual(group["lr"] / initial, 0.1**expected_phase, places=12)
            if step < 10:
                optimizer.step()
                scheduler.step()

    def test_coincident_milestones_are_all_applied(self):
        optimizer, scheduler = self.make_scheduler([1.0], [0.2, 0.2, 0.7])
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            expected_phase = sum(scheduler._step_count >= 10 * point for point in (0.2, 0.2, 0.7))
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1**expected_phase, places=12)

    def test_get_lr_is_idempotent(self):
        optimizer, scheduler = self.make_scheduler([1.0, 2.0], [0.1, 0.1, 0.2])
        first = scheduler.get_lr()
        second = scheduler.get_lr()
        self.assertEqual(first, second)
        self.assertEqual(first[1], 2 * first[0])

    def test_state_restore_keeps_step_based_phase(self):
        optimizer, scheduler = self.make_scheduler([1.0, 2.0], [0.3, 0.6, 1.0])
        for _ in range(4):
            optimizer.step()
            scheduler.step()
        other_optimizer, other_scheduler = self.make_scheduler([1.0, 2.0], [0.3, 0.6, 1.0])
        other_optimizer.load_state_dict(optimizer.state_dict())
        other_scheduler.load_state_dict(scheduler.state_dict())
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            other_optimizer.step()
            other_scheduler.step()
            self.assertEqual(scheduler.get_last_lr(), other_scheduler.get_last_lr())


if __name__ == "__main__":
    unittest.main()
