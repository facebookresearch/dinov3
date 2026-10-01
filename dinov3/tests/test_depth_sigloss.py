import unittest

import torch

from dinov3.eval.depth.loss import SigLoss


class SigLossTests(unittest.TestCase):
    def test_matches_population_moment_objective_and_gradient(self):
        for count in (1, 2, 17):
            with self.subTest(count=count):
                pred = torch.linspace(0.5, 3.0, count, dtype=torch.float64, requires_grad=True)
                target = torch.linspace(1.1, 2.0, count, dtype=torch.float64)
                g = (pred + 0.001).log() - (target + 0.001).log()
                expected = (g.square().mean() - 0.85 * g.mean().square()).sqrt()
                actual = SigLoss(warm_up=False)(pred, target)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(
                    torch.autograd.grad(actual, pred, retain_graph=True)[0],
                    torch.autograd.grad(expected, pred)[0],
                )

    def test_empty_mask_returns_connected_zero_without_using_invalid_depths(self):
        pred = torch.tensor([float("nan"), -3.0], requires_grad=True)
        target = torch.tensor([-2.0, float("nan")])
        loss = SigLoss(warm_up=True, warm_iter=1)
        value = loss(pred, target, torch.zeros(2, dtype=torch.bool))
        self.assertEqual(value.item(), 0.0)
        value.backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))
        self.assertEqual(loss.warm_up_counter, 0)

    def test_warmup_transitions_to_population_variance(self):
        pred = torch.tensor([1.0, 3.0], dtype=torch.float64)
        target = torch.tensor([2.0, 1.0], dtype=torch.float64)
        g = (pred + 0.001).log() - (target + 0.001).log()
        loss = SigLoss(warm_up=True, warm_iter=2)
        warm = (0.15 * g.mean().square()).sqrt()
        torch.testing.assert_close(loss(pred, target), warm)
        torch.testing.assert_close(loss(pred, target), warm)
        torch.testing.assert_close(loss(pred, target), (g.square().mean() - 0.85 * g.mean().square()).sqrt())

    def test_perfect_predictions_have_zero_finite_gradient(self):
        pred = torch.tensor([1.0, 2.0], requires_grad=True)
        value = SigLoss(warm_up=False)(pred, pred.detach())
        self.assertEqual(value.item(), 0.0)
        value.backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


if __name__ == "__main__":
    unittest.main()
