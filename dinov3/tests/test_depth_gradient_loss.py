import unittest

import torch

from dinov3.eval.depth.loss import GradientLogLoss, GradientLoss


def valid_edge_oracle(pred, target, mask, logarithmic):
    total = pred[mask].sum() * 0
    for stride in (1, 2, 4, 6):
        p, t, m = pred[..., ::stride, ::stride], target[..., ::stride, ::stride], mask[..., ::stride, ::stride]
        count = int(m.sum())
        if not count:
            continue
        for axis in (-2, -1):
            a = [slice(None)] * p.ndim
            b = [slice(None)] * p.ndim
            a[axis], b[axis] = slice(None, -2), slice(2, None)
            valid = m[tuple(a)] & m[tuple(b)]
            pa, pb = p[tuple(a)][valid], p[tuple(b)][valid]
            ta, tb = t[tuple(a)][valid], t[tuple(b)][valid]
            if logarithmic:
                pa, pb = (pa + 0.001).log(), (pb + 0.001).log()
                ta, tb = (ta + 0.001).log(), (tb + 0.001).log()
            total = total + ((pa - ta) - (pb - tb)).abs().sum() / count
    return total


class MaskedGradientLossTests(unittest.TestCase):
    def test_sparse_mask_can_disappear_at_coarser_scales(self):
        for cls in (GradientLoss, GradientLogLoss):
            with self.subTest(cls=cls):
                pred = torch.arange(1.0, 50.0).reshape(1, 7, 7).requires_grad_()
                mask = torch.zeros_like(pred, dtype=torch.bool)
                mask[:, 1, 1] = mask[:, 1, 3] = True
                target = torch.ones_like(pred)
                value = cls()(pred, target, mask)
                expected = valid_edge_oracle(pred, target, mask, cls is GradientLogLoss)
                torch.testing.assert_close(value, expected)
                torch.testing.assert_close(
                    torch.autograd.grad(value, pred, retain_graph=True)[0],
                    torch.autograd.grad(expected, pred)[0],
                )

    def test_excluded_nonfinite_and_negative_depths_do_not_poison_gradients(self):
        for cls in (GradientLoss, GradientLogLoss):
            with self.subTest(cls=cls):
                pred = torch.ones(1, 7, 7, dtype=torch.float64)
                target = pred.clone()
                mask = torch.ones_like(pred, dtype=torch.bool)
                mask[:, 2, 2] = mask[:, 4, 4] = False
                pred[:, 2, 2], target[:, 2, 2] = float("nan"), float("nan")
                pred[:, 4, 4], target[:, 4, 4] = -5.0, -2.0
                pred[:, 1, 1] = 2.0
                pred.requires_grad_()
                value = cls()(pred, target, mask)
                expected = valid_edge_oracle(pred, target, mask, cls is GradientLogLoss)
                torch.testing.assert_close(value, expected)
                value.backward()
                self.assertTrue(torch.isfinite(pred.grad).all())
                torch.testing.assert_close(pred.grad[~mask], torch.zeros_like(pred.grad[~mask]))

    def test_empty_masks_have_connected_zero_loss(self):
        for cls in (GradientLoss, GradientLogLoss):
            with self.subTest(cls=cls):
                pred = torch.full((1, 5, 5), float("nan"), requires_grad=True)
                mask = torch.zeros_like(pred, dtype=torch.bool)
                value = cls()(pred, torch.full_like(pred, -1.0), mask)
                self.assertEqual(value.item(), 0.0)
                value.backward()
                torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_dense_losses_match_explicit_valid_edges(self):
        torch.manual_seed(42)
        pred = torch.rand(2, 9, 9, dtype=torch.float64) + 0.5
        target = torch.rand_like(pred) + 0.5
        mask = torch.ones_like(pred, dtype=torch.bool)
        for cls in (GradientLoss, GradientLogLoss):
            with self.subTest(cls=cls):
                torch.testing.assert_close(
                    cls()(pred, target), valid_edge_oracle(pred, target, mask, cls is GradientLogLoss)
                )


if __name__ == "__main__":
    unittest.main()
