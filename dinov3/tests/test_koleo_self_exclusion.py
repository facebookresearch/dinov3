# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

import math
import unittest

import torch

from dinov3.loss.koleo_loss import KoLeoLoss, KoLeoLossDistributed


class KoLeoSelfExclusionTest(unittest.TestCase):
    def test_antipodal_vectors_choose_the_other_vector(self):
        vectors = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(KoLeoLoss().pairwise_NNs_inner(vectors), torch.tensor([1, 0]))

    def test_distributed_rank_offset_excludes_only_own_rows(self):
        all_vectors = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
        criterion = KoLeoLossDistributed()
        for rank in range(2):
            with self.subTest(rank=rank):
                indices = criterion.pairwise_NNs_inner(all_vectors[rank : rank + 1], all_vectors, rank)
                torch.testing.assert_close(indices, torch.tensor([[1 - rank]]))

    def test_antipodal_forward_matches_euclidean_entropy(self):
        vectors = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64, requires_grad=True)
        for criterion in (KoLeoLoss(), KoLeoLossDistributed()):
            with self.subTest(criterion=type(criterion).__name__):
                actual = criterion(vectors)
                self.assertAlmostEqual(actual.item(), -math.log(2.0), places=6)
                gradient = torch.autograd.grad(actual, vectors, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
