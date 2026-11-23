# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from dinov3.train.metadata_utils import PrototypicalContrastiveHead


def make_head():
    head = PrototypicalContrastiveHead(embed_dim=2, n_classes=2)
    head.class_centroids.copy_(torch.eye(2))
    return head


class ProjectedGuide(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(2))
        self.head = make_head()

    def forward(self, features, labels):
        return self.head(self.projection(features), labels, lambda_=1.0)


def check_distributed_missing_labels(rank, rendezvous):
    features = torch.tensor([[1.0, 0.5], [0.2, 1.0], [1.0, 0.8], [-0.5, 1.0]])
    label_batches = [
        torch.tensor([-1, -1, 0, 1]),
        torch.tensor([0, -1, 1, 1]),
        torch.full((4,), -1),
    ]
    references = []
    for labels in label_batches:
        reference = ProjectedGuide()
        loss, accuracy = reference(features, labels)
        loss.backward()
        reference.head.update_centroids_and_concentration(features, labels)
        references.append((reference, loss.detach(), accuracy))

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        local_features = features.chunk(2)[rank]
        for labels, (reference, expected_loss, expected_accuracy) in zip(label_batches, references):
            local_labels = labels.chunk(2)[rank]
            model = DistributedDataParallel(ProjectedGuide())
            loss, accuracy = model(local_features, local_labels)
            loss.backward()

            mean_loss = loss.detach().clone()
            dist.all_reduce(mean_loss)
            torch.testing.assert_close(mean_loss / 2, expected_loss)
            torch.testing.assert_close(accuracy, expected_accuracy)
            torch.testing.assert_close(model.module.projection.weight.grad, reference.projection.weight.grad)

            head = model.module.head
            head.update_centroids_and_concentration(local_features, local_labels)
            torch.testing.assert_close(head.centroid_counts, reference.head.centroid_counts)
            torch.testing.assert_close(head.class_centroids, reference.head.class_centroids)
    finally:
        dist.destroy_process_group()


class PrototypicalMissingLabelsTest(unittest.TestCase):
    def test_loss_and_centroids_match_labeled_subset(self):
        features = torch.tensor([[1.0, 0.5], [0.2, 1.0], [0.8, 1.0]], requires_grad=True)
        labels = torch.tensor([0, -1, 1])
        head = make_head()
        reference = make_head()
        loss, accuracy = head(features, labels, lambda_=1.0)
        reference_loss, reference_accuracy = reference(features[[0, 2]], labels[[0, 2]], lambda_=1.0)
        torch.testing.assert_close(loss, reference_loss)
        self.assertEqual(accuracy, reference_accuracy)
        (gradient,) = torch.autograd.grad(loss, features)
        torch.testing.assert_close(gradient[1], torch.zeros(2))

        head.update_centroids_and_concentration(features, labels)
        reference.update_centroids_and_concentration(features[[0, 2]], labels[[0, 2]])
        torch.testing.assert_close(head.centroid_counts, reference.centroid_counts)
        torch.testing.assert_close(head.class_centroids, reference.class_centroids)
        torch.testing.assert_close(head.class_concentration, reference.class_concentration)

    def test_all_missing_labels_produce_zero_loss_and_keep_centroids(self):
        features = torch.randn(2, 2, requires_grad=True)
        labels = torch.full((2,), -1)
        head = make_head()
        loss, accuracy = head(features, labels, lambda_=1.0)
        torch.testing.assert_close(loss, torch.zeros(()))
        self.assertEqual(accuracy, 0.0)
        (gradient,) = torch.autograd.grad(loss, features)
        torch.testing.assert_close(gradient, torch.zeros_like(features))

        head.update_centroids_and_concentration(features, labels)
        torch.testing.assert_close(head.centroid_counts, torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(head.class_centroids, torch.eye(2))
        torch.testing.assert_close(head.class_concentration, torch.full((2,), head.base_temperature))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
    def test_distributed_matches_global_batch_with_uneven_valid_counts(self):
        with tempfile.TemporaryDirectory() as workdir:
            rendezvous = (Path(workdir) / "gloo").as_uri()
            mp.spawn(check_distributed_missing_labels, args=(rendezvous,), nprocs=2, join=True)
