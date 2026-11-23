# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Prototypical contrastive head with EMA class centroids and dynamic concentration."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.distributed import get_process_subgroup

from .gradient_scaling import GradientScalingLayer


_MISSING_LABEL = -1


class PrototypicalContrastiveHead(nn.Module):
    """Supervised prototypical contrastive loss head.

    Maintains EMA class centroids updated from the teacher. Computes ProtoNCE loss
    with per-class dynamic concentration (temperature) scaling.

    Centroids are synchronized across GPUs via all_gather on raw embeddings before
    each update, so every GPU applies the same deterministic EMA step. This avoids
    drift and dilution from periodic buffer averaging.

    Args:
        embed_dim: Embedding dimension (D).
        n_classes: Number of classes (size of centroid bank).
        base_temperature: Initial temperature / concentration value.
        centroid_momentum: EMA momentum for centroid updates.
        phi_min: Minimum concentration to prevent division by zero.
    """

    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        base_temperature: float = 0.07,
        centroid_momentum: float = 0.999,
        phi_min: float = 0.01,
    ):
        super().__init__()
        self.gradient_scaling = GradientScalingLayer()
        self.base_temperature = base_temperature
        self.centroid_momentum = centroid_momentum
        self.phi_min = phi_min
        self.n_classes = n_classes

        self.register_buffer(
            "class_centroids",
            F.normalize(torch.randn(n_classes, embed_dim), dim=-1),
        )
        self.register_buffer(
            "class_concentration",
            torch.ones(n_classes) * base_temperature,
        )
        self.register_buffer(
            "centroid_counts",
            torch.zeros(n_classes, dtype=torch.long),
        )

    @torch.no_grad()
    def reset_buffers(self) -> None:
        """Re-initialize buffers after init_weights or checkpoint load."""
        self.class_centroids.copy_(
            F.normalize(torch.randn_like(self.class_centroids), dim=-1)
        )
        self.class_concentration.fill_(self.base_temperature)
        self.centroid_counts.zero_()

    def forward(self, student_cls: Tensor, labels: Tensor, lambda_: float) -> tuple[Tensor, float]:
        """Compute PCL loss and accuracy over labeled samples. Update centroids separately."""
        x = self.gradient_scaling(student_cls, lambda_)
        x_norm = F.normalize(x.float(), dim=-1)

        all_centroids_norm = F.normalize(self.class_centroids.detach().float(), dim=-1)
        all_phi = self.class_concentration.detach().float().clamp(min=self.phi_min)

        similarities = x_norm @ all_centroids_norm.T / all_phi.unsqueeze(0)  # [B, n_classes]
        loss_sum = F.cross_entropy(similarities, labels, ignore_index=_MISSING_LABEL, reduction="sum")

        valid_labels = labels != _MISSING_LABEL
        correct = (similarities.detach().argmax(dim=1) == labels) & valid_labels
        stats = torch.stack((valid_labels.sum(), correct.sum()))
        group = get_process_subgroup()
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        if world_size > 1:
            dist.all_reduce(stats, group=group)

        valid_count, correct_count = stats.unbind()
        denominator = valid_count.clamp_min(1)
        # FSDP averages gradients across ranks. Scale each local sum so the
        # averaged gradient is the mean over all valid samples in the subgroup.
        loss = loss_sum * (world_size / denominator)
        accuracy = (correct_count / denominator).item()
        return loss, accuracy

    @torch.no_grad()
    def update_centroids_and_concentration(self, teacher_cls: Tensor, labels: Tensor) -> None:
        """Vectorized EMA update of centroids and concentration from teacher embeddings.

        Uses all_gather to assemble the global batch so every GPU applies the same
        deterministic update. Concentration (phi) is measured against HISTORICAL
        centroids before the centroid EMA step, and only updated for classes with
        Z >= 2 samples (variance is ill-defined for a single point).
        """
        teacher_norm = F.normalize(teacher_cls.detach(), dim=-1)

        if dist.is_initialized() and dist.get_world_size() > 1:
            all_features = [torch.empty_like(teacher_norm) for _ in range(dist.get_world_size())]
            dist.all_gather(all_features, teacher_norm)
            teacher_norm = torch.cat(all_features, dim=0)
            all_labels = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
            dist.all_gather(all_labels, labels)
            labels = torch.cat(all_labels, dim=0)

        # Keep the fixed-size gather on every rank, including ranks that had
        # no valid local labels. Filter the shared batch before indexing classes.
        valid_labels = labels != _MISSING_LABEL
        labels = labels[valid_labels]
        teacher_norm = teacher_norm[valid_labels].float()
        _, D = teacher_norm.shape
        counts = torch.bincount(labels, minlength=self.n_classes)
        valid = counts > 0

        historical = F.normalize(self.class_centroids, dim=-1)
        dists_sq = ((teacher_norm - historical[labels]) ** 2).sum(dim=-1)  # [N]
        dist_sum = torch.zeros(self.n_classes, device=teacher_norm.device)
        dist_sum.scatter_add_(0, labels, dists_sq)

        feat_sum = torch.zeros(self.n_classes, D, device=teacher_norm.device)
        feat_sum.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), teacher_norm)
        batch_centroids = feat_sum / counts.unsqueeze(1).clamp(min=1).float()

        m = self.centroid_momentum
        first_seen = valid & (self.centroid_counts == 0)
        ema_mask = valid & ~first_seen
        self.class_centroids[first_seen] = batch_centroids[first_seen]
        self.class_centroids[ema_mask] = m * self.class_centroids[ema_mask] + (1 - m) * batch_centroids[ema_mask]

        # Phi update only for classes with Z >= 2 AND previously seen
        phi_mask = (counts >= 2) & (self.centroid_counts > 0)
        if phi_mask.any():
            batch_phi = (dist_sum[phi_mask] / counts[phi_mask].float()).sqrt()
            self.class_concentration[phi_mask] = m * self.class_concentration[phi_mask] + (1 - m) * batch_phi

        self.centroid_counts += counts
