# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F

import dinov3.distributed as dist


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        # Exclude self even when a real neighbor has cosine similarity -1.
        dots.view(-1)[:: (n + 1)].fill_(-torch.inf)
        _, indices = torch.max(dots, dim=1)  # max inner prod -> min distance
        return indices

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            indices = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[indices])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss


class KoLeoLossDistributed(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self, topk=1, loss_group_size: int | None = None):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        self.topk = topk
        self.loss_group_size = loss_group_size  # Size of the nearest neighbor set. If None, uses global batch size.

    def pairwise_NNs_inner(self, x, all_x, rank):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, all_x.t())  # local_B x global_B
        local_B, global_B = dots.shape
        # Exclude self even when a real neighbor has cosine similarity -1.
        dots.view(-1)[rank * local_B :: (global_B + 1)].fill_(-torch.inf)
        _, indices = torch.topk(dots, dim=1, k=self.topk)  # max inner prod -> min distance
        return indices

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)  # local_B x D

            if dist.is_enabled():
                all_student_outputs = torch.cat(torch_dist.nn.all_gather(student_output), dim=0)  # global_B x D
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                all_student_outputs = student_output
                world_size = 1
                rank = 0

            # Group the global batch into groups of size `loss_group_size` and use the features of the group
            # the local rank falls into as the nearest neighbor set for the local rank
            local_B = len(student_output)
            global_B = len(all_student_outputs)
            loss_group_size = self.loss_group_size if self.loss_group_size is not None else global_B
            if loss_group_size % local_B != 0:
                raise ValueError(
                    f"Loss group size size {loss_group_size} must be a multiple of local batch size {local_B}."
                )
            if global_B % loss_group_size != 0:
                raise ValueError(
                    f"Global batch size {global_B} must be divisible by loss group size {loss_group_size}."
                )
            n_groups = global_B // loss_group_size
            ranks_per_group = world_size // n_groups
            rank_in_group = rank % ranks_per_group
            group = rank // ranks_per_group
            all_student_outputs = all_student_outputs.view(n_groups, loss_group_size, student_output.shape[1])
            all_student_outputs = all_student_outputs[group]  # loss_group_size x D

            with torch.no_grad():
                indices = self.pairwise_NNs_inner(student_output, all_student_outputs, rank_in_group)  # local_B x topk

            student_output_expanded = (
                student_output.unsqueeze(1).repeat(1, self.topk, 1).flatten(0, 1)
            )  # (local_B * topk) x D
            distances = self.pdist(student_output_expanded, all_student_outputs[indices].flatten(0, 1))  # BxD, BxD -> B
            loss = -torch.log(distances.float() + eps).mean()

        return loss
