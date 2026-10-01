from enum import Enum
from functools import partial

import torch
from torch import nn


class LossType(Enum):
    SIGLOSS = "sigloss"
    GRADIENT_LOSS = "gradient_loss"
    GRADIENT_LOG_LOSS = "gradient_log_loss"
    L1 = "l1"

    def module(self, *args, **kwargs):
        return {
            LossType.SIGLOSS: partial(
                SigLoss,
                warm_up=True,
                warm_iter=100,
            ),  # default parameters for the custom loss (BW compatibility)
            LossType.GRADIENT_LOG_LOSS: GradientLogLoss,
            LossType.GRADIENT_LOSS: GradientLoss,
            LossType.L1: L1Loss,
        }[self](*args, **kwargs)


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 0.001

    def forward(self, input, target, valid_mask=None):
        input_downscaled = [input] + [input[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target] + [target[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        if valid_mask is not None:
            mask_downscaled = [valid_mask] + [valid_mask[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        else:
            mask_downscaled = [torch.ones_like(target, dtype=bool) for target in target_downscaled]

        gradient_loss = 0
        for input, target, mask in zip(input_downscaled, target_downscaled, mask_downscaled):
            N = torch.sum(mask)
            d_diff = torch.where(mask, input - target, 0)

            v_gradient = torch.abs(d_diff[..., 0:-2, :] - d_diff[..., 2:, :])
            v_mask = torch.mul(mask[..., 0:-2, :], mask[..., 2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(d_diff[..., :, 0:-2] - d_diff[..., :, 2:])
            h_mask = torch.mul(mask[..., :, 0:-2], mask[..., :, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)
            gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N.clamp_min(1)

        return gradient_loss


class GradientLogLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 0.001

    def forward(self, input, target, valid_mask=None):
        input_downscaled = [input] + [input[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target] + [target[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        if valid_mask is not None:
            mask_downscaled = [valid_mask] + [valid_mask[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        else:
            mask_downscaled = [torch.ones_like(target, dtype=bool) for target in target_downscaled]

        gradient_loss = 0
        for input, target, mask in zip(input_downscaled, target_downscaled, mask_downscaled):
            N = torch.sum(mask)
            # Invalid depths may be non-positive or non-finite. Mask before
            # logarithms, since multiplying NaN by zero does not remove it.
            input_log = torch.log(torch.where(mask, input, 0) + self.eps)
            target_log = torch.log(torch.where(mask, target, 0) + self.eps)
            log_d_diff = input_log - target_log

            log_d_diff = torch.mul(log_d_diff, mask)

            v_gradient = torch.abs(log_d_diff[..., 0:-2, :] - log_d_diff[..., 2:, :])
            v_mask = torch.mul(mask[..., 0:-2, :], mask[..., 2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(log_d_diff[..., :, 0:-2] - log_d_diff[..., :, 2:])
            h_mask = torch.mul(mask[..., :, 0:-2], mask[..., :, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)
            gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N.clamp_min(1)

        return gradient_loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, valid_mask=None):
        loss = nn.functional.l1_loss(input, target, reduce=False)
        mask = valid_mask if (valid_mask is not None) else torch.ones_like(input, dtype=bool)
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-7)


class SigLoss(nn.Module):
    """Sigloss

    Adapted from Binsformer who adapted from AdaBins
    https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/7c0c89c8db07631fec1737f3087e4f1f540ccd53/depth/models/losses/sigloss.py#L8
    """

    def __init__(self, warm_up=True, warm_iter=100):
        super(SigLoss, self).__init__()
        self.loss_name = "SigLoss"
        self.eps = 0.001  # avoid grad explode
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones_like(target, dtype=bool)
        input = input[valid_mask]
        target = target[valid_mask]

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = 0.15 * torch.pow(torch.mean(g), 2)
        if self.warm_up and self.warm_up_counter < self.warm_iter:
            self.warm_up_counter += 1
        else:
            Dg += torch.var(g)
        if Dg <= 0:
            return torch.abs(Dg)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt, valid_mask=None):
        """Forward function."""

        return self.sigloss(depth_pred, depth_gt, valid_mask)


class MultiLoss(nn.Module):
    """
    losses adapted from https://www.cs.cornell.edu/projects/megadepth/

    Args:
        dict_losses: (dict[LossType, float, Any]) a dict of losses in the format {LossType_1: Weight_1, ..., LossType_N: Weight_N}.
    """

    def __init__(
        self,
        dict_losses: dict[LossType, float],
    ):
        super(MultiLoss, self).__init__()
        self.dict_losses = nn.ModuleDict({loss_type.name: loss_type.module() for loss_type in dict_losses.keys()})
        self.dict_weights = {loss_type.name: weight for (loss_type, weight) in dict_losses.items()}
        self.eps = 0.001  # avoid grad explode

    def forward(self, depth_pred, depth_gt, valid_mask=None):
        """Forward function."""

        loss_depth = 0
        for loss_name in self.dict_losses.keys():
            weight = self.dict_weights[loss_name]
            loss = self.dict_losses[loss_name](depth_pred, depth_gt, valid_mask)
            loss_depth += weight * loss
        return loss_depth
