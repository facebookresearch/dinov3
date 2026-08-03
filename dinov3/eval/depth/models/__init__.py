# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
from dinov3.eval.depth.checkpoint_utils import load_checkpoint
from dinov3.eval.depth.utils import create_chmv2_mixlog_bins, create_outputs_with_chmv2_mixlog_norm
from dinov3.eval.setup import get_autocast_device_type, resolve_device

from .dpt_head import DPTHead
from .linear_head import LinearHead
from .encoder import BackboneLayersSet, DinoVisionTransformerWrapper, PatchSizeAdaptationStrategy


@dataclass
class DecoderConfig:
    min_depth: float = 0.001
    max_depth: float = 80
    bins_strategy: str = "linear"  # (choice: ["linear", "log", "chmv2_mixlog"]) distribution of bins across the range
    norm_strategy: str = "linear"  # (choice: ["linear", "softmax", "sigmoid", "chmv2_mixlog"]) activation used before normalization of depth-bin logits
    head_kwargs: dict[str, Any] = field(default_factory=dict)
    backbone_out_layers: Any = (
        BackboneLayersSet.FOUR_EVEN_INTERVALS  # One of BackboneLayersSet(Enum) or a list of indices e.g. [0, 1, 2, 3]
    )
    adapt_to_patch_size: PatchSizeAdaptationStrategy = PatchSizeAdaptationStrategy.CENTER_PADDING
    use_backbone_norm: bool = True
    # decoder
    type: str = "linear"  # choices: linear or dpt
    n_output_channels: int = 256
    use_batchnorm: bool = False
    use_cls_token: bool = False


class FeaturesToDepth(torch.nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=80,
        bins_strategy="linear",
        norm_strategy="linear",
    ):
        """
        Module which converts a feature maps into a depth map

        Args:
        min_depth (float): minimum depth, used to calibrate the depth range
        max_depth (float): maximum depth, used to calibrate the depth range
        bins_strategy (str): Choices are 'linear', 'log', or 'chmv2_mixlog'. The first two are for Uniform or Scale Invariant distributions
                             for depth bins. See AdaBins [1] for more details.
                             "chmv2_mixlog" uses customized mixed log bins for the CHMv2 head.
        norm_strategy (str): Choices are 'linear', 'softmax', 'sigmoid' or 'chmv2_mixlog', for the conversion of features to depth logits

        Example:
        x = depth_model(input_image)  # N C H W
        - If pure regression (C == 1), depth is obtained by scaling and/or shifting x
        - If C > 1, bins are used:
            Depth is obtained as a weighted sum of depth bins, where weights are predicted logits. (see AdaBins [1] for more details)

        [1] AdaBins: https://github.com/shariqfarooq123/AdaBins
        """
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        assert bins_strategy in ["linear", "log", "chmv2_mixlog"], "Support bins_strategy: linear, log, chmv2_mixlog"
        assert norm_strategy in ["linear", "softmax", "sigmoid", "chmv2_mixlog"], (
            "Support norm_strategy: linear, softmax, sigmoid, chmv2_mixlog"
        )

        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy

    def forward(self, x):
        n_bins = x.shape[1]  # N n_bins H W
        if n_bins > 1:
            if self.bins_strategy == "linear":
                bins = torch.linspace(self.min_depth, self.max_depth, n_bins, device=x.device)
            elif self.bins_strategy == "log":
                bins = torch.linspace(
                    torch.log(torch.tensor(self.min_depth)),
                    torch.log(torch.tensor(self.max_depth)),
                    n_bins,
                    device=x.device,
                )
                bins = torch.exp(bins)
            else:  # bins_strategy = "chmv2_mixlog"
                bins = create_chmv2_mixlog_bins(self.min_depth, self.max_depth, n_bins, x.device)

            # following Adabins, default is linear
            if self.norm_strategy in ["linear", "softmax", "sigmoid"]:
                if self.norm_strategy == "linear":
                    logit = torch.relu(x)
                    eps = 0.1
                    logit = logit + eps
                    logit = logit / logit.sum(dim=1, keepdim=True)
                elif self.norm_strategy == "softmax":
                    logit = torch.softmax(x, dim=1)
                else:  # norm_strategy = "sigmoid"
                    logit = torch.sigmoid(x)
                    logit = logit / logit.sum(dim=1, keepdim=True)
                output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)

            else:  # norm_strategy = "chmv2_mixlog"
                output = create_outputs_with_chmv2_mixlog_norm(x, bins)
        else:
            # standard regression
            output = torch.relu(x) + self.min_depth
        return output


def make_head(
    embed_dims: int | list[int],
    n_output_channels: int,
    use_batchnorm: bool = False,
    use_cls_token: bool = False,
    head_type: str = "linear",
    **kwargs,
) -> torch.nn.Module:
    if isinstance(embed_dims, int):
        embed_dims = [embed_dims]
    decoder: torch.nn.Module
    if head_type == "linear":
        decoder = LinearHead(
            in_channels=embed_dims,
            n_output_channels=n_output_channels,
            use_batchnorm=use_batchnorm,
            use_cls_token=use_cls_token,
        )
    elif head_type == "dpt":
        decoder = DPTHead(
            in_channels=embed_dims,
            n_output_channels=n_output_channels,
            readout_type="project" if use_cls_token else "ignore",
            use_batchnorm=use_batchnorm,
            **kwargs,
        )
    else:
        raise NotImplementedError("only linear and DPT head supported")
    return decoder


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Depther(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        min_depth: float,
        max_depth: float,
        bins_strategy: str = "linear",
        norm_strategy: str = "linear",
        autocast_dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.features_to_depth = FeaturesToDepth(
            min_depth=min_depth,
            max_depth=max_depth,
            bins_strategy=bins_strategy,
            norm_strategy=norm_strategy,
        )
        device = resolve_device(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.autocast_ctx = partial(
            torch.autocast,
            device_type=get_autocast_device_type(device),
            dtype=autocast_dtype,
            enabled=True,
        )

    def forward(self, x):
        with self.autocast_ctx():
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.features_to_depth(x)
        return x


def build_depther(
    backbone: torch.nn.Module,
    backbone_out_layers: tuple[int, ...] | BackboneLayersSet,
    n_output_channels: int,
    use_backbone_norm: bool = False,
    use_batchnorm: bool = False,
    use_cls_token: bool = False,
    adapt_to_patch_size: PatchSizeAdaptationStrategy = PatchSizeAdaptationStrategy.CENTER_PADDING,
    head_type: str = "dpt",
    autocast_dtype: torch.dtype = torch.float32,
    device=None,
    # depth args
    min_depth: float = 0.001,
    max_depth: float = 10.0,
    bins_strategy: str = "linear",
    norm_strategy: str = "linear",
    **kwargs,
):
    encoder = DinoVisionTransformerWrapper(
        backbone_model=backbone,
        backbone_out_layers=backbone_out_layers,
        use_backbone_norm=use_backbone_norm,
        adapt_to_patch_size=adapt_to_patch_size,
    )

    decoder = make_head(
        encoder.embed_dims,
        n_output_channels=n_output_channels,
        use_batchnorm=use_batchnorm,
        use_cls_token=use_cls_token,
        head_type=head_type,
        **kwargs,
    )

    depther = Depther(
        encoder=encoder,
        decoder=decoder,
        min_depth=min_depth,
        max_depth=max_depth,
        bins_strategy=bins_strategy,
        norm_strategy=norm_strategy,
        autocast_dtype=autocast_dtype,
        device=device,
    )
    depther.eval()
    return depther


def make_depther_from_config(
    backbone,
    config: DecoderConfig,
    checkpoint_path: str | None = None,
    autocast_dtype: torch.dtype = torch.float32,
    device=None,
) -> Depther:
    depther = build_depther(
        backbone,
        backbone_out_layers=config.backbone_out_layers,
        n_output_channels=config.n_output_channels,
        use_backbone_norm=config.use_backbone_norm,
        use_batchnorm=config.use_batchnorm,
        use_cls_token=config.use_cls_token,
        adapt_to_patch_size=config.adapt_to_patch_size,
        head_type=config.type,
        autocast_dtype=autocast_dtype,
        device=device,
        min_depth=config.min_depth,
        max_depth=config.max_depth,
        bins_strategy=config.bins_strategy,
        norm_strategy=config.norm_strategy,
        **config.head_kwargs,
    )

    # resume from checkpoint
    if checkpoint_path is not None:
        state_dicts, _ = load_checkpoint(checkpoint_path)
        # load head state dict, the backbone has its pretrained_weights
        depther.decoder.load_state_dict(state_dicts["model"], strict=True)

    return depther
