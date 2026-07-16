# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from pathlib import Path

from typing import Union

import torch
import torch.nn as nn

from dinov3.layers.fp8_linear import convert_linears_to_fp8

from . import vision_transformer as vits
from . import convnext

logger = logging.getLogger("dinov3")


def init_fp8(model: nn.Module, args) -> nn.Module:
    if not args.fp8_enabled:
        logger.info("fp8 matmuls: OFF (disabled in config)")
        return model
    logger.info("fp8 matmuls: ON")
    # Multi-kernel makes Inductor auto-tune between a regular "streaming"-based
    # reduction kernel and a "persistent" reduction kernel. Since fp8 has some
    # multi-pass steps (e.g., first get amax, then scale), persistent kernels
    # should perform better.
    torch._inductor.config.triton.multi_kernel = 1
    return convert_linears_to_fp8(model, filter=args.fp8_filter)


def build_model(args, only_teacher=False, img_size=224, device=None):
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            pos_embed_rope_base=args.pos_embed_rope_base,
            pos_embed_rope_min_period=args.pos_embed_rope_min_period,
            pos_embed_rope_max_period=args.pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=args.pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=args.pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=args.pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=args.pos_embed_rope_rescale_coords,
            qkv_bias=args.qkv_bias,
            layerscale_init=args.layerscale,
            norm_layer=args.norm_layer,
            ffn_layer=args.ffn_layer,
            ffn_bias=args.ffn_bias,
            proj_bias=args.proj_bias,
            n_storage_tokens=args.n_storage_tokens,
            mask_k_bias=args.mask_k_bias,
            untie_cls_and_patch_norms=args.untie_cls_and_patch_norms,
            untie_global_and_local_cls_norm=args.untie_global_and_local_cls_norm,
            device=device,
            in_chans=args.in_chans
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        teacher = init_fp8(teacher, args)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
        )
        embed_dim = student.embed_dim
    elif "convnext" in args.arch:
        convnext_cls = convnext.get_convnext_arch(args.arch)
        convnext_kwargs = dict(patch_size=args.patch_size)
        teacher = convnext_cls(**convnext_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = convnext_cls(**convnext_kwargs)
        embed_dim = student.embed_dim
    else:
        raise NotImplementedError(f"Unrecognized architecture {args.arch}")
    student = init_fp8(student, args)
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher: bool = False):
    outputs = build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size
        if isinstance(cfg.crops.global_crops_size, int)
        else max(cfg.crops.global_crops_size),
        device="meta",
    )
    if only_teacher:
        teacher, embed_dim = outputs
        return teacher, embed_dim
    else:
        student, teacher, embed_dim = outputs
        return student, teacher, embed_dim


def build_model_for_eval(
    config,
    pretrained_weights: Union[str, Path] | None,
    shard_unsharded_model: bool = False,  # If the model is not sharded, shard it. No effect if already sharded on disk
):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if pretrained_weights is None or pretrained_weights == "":
        logger.info("No pretrained weights")
        model.init_weights()
    elif Path(pretrained_weights).is_dir():
        logger.info("PyTorch DCP checkpoint")
        from dinov3.checkpointer import load_checkpoint
        from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize

        moduledict = nn.ModuleDict({"backbone": model})
        # Wrap with FSDP
        ac_compile_parallelize(moduledict, inference_only_models=[], cfg=config)
        # Move to CUDA
        model.to_empty(device="cuda")
        # Load checkpoint
        load_checkpoint(pretrained_weights, model=moduledict, strict_loading=True)
        shard_unsharded_model = False
    else:
        logger.info("PyTorch consolidated checkpoint")
        from dinov3.checkpointer import init_model_from_checkpoint_for_evals

        # consolidated checkpoint codepath
        model.to_empty(device="cuda")
        init_model_from_checkpoint_for_evals(model, pretrained_weights, "teacher")
    if shard_unsharded_model:
        logger.info("Sharding model")
        moduledict = nn.ModuleDict({"backbone": model})
        ac_compile_parallelize(moduledict, inference_only_models=[], cfg=config)
    model.eval()
    return model
