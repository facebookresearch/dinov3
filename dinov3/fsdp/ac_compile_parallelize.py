# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import register_fsdp_forward_method
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from dinov3.utils import utils


logger = logging.getLogger("dinov3")


def get_activation_checkpoint_wrapper(cfg):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    if cfg.train.checkpointing_full:
        _checkpointing_wrapper = checkpoint_wrapper
        logger.info("using selective checkpointing on backbone with full checkpointing policy")
    else:
        _save_list = [
            # mm
            torch.ops.aten.mm.default,
            torch.ops.aten._scaled_mm.default,
            # attentions
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
        ]
        _checkpointing_wrapper = partial(
            checkpoint_wrapper,
            context_fn=partial(create_selective_checkpoint_contexts, _save_list),
            preserve_rng_state=True,
        )
        logger.info("using selective checkpointing on backbone with selective policy")
    return _checkpointing_wrapper


def activation_checkpoint_convnext(cfg, model: nn.Module):
    _checkpointing_wrapper = get_activation_checkpoint_wrapper(cfg)
    for stage_id, stage in enumerate(model.stages):
        for block_id, block in enumerate(stage):
            model.stages[stage_id][block_id] = _checkpointing_wrapper(block)
    for dsl_id, dsl in enumerate(model.downsample_layers):
        model.downsample_layers[dsl_id] = _checkpointing_wrapper(dsl)


def activation_checkpoint_transformer(cfg, model: nn.Module):
    _checkpointing_wrapper = get_activation_checkpoint_wrapper(cfg)
    for block_id, b in enumerate(model.blocks):
        model.blocks[block_id] = _checkpointing_wrapper(b)


def wrap_compile_block(module: nn.Module, use_cuda_graphs: bool, is_backbone_block: bool) -> nn.Module:
    if use_cuda_graphs and is_backbone_block:
        module.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": True})
    else:
        module.compile()
    return module


def compile_convnext(cfg, model: nn.Module):
    assert isinstance(model.stages, nn.ModuleList)
    # Compile at stage level
    for stage_id, stage in enumerate(model.stages):
        model.stages[stage_id] = wrap_compile_block(stage, cfg.train.cudagraphs, is_backbone_block=False)
    assert isinstance(model.downsample_layers, nn.ModuleList)
    for dsl_id, dsl in enumerate(model.downsample_layers):
        model.downsample_layers[dsl_id] = wrap_compile_block(dsl, cfg.train.cudagraphs, is_backbone_block=False)


def compile_transformer(cfg, model: nn.Module):
    assert isinstance(model.blocks, nn.ModuleList)
    for block_id, block in enumerate(model.blocks):
        model.blocks[block_id] = wrap_compile_block(block, cfg.train.cudagraphs, is_backbone_block=True)


def fsdp_convnext(fsdp_config: Dict[str, Any], model: nn.Module, reshard_after_forward: bool):
    stages = model.stages
    assert isinstance(stages, nn.ModuleList)
    # FSDP wrap at stage level
    for stage_id, stage in enumerate(stages):
        stages[stage_id] = fully_shard(stage, **fsdp_config, reshard_after_forward=reshard_after_forward)
    downsample_layers = model.downsample_layers
    assert isinstance(downsample_layers, nn.ModuleList)
    for dsl_id, dsl in enumerate(downsample_layers):
        downsample_layers[dsl_id] = fully_shard(dsl, **fsdp_config, reshard_after_forward=reshard_after_forward)
    dsl: FSDPState
    stage: FSDPState
    for dsl, stage in zip(downsample_layers, stages):
        dsl.set_modules_to_forward_prefetch([stage])
        stage.set_modules_to_backward_prefetch([dsl])
    fully_shard(model, **fsdp_config, reshard_after_forward=reshard_after_forward)
    register_fsdp_forward_method(model, "get_intermediate_layers")


def fsdp_transformer(fsdp_config: Dict[str, Any], model: nn.Module, reshard_after_forward: bool):
    # Backbone - FSDP every block
    blocks = model.blocks
    assert isinstance(blocks, nn.ModuleList)
    for block_id, block in enumerate(blocks):
        blocks[block_id] = fully_shard(block, **fsdp_config, reshard_after_forward=reshard_after_forward)
    prev_block: FSDPState
    next_block: FSDPState
    for prev_block, next_block in zip(blocks, blocks[1:]):
        prev_block.set_modules_to_forward_prefetch([next_block])
        next_block.set_modules_to_backward_prefetch([prev_block])
    fully_shard(model, **fsdp_config, reshard_after_forward=reshard_after_forward)
    register_fsdp_forward_method(model, "get_intermediate_layers")


def ac_compile_parallelize(
    trained_model: nn.ModuleDict,
    inference_only_models: List[nn.ModuleDict],
    cfg: Any,
    trained_model_process_group: dist.ProcessGroup | None = None,
    inference_only_models_process_groups: List[dist.ProcessGroup] | None = None,
) -> None:
    """
    Order of the wrappers:
    1/ Activation checkpointing on blocks
    2/ Compile blocks
    3/ FSDP blocks + global model
    """
    assert (
        isinstance(trained_model, nn.ModuleDict) and "backbone" in trained_model.keys()
    ), f"{trained_model} does not contain a backbone?"
    logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
    if utils.has_batchnorms(trained_model):
        raise NotImplementedError

    from dinov3.models.convnext import ConvNeXt
    from dinov3.models.vision_transformer import DinoVisionTransformer

    # FSDP utils for each architecture type
    ARCH_TYPE_MAP = {
        ConvNeXt: dict(
            compile_fn=compile_convnext,
            fsdp_fn=fsdp_convnext,
            activation_checkpointing_fn=activation_checkpoint_convnext,
        ),
        DinoVisionTransformer: dict(
            compile_fn=compile_transformer,
            fsdp_fn=fsdp_transformer,
            activation_checkpointing_fn=activation_checkpoint_transformer,
        ),
    }

    # 1/ AC on blocks
    if cfg.train.checkpointing:
        ARCH_TYPE_MAP[type(trained_model.backbone)]["activation_checkpointing_fn"](cfg, trained_model["backbone"])
    # 2/ Compile blocks
    all_models = [trained_model] + inference_only_models
    if trained_model_process_group is None and inference_only_models_process_groups is None:
        all_pgs = [None] * len(all_models)
    elif trained_model_process_group is None:
        all_pgs = [None] + inference_only_models_process_groups
    elif inference_only_models_process_groups is None:
        all_pgs = [trained_model_process_group] + [None] * len(inference_only_models_process_groups)
    else:
        all_pgs = [trained_model_process_group] + inference_only_models_process_groups
    if cfg.train.compile:
        for model in all_models:
            for k in model.keys():
                if k == "backbone":
                    ARCH_TYPE_MAP[type(model[k])]["compile_fn"](cfg, model[k])
                else:
                    model[k] = wrap_compile_block(model[k], use_cuda_graphs=False, is_backbone_block=False)
    DTYPE_MAP = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if not isinstance(cfg.train.fsdp_reshard_after_forward, bool):
        raise ValueError("`train.fsdp_reshard_after_forward` must be a boolean")
    reshard_after_forward = cfg.train.fsdp_reshard_after_forward
    mp_policy = MixedPrecisionPolicy(
        param_dtype=DTYPE_MAP[cfg.compute_precision.param_dtype],
        reduce_dtype=DTYPE_MAP[cfg.compute_precision.reduce_dtype],
    )
    for model, pg in zip(all_models, all_pgs):
        if pg is None:
            world_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dist.get_world_size(),),
                mesh_dim_names=("dp",),
            )
        else:
            world_mesh = DeviceMesh.from_group(pg, "cuda")
        fsdp_config = {"mesh": world_mesh, "mp_policy": mp_policy}
        for k in model.keys():
            if k == "backbone":
                ARCH_TYPE_MAP[type(model[k])]["fsdp_fn"](fsdp_config, model[k], reshard_after_forward)
            else:
                model[k] = fully_shard(model[k], **fsdp_config, reshard_after_forward=reshard_after_forward)

    # 4/ Move to `cuda` device
    for model in all_models:
        model.to_empty(device="cuda")

    # 5/ FSDP2: Intentionally always reshard immediately after forward for inference-only models.
    # This is independent of `train.fsdp_reshard_after_forward`: inference-only modules do not run
    # backward/optimizer steps, so we force immediate reshard to keep memory usage predictable.
    for model in inference_only_models:
        for k in model.keys():
            fsdp_state: FSDPState = model[k]._get_fsdp_state()
            if not fsdp_state._fsdp_param_group:
                continue
            mi = fsdp_state._fsdp_param_group.post_forward_mesh_info
            fsdp_state._lazy_init()
            fsdp_state._fsdp_param_group.post_forward_mesh_info = mi
