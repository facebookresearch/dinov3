# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import math
import os
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, List, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

import dinov3.distributed as distributed
from dinov3.logging import cleanup_logging, setup_logging
from dinov3.utils import fix_random_seeds, get_conda_env, get_sha

logger = logging.getLogger("dinov3")


def _warn_deprecated_sharding_strategy(*cfgs: DictConfig) -> None:
    found_deprecated_key = False
    for cfg in cfgs:
        # We check for 'compute_precision' block presence
        if not OmegaConf.is_config(cfg) or "compute_precision" not in cfg:
            continue
        # Then check for 'compute_precision.sharding_strategy'
        compute_precision_cfg = cfg.compute_precision
        if not OmegaConf.is_config(compute_precision_cfg) or "sharding_strategy" not in compute_precision_cfg:
            continue
        # Delete it
        del compute_precision_cfg["sharding_strategy"]
        found_deprecated_key = True
    if found_deprecated_key:
        logger.warning(
            "`compute_precision.sharding_strategy` is deprecated and ignored. "
            "Use `train.fsdp_reshard_after_forward` instead."
        )


@dataclass
class DinoV3SetupArgs:
    config_file: str
    pretrained_weights: str | None = None
    shard_unsharded_model: bool = False
    output_dir: str = ""
    opts: List[Any] = field(default_factory=lambda: [])

    def __post_init__(self):
        # When loaded from benchmark.yaml, self.opts is a frozen omegaconf.ListConfig,
        # which works everywhere except when we want to modify it or when
        # we try to json-serialize it. So we convert it to a regular list here.
        if OmegaConf.is_config(self.opts):
            self.opts = OmegaConf.to_object(self.opts)


def apply_scaling_rules_to_cfg(cfg):  # to fix
    assert distributed.is_enabled(), "Setup distributed to get global size !"
    if "schedules" in cfg:
        # For schedules v2, the scaling rules are applied when building the schedules, the config is not modified
        return cfg

    if cfg.optim.scaling_rule == "linear_wrt_256":
        old_lr = cfg.optim.lr
        cfg.optim.lr *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        logger.info(f"linear scaling learning rate; old: {old_lr}, new: {cfg.optim.lr}")
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        old_lr = cfg.optim.lr
        cfg.optim.lr *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; old: {old_lr}, new: {cfg.optim.lr}")
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    output_dir = os.path.abspath(output_dir)
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_default_config() -> DictConfig:
    p = pathlib.Path(__file__).parent / "ssl_default_config.yaml"
    return OmegaConf.load(p)


def get_cfg_from_args(args: DinoV3SetupArgs, multidistillation=False, strict=True):
    overrides = [*args.opts]
    if args.output_dir is not None:
        overrides.append(f"train.output_dir={os.path.realpath(args.output_dir)}")

    # Config file
    cfg = OmegaConf.load(args.config_file)

    # Command line overrides
    opts_cfg = OmegaConf.from_cli(overrides)
    _warn_deprecated_sharding_strategy(cfg, opts_cfg)

    if multidistillation:
        cfg = OmegaConf.merge(cfg, opts_cfg)
    else:
        # Default config
        default_cfg = get_default_config()
        if strict:
            OmegaConf.set_struct(default_cfg, True)
        cfg = OmegaConf.merge(default_cfg, cfg, opts_cfg)
    return cfg


def setup_config(args: DinoV3SetupArgs, strict_cfg=True):
    """
    Create configs and perform basic setups.
    """
    # Create the cfg with OmegaConf
    cfg = get_cfg_from_args(args, strict=strict_cfg)
    # setup distributed, logging, and random seeds
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    # dump config before modifying so it can be reloaded
    if args.output_dir is not None:
        write_config(cfg, args.output_dir)
    # modify the config inplace by applying scaling rules
    apply_scaling_rules_to_cfg(cfg)
    return cfg


def _enumerate_all_subgroup_ranks(all_subgroup_rank_spans: Sequence[Tuple[int, int]]):
    """Expands a specification of process subgroups from spans to enumerated ranks.

    Args:
       all_group_rank_spans: a sequence of rank spans (first rank, last rank),
           one for each process group. Example: ((0, 1), (2, 3), (4, 7)).
    """
    for first, last in all_subgroup_rank_spans:
        assert first <= last
    return tuple(tuple(range(first, last + 1)) for first, last in all_subgroup_rank_spans)


def setup_multidistillation(args: DinoV3SetupArgs):
    base_output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    # get config file for this rank
    base_cfg = OmegaConf.load(args.config_file)
    _warn_deprecated_sharding_strategy(base_cfg)

    assert base_cfg.multidistillation.enabled

    global_batch_size = base_cfg.multidistillation.global_batch_size

    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_rank()

    # build process subgroups
    all_subgroup_rank_spans = tuple(
        (student.ranks_range[0], student.ranks_range[1] - 1) for student in base_cfg.multidistillation.students
    )
    all_subgroup_ranks = _enumerate_all_subgroup_ranks(all_subgroup_rank_spans)
    distributed.new_subgroups(all_subgroup_ranks)

    found = False
    for student in base_cfg.multidistillation.students:
        if rank in range(*student.ranks_range):
            found = True
            break
    assert found, "rank of worker not in defined range"

    name = student.name
    config_path = student.config_path
    n_gpus = student.ranks_range[1] - student.ranks_range[0]
    assert global_batch_size % n_gpus == 0
    total_n_gpus = distributed.get_world_size()

    args.output_dir = os.path.join(base_output_dir, name)
    args.opts += [f"train.output_dir={args.output_dir}"]
    args.opts += [f"train.batch_size_per_gpu={global_batch_size // total_n_gpus}"]
    args.config_file = os.path.abspath(config_path)
    default_cfg = get_default_config()
    cfg = OmegaConf.load(args.config_file)
    opts_cfg = OmegaConf.from_cli(args.opts)
    _warn_deprecated_sharding_strategy(cfg, base_cfg, opts_cfg)

    cfg = OmegaConf.merge(default_cfg, cfg, base_cfg, opts_cfg)

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)

    fix_random_seeds(seed + rank)

    write_config(cfg, args.output_dir)
    apply_scaling_rules_to_cfg(cfg)

    return cfg


def setup_job(
    output_dir: Optional[str] = None,
    distributed_enabled: bool = True,
    logging_enabled: bool = True,
    seed: Optional[int] = 0,
    restrict_print_to_main_process: bool = True,
    distributed_timeout: timedelta | None = None,
):
    """
    Setup methods that should be done in every fairvit job
    Initializes logging, distributed, random seeds and other utilities.
    """
    if output_dir is not None:
        output_dir = os.path.realpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    if logging_enabled:
        setup_logging(
            output=output_dir,
            level=logging.INFO,
            log_to_stdout_only_in_main_process=restrict_print_to_main_process,
        )

    if distributed_enabled:
        distributed.enable(
            overwrite=True,
            nccl_async_error_handling=True,
            restrict_print_to_main_process=restrict_print_to_main_process,
            timeout=distributed_timeout,
        )

    if seed is not None:
        rank = distributed.get_rank()
        fix_random_seeds(seed + rank)

    logger = logging.getLogger("dinov3")
    logger.info("git:\n  {}\n".format(get_sha()))

    # Log some python info
    conda_env_name, conda_env_path = get_conda_env()
    logger.info(f"conda env name: {conda_env_name}")
    logger.info(f"conda env path: {conda_env_path}")
    logger.info(f"python path: {sys.path}")


def exit_job(distributed_enabled: bool = True, logging_enabled: bool = True):
    if distributed_enabled:
        distributed.disable()
    if logging_enabled:
        cleanup_logging()
