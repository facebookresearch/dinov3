# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import json
import logging
import os
import sys
from typing import Any, Dict

import torch
from omegaconf import OmegaConf

import dinov3.distributed as distributed
from dinov3.eval.depth.checkpoint_utils import find_latest_checkpoint
from dinov3.eval.depth.config import DepthConfig
from dinov3.eval.depth.eval import evaluate_depther_with_config
from dinov3.eval.depth.models import make_depther_from_config
from dinov3.eval.depth.train import train_model_with_backbone

from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from dinov3.eval.setup import load_model_and_context, resolve_device
from dinov3.run.init import job_context
from dinov3.hub.depthers import _get_depther_config, dinov3_vit7b16_dd

RESULTS_FILENAME = "results-depth.csv"
MAIN_METRICS = [".*_abs_rel", ".*_a1", ".*_rmse"]


logger = logging.getLogger("dinov3")


def _add_dataset_prefix_to_results(results_dict: Dict[str, float], dataset_name: str):
    final_dict = {dataset_name + "_" + k: v for k, v in results_dict.items()}
    return final_dict


def eval_depther_with_model(*, depther: torch.nn.Module, config: DepthConfig):
    if config.load_from is None:
        config.load_from = find_latest_checkpoint(config.output_dir)

    logger.info(f"Using config: \n {OmegaConf.to_yaml(config)}")
    results_dict, _, _ = evaluate_depther_with_config(
        config=config,
        depther=depther,
        device=distributed.get_rank(),
        reduce_results=False,
    )
    test_config_name = config.datasets.test.split(":", 1)[0]
    test_save_dir = os.path.join(config.output_dir, test_config_name)
    # reduce results
    if distributed.is_main_process():
        if not os.path.exists(test_save_dir):
            os.makedirs(test_save_dir)
        with open(os.path.join(test_save_dir, "results.json"), "w") as f:
            json.dump(results_dict, f, indent=4)
    for metric, values in results_dict.items():
        results_dict[metric] = float(torch.Tensor(values).nanmean())  # result can be NaN if ground truth is all masked
    summary = " \n====== Summary ======\n"
    summary += (
        f"{test_config_name:<10} "
        + " ".join([f"{metric}: {value:.3f}" for metric, value in results_dict.items()])
        + "\n"
    )
    results_dict = _add_dataset_prefix_to_results(results_dict, test_config_name)
    summary += "====================="
    logger.info(summary)
    return results_dict


def benchmark_launcher(eval_args: dict[str, Any]) -> dict[str, Any]:
    """Initialization of distributed and logging are preconditions for this method"""
    if "config" in eval_args:
        base_config_path = eval_args.pop("config")
        output_dir = eval_args["output_dir"]
        base_config = OmegaConf.load(base_config_path)
        structured_config = OmegaConf.structured(DepthConfig)
        depth_config: DepthConfig = OmegaConf.to_object(  # type: ignore
            OmegaConf.merge(
                structured_config,
                base_config,
                OmegaConf.create(eval_args),
            )
        )
    else:
        depth_config, output_dir = args_dict_to_dataclass(
            eval_args=eval_args, config_dataclass=DepthConfig, save_config=False
        )
    OmegaConf.save(config=depth_config, f=os.path.join(output_dir, "depth_config.yaml"))

    config_autocast_dtype = depth_config.model_dtype.autocast_dtype if depth_config.model_dtype is not None else None
    model_device = depth_config.model.device if depth_config.model is not None else None
    device = resolve_device(model_device if model_device is not None else depth_config.device)
    if depth_config.load_from == "dinov3_vit7b16_dd":
        with torch.device(device):
            autocast_dtype = config_autocast_dtype or torch.float32
            # override config parameters with those of the pretrained depther
            depther_config = _get_depther_config("dinov3_vit7b16")
            depth_config.decoder_head = OmegaConf.to_object(  # type: ignore
                OmegaConf.merge(
                    depth_config.decoder_head,
                    depther_config,
                )
            )

            depther = dinov3_vit7b16_dd(
                pretrained=True,
                autocast_dtype=autocast_dtype,
                device=device,
            )
    else:
        if depth_config.model is not None and depth_config.model.device is None:
            depth_config.model.device = str(device)
        with torch.device(device):
            assert depth_config.model is not None
            model, model_context = load_model_and_context(depth_config.model, output_dir=output_dir)
            autocast_dtype = config_autocast_dtype or model_context["autocast_dtype"]
            device = model_context["device"]

        if depth_config.load_from:
            depther = make_depther_from_config(
                backbone=model,
                config=depth_config.decoder_head,
                checkpoint_path=depth_config.load_from,
                autocast_dtype=autocast_dtype,
                device=device,
            )
            logger.info(f"Depth config:\n {OmegaConf.to_yaml(depth_config)}")
        else:
            # train backbone
            depther = train_model_with_backbone(depth_config, model, autocast_dtype)

    results_dict = eval_depther_with_model(depther=depther, config=depth_config)
    write_results(results_dict, output_dir, RESULTS_FILENAME)
    return results_dict


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    eval_args = cli_parser(argv)
    with job_context(output_dir=eval_args["output_dir"]):
        benchmark_launcher(eval_args=eval_args)
    return 0


if __name__ == "__main__":
    main()
