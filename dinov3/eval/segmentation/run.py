# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from omegaconf import OmegaConf
import os
import sys
from typing import Any

from dinov3.eval.segmentation.config import SegmentationConfig
from dinov3.eval.segmentation.eval import test_segmentation
from dinov3.eval.segmentation.train import train_segmentation
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from dinov3.eval.setup import load_model_and_context
from dinov3.run.init import job_context


logger = logging.getLogger("dinov3")

RESULTS_FILENAME = "results-semantic-segmentation.csv"
MAIN_METRICS = ["mIoU"]


def run_segmentation_with_dinov3(
    backbone,
    config,
):
    if config.load_from:
        logger.info("Testing model performance on a pretrained decoder head")
        return test_segmentation(backbone=backbone, config=config)
    assert config.decoder_head.type == "linear", "Only linear head is supported for training"
    return train_segmentation(backbone=backbone, config=config)


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    """Initialization of distributed and logging are preconditions for this method"""
    if "config" in eval_args:  # using a config yaml file, useful for training
        base_config_path = eval_args.pop("config")
        output_dir = eval_args["output_dir"]
        base_config = OmegaConf.load(base_config_path)
        structured_config = OmegaConf.structured(SegmentationConfig)
        dataclass_config: SegmentationConfig = OmegaConf.to_object(
            OmegaConf.merge(
                structured_config,
                base_config,
                OmegaConf.create(eval_args),
            )
        )
    else:  # either using default values, or only adding some args to the command line
        dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=SegmentationConfig)
    backbone = None
    if dataclass_config.model:
        if dataclass_config.model.device is None and dataclass_config.device is not None:
            dataclass_config.model.device = dataclass_config.device
        backbone, _ = load_model_and_context(dataclass_config.model, output_dir=output_dir)
    else:
        assert dataclass_config.load_from == "dinov3_vit7b16_ms"
    logger.info(f"Segmentation Config:\n{OmegaConf.to_yaml(dataclass_config)}")
    segmentation_file_path = os.path.join(output_dir, "segmentation_config.yaml")
    OmegaConf.save(config=dataclass_config, f=segmentation_file_path)
    results_dict = run_segmentation_with_dinov3(backbone=backbone, config=dataclass_config)
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
    sys.exit(main())
