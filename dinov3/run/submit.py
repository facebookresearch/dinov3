# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import logging
import os
from pathlib import Path

from dinov3.logging import setup_logging
from dinov3.utils.cluster import (
    get_slurm_account,
    get_slurm_executor_parameters,
    get_slurm_partition,
    get_slurm_qos,
    get_user_checkpoint_path,
)
from dinov3.utils.custom_callable import load_custom_callable

logger = logging.getLogger("dinov3")


def get_submitit_parser():
    slurm_partition = get_slurm_partition()
    slurm_account = get_slurm_account()
    slurm_qos = get_slurm_qos()
    parser = argparse.ArgumentParser("Submitit arguments", add_help=False)
    parser.add_argument(
        "--ngpus",
        default=8,
        type=int,
        help="Number of gpus to request on each node, default: %(default)s",
    )
    parser.add_argument(
        "--nodes",
        default=1,
        type=int,
        help="Number of nodes to request, default: %(default)s",
    )
    parser.add_argument(
        "--timeout",
        default=2800,
        type=int,
        help="Duration of the job, default: %(default)s",
    )
    parser.add_argument(
        "--slurm-partition",
        default=slurm_partition,
        type=str,
        help="Partition where to submit, default: %(default)s",
    )
    parser.add_argument(
        "--slurm-qos",
        default=slurm_qos,
        metavar="SLURM_QOS",
        type=str,
        dest="slurm_qos",
        help="slurm QoS to use for jobs in cluster environment, default: %(default)s",
    )
    parser.add_argument(
        "--slurm-array-parallelism",
        default=256,
        type=int,
        help="Maximum number of jobs that will be executed in parallel, default: %(default)s",
    )
    parser.add_argument(
        "--slurm-nice",
        default=0,
        type=int,
        help="Adjusted scheduling priority within Slurm, default: %(default)s",
    )
    parser.add_argument(
        "--slurm-account",
        default=slurm_account,
        type=str,
        help="Slurm account name, default: %(default)s",
    )
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message, default: '%(default)s'",
    )
    parser.add_argument(
        "--exclude",
        default="",
        type=str,
        help="Nodes to exclude, default: '%(default)s'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="output dir",
    )
    return parser


def get_run_parser():
    parser = argparse.ArgumentParser("Launcher arguments", parents=[get_submitit_parser()])
    parser.add_argument(
        "module_path",
        type=str,
        help="Full path to the program/script to be launched in parallel, "
        "followed by all the arguments for the training script.",
    )
    parser.add_argument(
        "--callable-name",
        type=str,
        default="main",
        help="Name of the callable to execute in the script",
    )
    return parser


def get_shared_folder() -> Path:
    user_checkpoint_path = get_user_checkpoint_path()
    if user_checkpoint_path is None:
        raise RuntimeError("Path to user checkpoint cannot be determined")
    path = user_checkpoint_path / "experiments"
    path.mkdir(exist_ok=True)
    return path


class CheckpointableSubmitter:
    def __init__(self, module_path, callable_name, args, output_dir):
        self.args = args
        self.callable_name = callable_name
        self.module_path = os.path.realpath(module_path)
        self.output_dir = os.path.realpath(output_dir)

    def __call__(self):
        self._setup_args()
        callable_ = load_custom_callable(self.module_path, self.callable_name)
        callable_(self.args)

    def checkpoint(self):
        import submitit

        logger.info(f"Requeuing {self.callable_name} from {self.module_path} with {self.args}")
        empty_class = type(self)(self.module_path, self.callable_name, self.args, self.output_dir)
        return submitit.helpers.DelayedSubmission(empty_class)

    def _setup_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.output_dir = str(self.output_dir).replace("%j", str(job_env.job_id))
        if "--output-dir" not in self.args:
            self.args.insert(0, f"--output-dir={self.output_dir}")

        # Setup logging with exact same arguments as in fairvit/run/init.py
        # to use lru_cache memoization and avoid setting up the logger twice
        setup_logging(output=self.output_dir, level=logging.INFO)
        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        logger.info(f"Module Path: {self.module_path}")
        logger.info(f"Callable Name: {self.callable_name}")
        logger.info(f'Args: {" ".join(self.args)}')


def submit_jobs(class_to_submit, output_dir, submitit_args, name="fairvit"):
    import submitit

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=output_dir, slurm_max_num_timeout=30)

    kwargs = {}
    if submitit_args.comment:
        kwargs["slurm_comment"] = submitit_args.comment
    if submitit_args.exclude:
        kwargs["slurm_exclude"] = submitit_args.exclude

    executor_params = get_slurm_executor_parameters(
        nodes=submitit_args.nodes,
        num_gpus_per_node=submitit_args.ngpus,
        timeout_min=submitit_args.timeout,  # max is 60 * 72
        slurm_signal_delay_s=120,
        slurm_partition=submitit_args.slurm_partition,
        slurm_qos=submitit_args.slurm_qos,
        # slurm_account=submitit_args.slurm_account,
        slurm_additional_parameters=dict(nice=submitit_args.slurm_nice),
        **kwargs,
    )
    executor.update_parameters(name=name, **executor_params)
    job = executor.submit(class_to_submit)

    logger.info(f"Submitted job_id: {job.job_id}")
    str_output_dir = os.path.abspath(output_dir).replace("%j", str(job.job_id))
    logger.info(f"Logs and checkpoints will be saved at: {str_output_dir}")


def main():
    setup_logging(level=logging.INFO)
    args, script_args = get_run_parser().parse_known_args()
    assert os.path.exists(args.module_path), "The module path does not exist"

    file_name = os.path.splitext(os.path.split(args.module_path)[1])[0]
    name = f"{file_name}:{args.callable_name}"

    if args.output_dir is None:
        args.output_dir = get_shared_folder() / "%j"

    class_to_submit = CheckpointableSubmitter(args.module_path, args.callable_name, script_args, args.output_dir)
    submit_jobs(class_to_submit, args.output_dir, args, name=name)


if __name__ == "__main__":
    main()
