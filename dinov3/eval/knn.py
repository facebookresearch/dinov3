# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
from omegaconf import MISSING
from torch.nn.functional import one_hot, softmax

import dinov3.distributed as distributed
from dinov3.data import SamplerType, make_data_loader, make_dataset
from dinov3.data.adapters import DatasetWithEnumeratedTargets
from dinov3.data.transforms import (
    CROP_DEFAULT_SIZE,
    RESIZE_DEFAULT_SIZE,
    get_target_transform,
    make_classification_eval_transform,
)
from dinov3.distributed import gather_all_tensors
from dinov3.eval.data import (
    create_train_dataset_dict,
    extract_features_for_dataset_dict,
    get_num_classes,
    pad_multilabel_and_collate,
)
from dinov3.eval.helpers import args_dict_to_dataclass, cli_parser, write_results
from dinov3.eval.metrics import ClassificationMetricType, build_classification_metric
from dinov3.eval.setup import ModelConfig, get_autocast_device_type, load_model_and_context, resolve_device
from dinov3.eval.utils import ModelWithNormalize, average_metrics, evaluate
from dinov3.eval.utils import save_results as default_save_results_func
from dinov3.run.init import job_context

logger = logging.getLogger("dinov3")


RESULTS_FILENAME = "results-knn.csv"
MAIN_METRICS = [".* Top 1"]


@dataclass
class TrainConfig:
    dataset: str = MISSING  # train dataset path
    batch_size: int = 256  # batch size for train set feature extraction
    num_workers: int = 5  # number of workers for train set feature extraction
    ks: Tuple[int, ...] = (10, 20, 100, 200)  # values of k to evaluate
    temperature: float = 0.07
    """
    Whether to skip the first nearest neighbor for each image in the test set.
    Useful when training and testing on the same dataset split.
    """
    skip_first_nn: bool = False


@dataclass
class EvalConfig:
    test_dataset: str = MISSING  # test dataset path
    test_metric_type: ClassificationMetricType = ClassificationMetricType.MEAN_ACCURACY
    batch_size: int | None = None  # batch size for evaluation, None to use train batch size
    num_workers: int = 5  # number of workers for evaluation


@dataclass
class TransformConfig:
    resize_size: int = RESIZE_DEFAULT_SIZE
    crop_size: int = CROP_DEFAULT_SIZE


@dataclass
class FewShotConfig:
    enable: bool = False  # whether to use few-shot evaluation
    k_or_percent: Optional[float] = None  # number of elements or % to take per class
    n_tries: int = 1  # number of tries for few-shot evaluation


@dataclass
class KnnEvalConfig:
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    save_results: bool = False  # save predictions and targets in the output directory
    output_dir: str = ""


class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(self, *, train_features, train_labels, device, ks, T, num_classes=1000, skip_first_nn=False):
        super().__init__()

        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.device = device
        self.train_features_rank_T = train_features.chunk(self.world_size)[self.rank].T.to(self.device)
        # Labels can either be integers, or in a one-hot format
        self.candidates = train_labels.chunk(self.world_size)[self.rank].unsqueeze(0).to(self.device)

        self.ks = ks
        self.max_k = max(self.ks) + skip_first_nn
        self.T = T
        self.num_classes = num_classes
        self.skip_first_nn = skip_first_nn

        if self.skip_first_nn:
            logger.info("Skipping the first nearest neighbor of each element in the test dataset")

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(min(self.max_k, similarity.shape[1]), largest=True, sorted=True)
        if len(train_labels.shape) == 3:  # If the labels are in one_hot format
            indices = indices.unsqueeze(2).expand(-1, -1, self.num_classes)  # Orignally [bs, max_k]
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        """
        Broadcasts `features_rank` from `source_rank` and compute similarities
        with the train features chunks from all ranks
        """
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        torch.distributed.broadcast(broadcast_shape, source_rank)

        broadcasted = features_rank
        if self.rank != source_rank:
            broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype, device=self.device)
        torch.distributed.broadcast(broadcasted, source_rank)

        # Compute the neighbors for `source_rank` among `train_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), *self.candidates.shape[1:])
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def compute_neighbors(self, features_rank):
        """
        If we are on rank `rank`, we broadcast the test features to other ranks, compute similarities
        with their chunks of the train features, then gather these partial similarities back on `rank`
        """
        topk_sims_rank, neighbors_labels_rank = None, None
        for rank in range(self.world_size):
            partial_topk_sims, partial_neighbors_labels = self._similarity_for_rank(features_rank, rank)
            gathered_topk_sims = torch.cat(gather_all_tensors(partial_topk_sims), dim=1)
            gathered_neighbor_labels = torch.cat(gather_all_tensors(partial_neighbors_labels), dim=1)
            if self.rank == rank:  # Performing a second top-k to get k neighbors from the gathered k * world_size
                topk_sims_rank, neighbors_labels_rank = self._get_knn_sims_and_labels(
                    gathered_topk_sims, gathered_neighbor_labels
                )
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.ks` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.ks)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        if self.skip_first_nn:
            topk_sims, neighbors_labels = topk_sims[:, 1:], neighbors_labels[:, 1:]
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        voting_coefficient = topk_sims_transform.view(batch_size, -1, 1)
        if len(neighbors_labels.shape) == 2:  # If the labels are not yet one hot
            neighbors_labels = one_hot(neighbors_labels, num_classes=self.num_classes)
        matmul = torch.mul(neighbors_labels, voting_coefficient)
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.ks}
        return probas_for_k


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        return {"preds": features_dict, "target": targets}


def make_transform(config: TransformConfig):
    if config.resize_size / config.crop_size != 256 / 224:
        logger.warning(
            f"Default resize / crop ratio is 256 / 224, here we have {config.resize_size} / {config.crop_size}"
        )
    transform = make_classification_eval_transform(resize_size=config.resize_size, crop_size=config.crop_size)
    return transform


def make_test_data_loader(config: EvalConfig, transform):
    # Create test data loader. Do not extract features in advance due to difficulties with multilabel datasets.
    multilabel_collate_fn = config.test_metric_type == ClassificationMetricType.ANY_MATCH_ACCURACY
    test_dataset = make_dataset(
        dataset_str=config.test_dataset,
        transform=transform,
        target_transform=get_target_transform(config.test_dataset),
    )
    assert isinstance(config.batch_size, int)  # eval batch size has been replaced by train batch size if None

    return make_data_loader(
        dataset=DatasetWithEnumeratedTargets(test_dataset, pad_dataset=True, num_replicas=distributed.get_world_size()),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
        collate_fn=pad_multilabel_and_collate if multilabel_collate_fn else None,
    )


def eval_knn(
    *,
    model,
    train_data_dict,
    test_data_loader,
    metric_collection,
    knn_config: TrainConfig,
    num_classes: int,
    save_results_func=None,
    device=None,
):
    logger.info("Start the k-NN classification.")
    eval_metrics_dict: Dict[int, Dict[int, Dict[str, float]]] = {}  # {k: {try: {metric_name: metric_value}}}
    save_results = save_results_func is not None
    device = resolve_device(device)
    partial_knn_module = partial(
        KnnModule,
        device=device,
        num_classes=num_classes,
        T=knn_config.temperature,
        skip_first_nn=knn_config.skip_first_nn,
    )

    for try_ in train_data_dict.keys():
        train_features, train_labels = train_data_dict[try_]["train_features"], train_data_dict[try_]["train_labels"]
        ks = sorted(set([el if el < len(train_features) else len(train_features) for el in knn_config.ks]))
        knn_module = partial_knn_module(train_features=train_features, train_labels=train_labels, ks=ks)
        postprocessors, metrics = {k: DictKeysModule([k]) for k in ks}, {k: metric_collection.clone() for k in ks}
        _, eval_metrics, accumulated_results = evaluate(
            torch.nn.Sequential(model, knn_module),
            test_data_loader,
            postprocessors,
            metrics,
            device,
            accumulate_results=save_results,
        )
        for k in ks:
            if save_results:
                if len(train_data_dict) > 1:
                    split_results_saver = partial(save_results_func, filename_suffix=f"try_{try_}_k_{k}")
                else:
                    split_results_saver = partial(save_results_func, filename_suffix=f"k_{k}")
                split_results_saver(**accumulated_results[k])

            if k not in eval_metrics_dict:
                eval_metrics_dict[k] = {}
            eval_metrics_dict[k][try_] = {metric: v.item() * 100.0 for metric, v in eval_metrics[k].items()}

    if len(train_data_dict) > 1:
        return {k: average_metrics(eval_metrics_dict[k]) for k in eval_metrics_dict.keys()}

    return {k: eval_metrics_dict[k][0] for k in eval_metrics_dict.keys()}


def _log_and_format_results_dict(input_results_dict, few_shot_n_tries: int) -> Dict[str, float]:
    results_dict = {}
    for knn_ in input_results_dict.keys():
        if few_shot_n_tries == 1:
            top1 = input_results_dict[knn_]["top-1"]
            results_dict[f"{knn_} Top 1"] = top1
            results_string = f"{knn_} NN classifier result: Top1: {top1:.2f}"
            if "top-5" in input_results_dict[knn_]:
                top5 = input_results_dict[knn_]["top-5"]
                results_dict[f"{knn_} Top 5"] = top5
                results_string += f" Top5: {top5:.2f}"
        else:
            top1_mean, top1_std = input_results_dict[knn_]["top-1_mean"], input_results_dict[knn_]["top-1_std"]
            results_dict[f"{knn_} Top 1"] = top1_mean
            results_string = f"{knn_} NN classifier result: Top1 Avg: {top1_mean:.2f}, Top1 Std {top1_std:.2f}"
            if "top-5_mean" in input_results_dict[knn_]:
                top5_mean, top5_std = input_results_dict[knn_]["top-5_mean"], input_results_dict[knn_]["top-5_std"]
                results_dict[f"{knn_} Top 5"] = top5_mean
                results_string += f" Top5 Avg: {top5_mean:.2f}, Top5 Std {top5_std:.2f}"
        logger.info(results_string)
    return results_dict


def eval_knn_with_model(*, model: torch.nn.Module, autocast_dtype, config: KnnEvalConfig, device=None):
    start = time.time()
    cudnn.benchmark = True
    device = resolve_device(device)

    # Setting up datasets
    transform = make_transform(config.transform)
    train_dataset = make_dataset(
        dataset_str=config.train.dataset,
        transform=transform,
        target_transform=get_target_transform(config.train.dataset),
    )
    train_dataset_dict = create_train_dataset_dict(
        train_dataset,
        few_shot_eval=config.few_shot.enable,
        few_shot_k_or_percent=config.few_shot.k_or_percent,
        few_shot_n_tries=config.few_shot.n_tries,
    )

    # Setting up metrics
    num_classes = get_num_classes(train_dataset)
    metric_collection = build_classification_metric(config.eval.test_metric_type, num_classes=num_classes)
    config.eval.batch_size = config.eval.batch_size or config.train.batch_size
    test_data_loader = make_test_data_loader(config.eval, transform)

    # Setting up save results function
    save_results_func = None
    if config.save_results:
        save_results_func = partial(default_save_results_func, output_dir=config.output_dir)

    model = ModelWithNormalize(model)
    with torch.autocast(get_autocast_device_type(device), dtype=autocast_dtype):
        logger.info("Extracting features for train set...")
        train_data_dict = extract_features_for_dataset_dict(
            model,
            train_dataset_dict,
            config.train.batch_size,
            config.train.num_workers,
            gather_on_cpu=True,
            device=device,
        )
        results_dict_knn = eval_knn(
            model=model,
            train_data_dict=train_data_dict,
            test_data_loader=test_data_loader,
            metric_collection=metric_collection,
            knn_config=config.train,
            num_classes=num_classes,
            save_results_func=save_results_func,
            device=device,
        )
    results_dict = _log_and_format_results_dict(results_dict_knn, config.few_shot.n_tries)

    # TODO: Remove as cleaner writers are used
    metrics_file_path = os.path.join(config.output_dir, "results_eval_knn.json")
    with open(metrics_file_path, "a") as f:
        for k, v in results_dict.items():
            f.write(json.dumps({k: v}) + "\n")

    if distributed.is_enabled():
        torch.distributed.barrier()
    logger.info(f"Knn evaluation done in {int(time.time() - start)}s")
    return results_dict


def benchmark_launcher(eval_args: dict[str, object]) -> dict[str, Any]:
    """Initialization of distributed and logging are preconditions for this method"""
    dataclass_config, output_dir = args_dict_to_dataclass(eval_args=eval_args, config_dataclass=KnnEvalConfig)
    model, model_context = load_model_and_context(dataclass_config.model, output_dir=output_dir)
    results_dict = eval_knn_with_model(
        model=model,
        config=dataclass_config,
        autocast_dtype=model_context["autocast_dtype"],
        device=model_context["device"],
    )
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
