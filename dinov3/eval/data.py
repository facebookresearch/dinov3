# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import lru_cache
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets.vision import StandardTransform

from dinov3.eval.utils import extract_features

logger = logging.getLogger("dinov3")


class SubsetEx(Subset):
    def _get_actual_index(self, index: int) -> int:
        return self.indices[index]

    def get_target(self, index: int) -> Any:
        actual_index = self._get_actual_index(index)
        return self.dataset.get_target(actual_index)

    @property
    def transforms(self):
        return self.dataset.transforms


def get_target_transform(dataset) -> Optional[Callable]:
    if hasattr(dataset, "transforms"):
        if isinstance(dataset.transforms, StandardTransform):
            return dataset.transforms.target_transform
        raise ValueError("Dataset has a non-standard .transforms property")
    if hasattr(dataset, "target_transform"):
        return dataset.target_transform
    return None


@lru_cache(maxsize=1)
def get_labels(dataset) -> torch.Tensor:
    """
    Get the labels of a classification dataset, as a Tensor, using the `get_targets` method
    if it is present or loading the labels one by one with `get_target`, if it exists.
    If the dataset has a target transform, iterate over the whole dataset to get the
    transformed labels for each element, then stack them as a torch tensor.
    """
    logger.info("Getting dataset labels ...")
    if hasattr(dataset, "get_targets") or hasattr(dataset, "get_target"):
        if hasattr(dataset, "get_targets"):  # Returns a np.array
            labels = dataset.get_targets()
        elif hasattr(dataset, "get_target"):
            labels = [dataset.get_target(i) for i in range(len(dataset))]
        target_transform = get_target_transform(dataset)
        if target_transform is not None:
            labels = [target_transform(label) for label in labels]
    else:
        # Target transform is applied in this case
        labels = [dataset[i][1] for i in range(len(dataset))]
    return torch.stack([torch.tensor(label, dtype=int) for label in labels])


def get_num_classes(dataset) -> int:
    """
    Get the labels of a dataset and compute the number of classes
    """
    labels = get_labels(dataset)
    if len(labels.shape) > 1:
        return int(labels.shape[1])
    return int(labels.max() + 1)


def create_class_indices_mapping(labels: torch.Tensor) -> dict[int, torch.Tensor]:
    """
    Efficiently creates a mapping between the labels and tensors containing
    the indices of all the dataset elements that share this label.
    In the case of multiple labels, it is not guaranteed that there
    will be exactly the specified percentage of labels.
    """
    if len(labels.shape) > 1:  # labels are a one-hot encoding
        assert len(labels.shape) == 2
        sorted_labels, indices = torch.nonzero(labels.T, as_tuple=True)
    else:
        sorted_labels, indices = torch.sort(labels, stable=True)
    unique_labels, counts = torch.unique_consecutive(sorted_labels, return_counts=True)
    mapping = dict(zip(unique_labels.tolist(), torch.split(indices, counts.tolist())))
    return mapping


def _shuffle_dataset(dataset: torch.Tensor, seed: int = 0):
    """
    Shuffling a dataset by subsetting it with a random permutation of its indices
    """
    random_generator = torch.Generator()
    random_generator.manual_seed(seed)
    random_indices = torch.randperm(len(dataset), generator=random_generator)
    return SubsetEx(dataset, random_indices)


def _subset_dataset_per_class(
    class_indices_mapping: dict[int, torch.Tensor],
    n_or_percent_per_class: int | float,
    dataset_size: int,
    seed: int = 0,
    is_percent: bool = False,
) -> torch.Tensor:
    """
    Helper function to select a percentage of a dataset, equally distributed across classes,
    or to take the same number of elements from each class of the dataset.
    Returns a boolean mask tensor being True at indices of selected elements
    """

    random_generator = torch.Generator()
    random_generator.manual_seed(seed)

    final_indices_bool = torch.zeros(dataset_size, dtype=bool)
    for class_indices in class_indices_mapping.values():
        # Select at least one element
        n_for_class = max(int(len(class_indices) * n_or_percent_per_class), 1) if is_percent else n_or_percent_per_class
        assert isinstance(n_for_class, int)
        filtered_index = torch.randperm(len(class_indices), generator=random_generator)[:n_for_class]
        final_indices_bool[class_indices[filtered_index]] = True
    return final_indices_bool


def _multilabel_rebalance_subset(
    class_indices_mapping: dict[int, torch.Tensor],
    n_or_percent_per_class: int | float,
    labels: torch.Tensor,
    indices_bool: torch.Tensor,
    dataset_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """
    Helper function to refine a subset of a multi-label dataset (indices_bool)
    to better match a target percentage of labels.
    Returns a boolean mask tensor being True at indices of selected elements.
    """

    # Compute the number of selected labels in indices_bool
    num_total_labels = labels.sum()
    num_wanted_labels = int(num_total_labels * n_or_percent_per_class)
    num_selected_labels = (labels[indices_bool] > 0).sum()
    logger.info(f" {num_selected_labels} labels instead of {num_wanted_labels}")

    # Compute a new percentage and new set selecting less images, therefore less labels, to match approximatelly the exact percentage of labels selected
    n_or_percent_per_class = n_or_percent_per_class / (num_selected_labels / num_wanted_labels)
    final_indices_bool = _subset_dataset_per_class(
        class_indices_mapping, n_or_percent_per_class, dataset_size, seed, True
    )

    # Compute the number of labels finally used
    num_selected_labels = (labels[final_indices_bool] > 0).sum()
    logger.info(f" {num_selected_labels} labels instead of {num_wanted_labels}")

    return final_indices_bool


def split_train_val_datasets(train_dataset, split_percentage: float = 0.1, shuffle_train: bool = True):
    """
    Splitting a percent of the train dataset to choose hyperparameters, taking the same percentage for each class.
    If `shuffle` is False, taking the first elements of each class as the validaton set.
    """
    assert 0 < split_percentage < 1
    logger.info(f"Selecting {int(split_percentage * 100)}% of the train dataset as the validation set")
    if shuffle_train:
        logger.info("Shuffling train dataset before splitting in train and validation sets")
        train_dataset = _shuffle_dataset(train_dataset)
    train_labels = get_labels(train_dataset)
    class_indices_mapping = create_class_indices_mapping(train_labels)
    val_mask = torch.zeros(len(train_labels), dtype=bool)
    for class_indices in class_indices_mapping.values():
        # If there is only one element, it goes in the train set
        n_for_val = max(1, int(split_percentage * len(class_indices))) if len(class_indices) > 1 else 0
        val_mask[class_indices[:n_for_val]] = True

    val_dataset = SubsetEx(train_dataset, val_mask.nonzero().flatten())
    train_dataset = SubsetEx(train_dataset, (~val_mask).nonzero().flatten())
    return train_dataset, val_dataset


def create_train_dataset_dict(
    train_dataset,
    few_shot_eval: bool = False,
    few_shot_k_or_percent: float | None = None,
    few_shot_n_tries: int = 1,
) -> dict[int, dict[int, Any]]:
    """
    Randomly split a dataset for few-shot evaluation, with `few_shot_k_or_percent` being
    n elements or x% of a class. Produces a dict, which keys are number of random "tries"
    and values are the dataset subset for this "try".

    Format is {"nth-try": dataset}
    """
    if few_shot_eval is False:
        assert few_shot_k_or_percent is None
        assert few_shot_n_tries == 1
        return {0: train_dataset}

    assert few_shot_k_or_percent is not None
    train_labels = get_labels(train_dataset)
    class_indices_mapping = create_class_indices_mapping(train_labels)
    train_dataset_dict: dict[int, Any] = {}
    is_percent = few_shot_k_or_percent < 1
    if not is_percent:
        few_shot_k_or_percent = int(few_shot_k_or_percent)

    for t in range(few_shot_n_tries):
        t_subset_bool = _subset_dataset_per_class(
            class_indices_mapping=class_indices_mapping,
            n_or_percent_per_class=few_shot_k_or_percent,
            dataset_size=len(train_labels),
            is_percent=is_percent,
            seed=t,
        )
        if len(train_labels.shape) > 1 and is_percent:
            t_subset_bool = _multilabel_rebalance_subset(
                class_indices_mapping=class_indices_mapping,
                n_or_percent_per_class=few_shot_k_or_percent,
                dataset_size=len(train_labels),
                labels=train_labels,
                indices_bool=t_subset_bool,
                seed=t,
            )
        train_dataset_dict[t] = SubsetEx(train_dataset, t_subset_bool.nonzero().flatten())
    return train_dataset_dict


def extract_features_for_dataset_dict(
    model,
    dataset_dict: dict[int, dict[int, Any]],
    batch_size: int,
    num_workers: int,
    gather_on_cpu=False,
    device=None,
) -> dict[int, dict[str, torch.Tensor]]:
    """
    Extract features for each subset of dataset in the context of few-shot evaluations
    """
    few_shot_data_dict: dict[int, dict[str, torch.Tensor]] = {}
    for try_n, dataset in dataset_dict.items():
        features, labels = extract_features(
            model, dataset, batch_size, num_workers, gather_on_cpu=gather_on_cpu, device=device
        )
        few_shot_data_dict[try_n] = {"train_features": features, "train_labels": labels}
    return few_shot_data_dict


def pad_multilabel_and_collate(batch, pad_value=-1):
    """
    This method pads and collates a batch of (image, (index, target)) tuples, coming from
    DatasetWithEnumeratedTargets, with targets that are list of potentially varying sizes.
    The targets are padded to the length of the longest target list in the batch.
    """
    maxlen = max(len(targets) for _, (_, targets) in batch)
    padded_batch = [
        (image, (index, np.pad(targets, (0, maxlen - len(targets)), constant_values=pad_value)))
        for image, (index, targets) in batch
    ]
    return torch.utils.data.default_collate(padded_batch)
