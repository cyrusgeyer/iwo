# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List
import os

import numpy as np
import torch.utils.data


def load_dataset_as_np(
    representation_path, factor_path, shuffle=True, data_fraction=None
):
    """using this function necessitates having extracted representations from Locatello et al. disentanglement_lib
    base_path (str): path to the folder containing the locatello result folders
    exp_num (int): locatello experiment number
    shuffle (bool): whether to shuffle the dataset
    data_fraction (float): fraction of the dataset to load
    """

    if os.path.exists(representation_path):
        print(f"Load features from: {representation_path}")
        representations = np.load(representation_path).astype(np.float32)
        print("Loaded")
    else:
        raise ValueError(f"Data at: {representation_path} does not exists.")

    if os.path.exists(factor_path):
        print(f"Load factor from: {factor_path}")
        factors = np.load(factor_path).astype(np.float32)
        print("Loaded")

        _min = np.min(factors, 0)
        _max = np.max(factors, 0)

        unnormalized_factors = factors
        factors = (factors - _min) / (_max - _min)
    else:
        raise ValueError(f"Data at: {factor_path} does not exists.")

    if shuffle or data_fraction is not None:
        perm = np.random.permutation(len(factors))

        representations = representations[perm]
        factors = factors[perm]
        unnormalized_factors = unnormalized_factors[perm]

    if data_fraction is not None and data_fraction != 1.0:
        if data_fraction < 1.0:
            num_samples = int(len(factors) * data_fraction)
        else:
            num_samples = data_fraction
        print(f"Loading only fraction of the dataset: {num_samples} samples")

        representations = representations[:num_samples]
        factors = factors[:num_samples]
        unnormalized_factors = unnormalized_factors[:num_samples]

    return representations, factors, unnormalized_factors


def make_datasets(cfg):
    reps, factors, un_factors = load_dataset_as_np(
        cfg.representation_path, cfg.factor_path, True, cfg.fraction
    )

    N = len(reps)
    split = cfg.split

    train_end_idx = int(split[0] * N)
    val_end_idx = train_end_idx + int(split[1] * N)

    splits = [train_end_idx, val_end_idx]

    train_dataset = LocatelloDataset(
        reps[: splits[0]],
        factors[: splits[0]],
        un_factors[: splits[0]],
        cfg.name,
    )
    val_dataset = LocatelloDataset(
        reps[splits[0] : splits[1]],
        factors[splits[0] : splits[1]],
        un_factors[splits[0] : splits[1]],
        cfg.name,
    )
    test_dataset = LocatelloDataset(
        reps[splits[1] :],
        factors[splits[1] :],
        un_factors[splits[1] :],
        cfg.name,
    )

    return train_dataset, val_dataset, test_dataset


class LocatelloDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_feats,
        dataset_targets,
        unnormalized_targets,
        dataset_name,
    ):
        super().__init__()

        self._dataset_feats = dataset_feats
        self._dataset_targets = dataset_targets
        self._unnormalized_targets = unnormalized_targets

        self._factor_sizes = None
        self._factor_names = None
        self._factor_discrete = None
        self._factor_discrete_more = None
        self.dataset_name = dataset_name

        if dataset_name == "dsprites":
            self._factor_sizes = [3, 6, 40, 32, 32]
            self._factor_names = [
                "shape",
                "scale",
                "orientation",
                "x-position",
                "y-position",
            ]
            self._factor_discrete = [True, False, True, False, False]
            self._factor_discrete_more = [True, True, False, False, False]

        elif dataset_name == "cars3d":
            self._factor_sizes = [4, 24, 183]
            self._factor_names = ["elevation", "azimuth", "object"]
            # self._factor_discrete = [False, True, False, False, False, False, False]
            self._factor_discrete = [False, False, True]
            self._factor_discrete_more = [False, False, True]

        elif dataset_name == "smallnorb":
            self._factor_sizes = [5, 9, 18, 6]
            self._factor_names = ["category", "elevation", "azimuth", "lighting"]
            # self._factor_discrete = [False, True, False, False, False, False, False]
            self._factor_discrete = [True, False, False, False]

    def __len__(self):
        return len(self._dataset_targets)

    @property
    def normalized_targets(self):
        return self._dataset_targets

    def __getitem__(self, idx: int, normalize: bool = True):
        image = self._dataset_feats[idx]
        targets = self._dataset_targets[idx]
        return image, targets
