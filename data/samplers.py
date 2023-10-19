# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math

import torch
from torch.utils.data import Sampler


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        indexes: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
    """

    def __init__(self, indexes, num_replicas, rank):
        super().__init__(None)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        n_samples_per_rep = math.ceil(len(indexes) / self.num_replicas)
        indices = torch.split(indexes, n_samples_per_rep)
        sizes = [0]
        for i in range(1, len(indices)):
            if indices[i][0] == indices[i - 1][-1]:
                sizes.append(indices[i][0].item() - 1)
            else:
                sizes.append(indices[i][0].item())

        sizes.append(indexes[-1].item() + 1)
        item_count_per_rank = []
        for i in range(len(sizes) - 1):
            mask_items = torch.greater_equal(indexes, sizes[i])
            mask_items = torch.logical_and(mask_items, torch.less_equal(indexes, sizes[i + 1]))
            item_count_per_rank.append(torch.sum(mask_items))

        self.max_items_count_per_gpu = max(item_count_per_rank)
        all_indicates = []
        n_items = []
        for i in range(len(sizes) - 1):
            all_indicates.append(torch.arange(sizes[i], sizes[i + 1]))
            n_items.append(torch.sum(indexes < sizes[i + 1]))
        self.samples = all_indicates[self.rank]
        self.n_items = n_items
        self.num_samples = len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return self.num_samples
