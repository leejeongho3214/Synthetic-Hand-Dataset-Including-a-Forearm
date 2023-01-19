"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""


import os.path as op
import torch
import logging
from torch.utils.data import Dataset, DataLoader, random_split
import code
from src.utils.comm import get_world_size
from src.datasets.human_mesh_tsv import (MeshTSVDataset, MeshTSVYamlDataset)
from src.datasets.hand_mesh_tsv import (HandMeshTSVDataset, HandMeshTSVYamlDataset)


def build_dataset(yaml_file, args, is_train=True, scale_factor=1):
    # print(yaml_file)
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        # code.interact(local=locals())
        assert op.isfile(yaml_file)
    return MeshTSVYamlDataset(yaml_file, is_train, False, scale_factor)


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_batch_data_sampler(sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, yaml_file, is_distributed=True, 
        is_train=True, start_iter=0, scale_factor=1):

    dataset = build_dataset(yaml_file, args, is_train=is_train, scale_factor=scale_factor)
    # logger = logging.getLogger(__name__)
    if is_train==True:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        # logger.info("Train with {} images per GPU.".format(images_per_gpu))
        # logger.info("Total batch size {}".format(images_per_batch))
        # logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader


#==============================================================================================

def build_hand_dataset(yaml_file, args, is_train=True, scale_factor=1, s_j = None):
    # print(yaml_file)
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        # code.interact(local=locals())
        assert op.isfile(yaml_file)
    return HandMeshTSVYamlDataset(args, yaml_file, is_train, False, scale_factor, s_j)


def make_hand_data_loader(args, yaml_file, is_distributed=False,
        is_train=True, start_iter=0, scale_factor=1, s_j = None):

    dataset = build_hand_dataset(yaml_file, args, is_train=is_train, scale_factor=scale_factor, s_j = s_j)

    # train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.9), int(len(dataset) * 0.1)])
    # train_data_loader = torch.utils.data.DataLoader(
    #     train_dataset, num_workers=args.num_workers, batch_size=32,
    #     pin_memory=True,
    # )
    # test_data_loader = torch.utils.data.DataLoader(
    #     test_dataset, num_workers=args.num_workers, batch_size=32,
    #     pin_memory=True,
    # )
    # return train_data_loader, test_data_loader, train_dataset, test_dataset
    return dataset

