# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
import h5py
from torch.utils.data import SubsetRandomSampler
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    dataset = GsocDataset2(
        root=root_path,
        transform=transform)
    logger.info('GSOC dataset created')

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_size = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        sampler=train_sampler,
        persistent_workers=True)

    val_data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        sampler=val_sampler,
        persistent_workers=True)
    
    logger.info('GSOC unsupervised data loaders created')

    return dataset, train_data_loader, val_data_loader

class GsocDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.file_path = root
        self.transform = transform
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["jet"])

    def __getitem__(self, index):
        if self.dataset is None:
            t0 = time.time()
            self.dataset = h5py.File(self.file_path, 'r')["jet"]
            t1 = time.time()
            print('Time to open file: ', t1-t0)
        return self.transform(self.dataset[index])

    def __len__(self):
        return self.dataset_len

class GsocDataset2(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        t0 = time.time()
        f =  h5py.File(root, 'r')
        self.jet = f['jet']
        t1 = time.time()
        print('time to open file: ', t1-t0)
        self.transforms = transform
    def __len__(self):
        return len(self.jet)
    def __getitem__(self, idx):
        #t0 = time.time()
        data = self.transforms(self.jet[idx])
        #t1 = time.time()
        #print('time to get data: ', t1-t0)
        return data