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
from torch.utils.data import RandomSampler
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
    dataset = GsocDataset(
        root=root_path,
        transform=transform)
    logger.info('GSOC dataset created')
    dist_sampler = RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('GSOC unsupervised data loader created')

    return dataset, data_loader, dist_sampler

class GsocDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.h5_file = h5py.File(root, "r")
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.h5_file["jet"][index])
        else:
            return self.h5_file["jet"][index]
        
    def __len__(self):
        return self.h5_file["jet"].size

class CustomDataLoader():
    def __init__(self, file_path, indices, batch_size):
        self.file_path = file_path
        self.h5_file = h5py.File(file_path, "r")
        self.indices = indices
        self.batch_size = batch_size
        self.on_epoch_end()
        self.idx = 0

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self):
        start_idx = self.idx * self.batch_size
        end_idx = min((self.idx + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]

        t0 = time.time()
        batch_indices = np.sort(batch_indices)
        t1 = time.time()
        diff = t1-t0
        print('Time to sort: ', diff)
        # Load data from file based on batch indices
        t0 = time.time()
        imgs = self.load_data(batch_indices)
        t1 = time.time()
        diff = t1-t0
        print('Time to load: ', diff)

        # Preprocess data
        imgs = self.process_data(imgs)
        self.idx +=1

        return imgs

    def load_data(self, batch_indices):
        # Load rows corresponding to batch_indices from the file
        #with h5py.File(self.file_path, 'r') as f:
        #  data = f['jet'][batch_indices]
        data = self.h5_file['jet'][batch_indices]

        return data

    def process_data(self, data):

        return data / 255.

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        np.random.shuffle(self.indices)

