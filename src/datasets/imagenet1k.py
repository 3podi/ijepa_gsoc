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
from torch.utils.data import DataLoader
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
    
    dataset = GsocDataset3( root_path, preload_size=batch_size)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_size = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_sampler = ChunkedSampler(train_indices, chunk_size=batch_size, shuffle=True)
    val_sampler = ChunkedSampler(val_indices, chunk_size=batch_size, shuffle=False)

    train_data_loader = DataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   pin_memory=pin_mem,
                                   collate_fn=collator,
                                   num_workers=num_workers)

    val_data_loader = DataLoader(dataset,
                                 batch_size=batch_size, 
                                 sampler=val_sampler, 
                                 pin_memory=pin_mem,
                                 collate_fn=collator, 
                                 num_workers=num_workers)
    
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
        t0 = time.time()
        #data = self.transforms(self.jet[idx])
        data = self.jet[idx]
        #data = np.pad(data, ((0, 0), (0, 1), (0, 1)), mode='constant')
        t1 = time.time()
        print('time to get data: ', t1-t0)
        return torch.tensor(data)


class ChunkedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, chunk_size=3200, shuffle=False):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size
        self.indices = list(range(len(data_source)))
        self.shuffle = shuffle

    def shuffle_indices(self):
        chunk_indices = [self.indices[i * self.chunk_size:(i + 1) * self.chunk_size] for i in range(self.num_chunks)]
        np.random.shuffle(chunk_indices)
        self.indices = [idx for chunk in chunk_indices for idx in chunk]

    def __iter__(self):
        if self.shuffle:
            self.shuffle_indices()
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)

class GsocDataset3(torch.utils.data.Dataset):
    def __init__(self, h5_path, transforms=None, preload_size=3200):
        self.h5_path = h5_path
        self.transforms = transforms
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        self.data = self.h5_file['all_jet']
        #self.labels = self.h5_file['m0']
        self.dataset_size = self.data.shape[0]

        self.chunk_size = self.data.chunks

        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]
            #self.preloaded_labels = self.labels[preload_start:preload_end]

        local_idx = idx - self.preload_start
        data = self.preloaded_data[local_idx]
        #labels = self.preloaded_labels[local_idx]
        if self.transforms:
            data = self.transforms(data)
        return torch.from_numpy(data)#, torch.from_numpy(labels)

    def __del__(self):
        self.h5_file.close()

 """    dataset = GsocDataset2(
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
        persistent_workers=True) """