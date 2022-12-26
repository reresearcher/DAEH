# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/5/24 10:43
# @Author : Yufeng Shi
# @Email : yufengshi17@hust.edu.cn
# @File : yuanFlickr.py
# @Software: PyCharm

import torch
import numpy as np
import os
import h5py
import scipy.io as sio
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from .util import read_pkl
from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

def range_data(data):
    data = np.array(data).astype(np.int) - 1
    num_data = max(data.shape)
    return np.reshape(data, (num_data,))

def load_data(root, num_query, num_train, batch_size, num_workers):
    """
    Loading dataset.

    Args:
        root(str): Path of image files.
        num_query(int): Number of query data.
        num_train(int): Number of training data.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader (torch.evaluate.data.DataLoader): Data loader.
    """

    Flickr25k.init(root, num_query, num_train)
    query_dataset = Flickr25k(root, 'query', query_transform())
    train_dataset = Flickr25k(root, 'train', train_transform())
    retrieval_dataset = Flickr25k(root, 'retrieval', query_transform())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader


class Flickr25k(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform

        if mode == 'train':
            self.data = [Flickr25k.TRAIN_DATA, Flickr25k.TRAIN_Y, Flickr25k.TRAIN_X]
            self.targets = Flickr25k.TRAIN_TARGETS
        elif mode == 'query':
            self.data = [Flickr25k.QUERY_DATA, Flickr25k.QUERY_Y, Flickr25k.QUERY_X]
            self.targets = Flickr25k.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = [Flickr25k.RETRIEVAL_DATA, Flickr25k.RETRIEVAL_Y, Flickr25k.RETRIEVAL_X]
            self.targets = Flickr25k.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        sample_image = self.data[0][index].astype(np.float64)
        sample_image = Image.fromarray(np.uint8(sample_image))
        if self.transform is not None:
            img = self.transform(sample_image)

        return img, self.data[1][index].astype(np.float64), self.targets[index].astype(np.float64), index, self.data[2][index].astype(np.float64)

    def __len__(self):
        return self.data[0].shape[0]

    def get_targets(self):
        return torch.FloatTensor(self.targets)

    @staticmethod
    def init(root, num_query, num_train):

        image_root = '/home/shiyufeng/dataset/FLICKR-25K.mat'
        mat_root = '/home/shiyufeng/dataset/Flickr4096.mat'
        mat_file = h5py.File(mat_root, 'r')
        
        IData = range_data(np.squeeze(np.transpose(mat_file['IData'][:])))
        ITest = range_data(np.squeeze(np.transpose(mat_file['ITest'][:])))
        np.random.shuffle(IData)
        ITrain = IData[:num_train]
        
        IdAll = np.arange(len(IData)+len(ITest))
        LAll = np.squeeze(np.transpose(mat_file['LAll']))
        XAll = np.squeeze(np.transpose(mat_file['XAll']))
        YAll = np.squeeze(np.transpose(mat_file['YAll']))
        
        with h5py.File(image_root, 'r') as file:
           images = np.squeeze(file['images'][:].transpose(0, 3, 2, 1))
           file.close()
           
        
        print(np.shape(images))

        Flickr25k.QUERY_DATA = images[ITest, :, :, :]
        Flickr25k.QUERY_X = XAll[ITest, :]
        Flickr25k.QUERY_Y = YAll[ITest, :]
        Flickr25k.QUERY_TARGETS = LAll[ITest, :]

        Flickr25k.TRAIN_DATA = images[ITrain, :, :, :]
        Flickr25k.TRAIN_X = XAll[ITrain, :]
        Flickr25k.TRAIN_Y = YAll[ITrain, :]
        Flickr25k.TRAIN_TARGETS = LAll[ITrain, :]

        Flickr25k.RETRIEVAL_DATA = images[IData, :, :, :]
        Flickr25k.RETRIEVAL_X = XAll[IData, :]
        Flickr25k.RETRIEVAL_Y = YAll[IData, :]
        Flickr25k.RETRIEVAL_TARGETS = LAll[IData, :]
