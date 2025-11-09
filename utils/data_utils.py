# -*- coding: utf-8 -*-
import copy
import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Normalize, \
    ToPILImage, RandomCrop, RandomHorizontalFlip
from utils.subsets import CustomSubset


def load_dataset(args):
    """load train and test dataset

    Args:
        args: the namespace object including args

    Returns:
        train_data: train_data
    """
    if not os.path.exists("./data"):
        os.mkdir("./data")

    data_info = {}
    if args.dataset == "SVHN":
        transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )
        # train = True，从训练集create数据
        train_data = datasets.SVHN(
            root="./data", split="byclass", download=True, transform=transform, train=True)
        # test = False，从测试集create数据
        test_data = datasets.SVHN(
            root="./data", split="byclass", download=True, transform=transform, train=False)
    elif args.dataset == "CIFAR10":
        train_transform = transforms.Compose(
            [
                ToTensor(),
                lambda x: x - torch.mean(x, dim=(1, 2), keepdim=True),
                RandomCrop(size=(32, 32), padding=4),
            ]
        )
        train_data = datasets.CIFAR10(
            root="./data", download=True, transform=train_transform, train=True)
        # test = False，从测试集create数据
        test_transform = transforms.Compose(
            [
                ToTensor(),
                lambda x: x - torch.mean(x, dim=(1, 2), keepdim=True),
            ]
        )
        test_data = datasets.CIFAR10(
            root="./data", download=True, transform=test_transform, train=False)
    elif args.dataset == "CIFAR100":
        transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )
        # train = True，从训练集create数据
        train_data = datasets.CIFAR100(
            root="./data", download=True, transform=transform, train=True)
        # test = False，从测试集create数据
        test_data = datasets.CIFAR100(
            root="./data", download=True, transform=transform, train=False)
    else:
        raise IOError("Please input the correct dataset name, it must be one of:"
                      "SVHN, CIFAR10, CIFAR100.")

    # 获取训练集相关属性
    if len(train_data.data[0].shape) == 2:
        data_info["n_channels"] = 1
    else:
        data_info["n_channels"] = train_data.data[0].shape[2]
    data_info["classes"] = train_data.classes
    data_info["input_sz"], data_info["num_cls"] = train_data.data[0].shape[0], len(
        train_data.classes)
    labels = np.concatenate(
        [np.array(train_data.targets), np.array(test_data.targets)], axis=0)
    dataset = ConcatDataset([train_data, test_data])

    n_samples = len(dataset)
    n_train = int(n_samples * args.train_frac)
    n_test = n_samples - n_train
    if args.val_frac > 0:
        n_val = int(n_train * args.val_frac)
        n_train -= n_val
        val_dataset = CustomSubset(
            dataset, list(range(n_train, n_train + n_val)))
    else:
        n_val = 0
        val_dataset = None
    train_transform = transforms.Compose(
        [
            # Normalize(
            #     (0.4914, 0.4822, 0.4465),
            #     (0.2023, 0.1994, 0.2010)
            # ),
            # transforms.Grayscale(num_output_channels=1)
            RandomCrop(size=(32, 32), padding=4),
            RandomHorizontalFlip(p=0.5),
        ]
    )
    train_dataset = CustomSubset(
        dataset, list(range(n_train)), train_transform)
    test_dataset = CustomSubset(dataset, list(
        range(n_test, n_samples)))
    data_info["n_samples"] = n_samples
    data_info["n_samples_train"] = n_train
    data_info["n_samples_valid"] = n_val
    data_info["n_samples_test"] = n_test

    return train_dataset, test_dataset, val_dataset, data_info
