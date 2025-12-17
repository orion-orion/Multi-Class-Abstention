# -*- coding: utf-8 -*-
import copy
import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets, transforms
from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Normalize, \
    ToPILImage, RandomCrop, RandomHorizontalFlip
from utils.cus_dataset import CustomSubset, VehicleDataset


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
            root="./data", split="byclass", download=True, transform=transform,
            train=True)
        # test = False，从测试集create数据
        test_data = datasets.SVHN(
            root="./data", split="byclass", download=True, transform=transform,
            train=False)
    elif args.dataset == "CIFAR10":
        train_transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )
        train_data = datasets.CIFAR10(
            root="./data", download=True, transform=train_transform,
            train=True)
        # test = False，从测试集create数据
        test_transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )
        test_data = datasets.CIFAR10(
            root="./data", download=True, transform=test_transform,
            train=False)
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
    elif args.dataset == "vehicle":
        # train_transform = transforms.Compose(
        #     [
        #         lambda x: torch.FloatTensor(x)
        #     ]
        # )
        X, y = sk_datasets.load_svmlight_file(
            f="data/vehicle/vehicle_dataset.txt")
        X, y = torch.FloatTensor(X.todense()), torch.Tensor(
            y).type(torch.int64) - 1
        X = X - torch.mean(X, dim=0) / torch.std(X, dim=0)
        dataset = VehicleDataset(X, y)
    else:
        raise IOError("Please input the correct dataset name, it must be one of:"
                      "SVHN, CIFAR10, CIFAR100, vehicle.")

    # 获取训练集相关属性
    if args.dataset != "vehicle":
        if len(train_data.data[0].shape) == 2:
            data_info["n_channels"] = 1
        else:
            data_info["n_channels"] = train_data.data[0].shape[2]
        data_info["classes"] = train_data.classes
        data_info["input_sz"], data_info["num_cls"] = train_data.data[0].shape[0], len(
            train_data.classes)
    else:
        data_info["input_sz"], data_info["num_cls"] = dataset.X.shape[1], int(
            dataset.y.max().item()) + 1
    if args.train_val_frac:
        train_frac, val_frac = [float(x) for x in args.train_val_frac]
        if args.dataset != "vehicle":
            dataset = ConcatDataset([train_data, test_data])

        n_samples = len(dataset)
        # n_train = int(n_samples * train_frac)
        # n_test = n_samples - n_train
        train_idcs, test_idcs = train_test_split(
            np.arange(n_samples), train_size=train_frac,
            random_state=args.seed)
        if val_frac > 0:
            # n_val = int(n_train * val_frac)
            # n_train -= n_val
            train_idcs, valid_idcs = train_test_split(
                train_idcs, train_size=1 - val_frac, random_state=args.seed
            )
            if args.dataset != "vehicle":
                valid_transform = transforms.Compose(
                    [
                        Normalize(
                            (0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                )
            else:
                valid_transform = None
            val_dataset = CustomSubset(
                dataset, valid_idcs, valid_transform)
        else:
            # n_val = 0
            val_dataset = None
        if args.dataset != "vehicle":
            train_transform = transforms.Compose(
                [
                    # RandomCrop(size=(32, 32), padding=4),
                    RandomHorizontalFlip(p=0.5),
                    RandomCrop(size=(32, 32), padding=4),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            test_transform = transforms.Compose(
                [
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        else:
            train_transform, test_transform = None, None
        train_dataset = CustomSubset(
            dataset, train_idcs, train_transform)
        test_dataset = CustomSubset(
            dataset, test_idcs, test_transform)
        data_info["n_samples"] = n_samples
        data_info["n_samples_train"] = len(train_idcs)
    else:
        if args.dataset == "vehicle":
            raise Exception(
                "For Vehicle, must specify `train_val_frac`")

        train_transform = transforms.Compose(
            [
                # RandomCrop(size=(32, 32), padding=4),
                RandomHorizontalFlip(p=0.5),
                RandomCrop(size=(32, 32), padding=4),
                Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        n_train, n_test = len(train_data), len(test_data)
        train_dataset = CustomSubset(
            train_data, np.arange(n_train), train_transform)
        test_dataset = CustomSubset(
            test_data, np.arange(n_test), test_transform)
        val_dataset = None
        data_info["n_samples"] = n_train + n_test
        data_info["n_samples_train"] = n_train

    return train_dataset, val_dataset, test_dataset, data_info
