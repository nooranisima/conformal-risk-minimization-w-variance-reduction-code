import torch
import numpy as np
import random
import pickle
from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as dset
import torchvision.transforms as trn
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from enum import Enum 


def set_random_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

SUPPORTED_DATASETS = {'mnist', 'fmnist'}


def get_transform(dataset_name: str):
    if dataset_name in ['mnist', 'fmnist']:
        return trn.Compose([
            trn.ToTensor(),
            trn.Normalize((0.5,), (0.5,))
        ])

def get_dataset(dataset_name: str, data_dir: Path = Path('data'), mode: str = 'train'):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    transform = get_transform(dataset_name)
    if dataset_name == 'mnist':
        dataset_class = dset.MNIST
    elif dataset_name == 'fmnist':
        dataset_class = dset.FashionMNIST

    
    
    dataset_args = {'download': True, 'train': mode == 'train'}
    
    return dataset_class(root=data_dir, transform=transform, **dataset_args)

def load_data(config, seed):
    set_random_seeds(seed)

    if config.dataset_name in ['mnist', 'fmnist']:
        train = get_dataset(config.dataset_name, mode = 'train')
        train, calib = random_split(train, [config.train_size, config.calib_size])
        test = get_dataset(config.dataset_name, mode='test')

    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, drop_last = False)
    calib_loader = DataLoader(calib, batch_size=config.batch_size, shuffle=False, drop_last = False)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False, drop_last = False)

    return train_loader, calib_loader, test_loader