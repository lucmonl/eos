import torch
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset
from cifar import load_cifar, load_cifar_random, load_synthetic
from synthetic import make_chebyshev_dataset, make_linear_dataset
# from wikitext import load_wikitext_2

DATASETS = [
    "cifar10", "cifar10-10", "cifar10-100", "cifar10-1k", "cifar10-2k", "cifar10-5k", "cifar10-10k", "cifar10-20k", "chebyshev-3-20",
    "chebyshev-4-20", "chebyshev-5-20", "linear-50-50","cifar10-1k-random-10", "cifar10-1k-random-5", 
    "cifar10-1k-random-2", "cifar10-1k-random-1", "cifar10-1k-random-3", "synthetic-1", "synthetic-5", "synthetic-cifar-1", 
    "synthetic-cifar-5", "synthetic-cifar-6", "synthetic-cifar-7", "synthetic-cifar-8", 
    "synthetic-reduced-cifar-1"
]

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def num_input_channels(dataset_name: str) -> int:
    if 'reduced-cifar' in dataset_name:
        return 1
    elif "cifar" in dataset_name:
        #if dataset_name.startswith("cifar10"):
        return 3
    elif dataset_name == 'fashion':
        return 1

def image_size(dataset_name: str) -> int:
    if 'reduced-cifar' in dataset_name:
        return 8
    elif "cifar" in dataset_name:
        #if dataset_name.startswith("cifar10"):
        return 32
    elif dataset_name == 'fashion':
        return 28

def num_classes(dataset_name: str) -> int:
    if dataset_name.startswith('cifar10'):
        if "random" in dataset_name:
            num_class = int(dataset_name.split("-")[-1])
            return num_class
        return 10
    elif dataset_name == 'fashion':
        return 10
    elif dataset_name.startswith('synthetic'):
        num_class = int(dataset_name.split("-")[-1])
        return num_class

def get_pooling(pooling: str):
    if pooling == 'max':
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == 'average':
        return torch.nn.AvgPool2d((2, 2))
    else:
        raise NotImplementedError("unknown pooling: {}".format(pooling))

def num_pixels(dataset_name: str) -> int:
    if "cifar" in dataset_name:
        return num_input_channels(dataset_name) * image_size(dataset_name)**2
    elif dataset_name.startswith("synthetic"):
        return 50
    else:
        assert False

def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])

def load_dataset(dataset_name: str, loss: str) -> (TensorDataset, TensorDataset):
    if dataset_name == "cifar10":
        return load_cifar(loss)
    elif dataset_name == "cifar10-10":
        train, test = load_cifar(loss)
        return take_first(train, 10), take_first(test, 10)
    elif dataset_name == "cifar10-100":
        train, test = load_cifar(loss)
        return take_first(train, 100), take_first(test, 100)
    elif dataset_name == "cifar10-1k":
        train, test = load_cifar(loss)
        return take_first(train, 1000), test
    elif dataset_name == "cifar10-2k":
        train, test = load_cifar(loss)
        return take_first(train, 2000), test
    elif dataset_name == "cifar10-5k":
        train, test = load_cifar(loss)
        return take_first(train, 5000), test
    elif dataset_name == "cifar10-10k":
        train, test = load_cifar(loss)
        return take_first(train, 10000), test
    elif dataset_name == "cifar10-20k":
        train, test = load_cifar(loss)
        return take_first(train, 20000), test
    elif dataset_name == "chebyshev-5-20":
        return make_chebyshev_dataset(k=5, n=20)
    elif dataset_name == "chebyshev-4-20":
        return make_chebyshev_dataset(k=4, n=20)
    elif dataset_name == "chebyshev-3-20":
        return make_chebyshev_dataset(k=3, n=20)
    elif dataset_name == 'linear-50-50':
        return make_linear_dataset(n=50, d=50)
    #elif dataset_name == "cifar10-1k-random-10":
    elif dataset_name.startswith("cifar10-1k-random-"):
        num_class = int(dataset_name.split("-")[-1])
        train, test = load_cifar_random(loss, num_class)
        return take_first(train, 1000), test

    elif dataset_name.startswith("synthetic"):
        train, test = load_synthetic(dataset_name)
        return train, test 


