import numpy as np
from torchvision.datasets import CIFAR10
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import sys
import torch
from torch import Tensor
import torch.nn.functional as F

DATASETS_FOLDER = os.environ["DATASETS"]

def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y, loss):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, 10, 0)
    elif loss == "exp":
        return _one_hot(y, 10, 0)

def make_labels_random(y, loss, num_class):
    if loss == "ce":
        return torch.randint(num_class, y.shape)
    elif loss == "mse":
        return _one_hot(torch.randint(num_class, y.shape), num_class, 0)
        #return torch.rand(y.shape[0], 1)

def load_synthetic(dataset: str):
    x_train = np.load("../synthetic/{}/x_train.npy".format(dataset))
    y_train = np.load("../synthetic/{}/y_train.npy".format(dataset))
    x_test = np.load("../synthetic/{}/x_test.npy".format(dataset))
    y_test = np.load("../synthetic/{}/y_test.npy".format(dataset))
    train = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
    return train, test

def load_random(dataset: str):
    N = 40
    d = 100
    np.random.seed(0)
    X=np.random.normal(size=(N,d))

    # Initialization beta_star,y,Loss array
    sample_index=np.random.choice(np.arange(0,d),size=5)
    beta_star=np.zeros((d,1))
    for i in range(sample_index.shape[0]):
        beta_star[sample_index[i],0]=np.random.normal()
    # print(beta_star)

    y=X@beta_star

    x_train = X
    y_train = y
    x_test = X
    y_test = y
    train = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
    return train, test

def load_cifar(loss: str) -> (TensorDataset, TensorDataset):
    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test

def load_cifar_uncentered(loss: str) -> (TensorDataset, TensorDataset):
    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    #center_X_train, center_X_test = center(X_train, X_test)
    center_X_train, center_X_test = X_train, X_test
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test

def load_cifar_random(loss: str, num_class: int) -> (TensorDataset, TensorDataset):
    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels_random(torch.tensor(cifar10_train.targets), loss, num_class), \
        make_labels_random(torch.tensor(cifar10_test.targets), loss, num_class)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test

