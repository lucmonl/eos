from typing import List

import numpy as np
import torch
import torch.nn as nn

from resnet_cifar import resnet32
from vgg import vgg11_nodropout, vgg11_nodropout_bn
from data import num_classes, num_input_channels, image_size, num_pixels

_CONV_OPTIONS = {"kernel_size": 3, "padding": 1, "stride": 1}

def get_activation(activation: str):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'hardtanh':
        return torch.nn.Hardtanh()
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise NotImplementedError("unknown activation function: {}".format(activation))

def get_pooling(pooling: str):
    if pooling == 'max':
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == 'average':
        return torch.nn.AvgPool2d((2, 2))

class Scale(nn.Module):
    def __init__(self, factor) -> None:
        super(Scale, self).__init__()
        self.factor = factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.factor * input

def fully_connected_net(dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            Scale(1/np.sqrt(prev_width)),
            get_activation(activation),
        ])
    modules.append(nn.Linear(widths[-1], num_classes(dataset_name), bias=bias))
    modules.append(Scale(1/np.sqrt(widths[-1])))
    return nn.Sequential(*modules)

def nfc_net(dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
        ])
    modules.append(nn.Linear(widths[-1], num_classes(dataset_name), bias=bias))
    return nn.Sequential(*modules)

def layer_norm_net(dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            #Scale(1/np.sqrt(prev_width)),
            get_activation(activation),
            nn.LayerNorm(widths[l]),
        ])
    modules.append(nn.Linear(widths[-1], num_classes(dataset_name), bias=bias))
    #modules.append(Scale(1/np.sqrt(widths[-1])))
    return nn.Sequential(*modules)

def batch_norm_net(dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            #Scale(1/np.sqrt(prev_width)),
            get_activation(activation),
            nn.BatchNorm1d(widths[l]),
        ])
    modules.append(nn.Linear(widths[-1], num_classes(dataset_name), bias=bias))
    #modules.append(Scale(1/np.sqrt(widths[-1])))
    return nn.Sequential(*modules)


def fully_connected_net_bn(dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
            nn.BatchNorm1d(widths[l])
        ])
    modules.append(nn.Linear(widths[-1], num_classes(dataset_name), bias=bias))
    return nn.Sequential(*modules)


def convnet(dataset_name: str, widths: List[int], activation: str, pooling: str, bias: bool) -> nn.Module:
    modules = []
    size = image_size(dataset_name)
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_input_channels(dataset_name)
        modules.extend([
            nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
            get_activation(activation),
            get_pooling(pooling),
        ])
        size //= 2
    modules.append(nn.Flatten())
    modules.append(nn.Linear(widths[-1]*size*size, num_classes(dataset_name)))
    return nn.Sequential(*modules)


def convnet_bn(dataset_name: str, widths: List[int], activation: str, pooling: str, bias: bool) -> nn.Module:
    modules = []
    size = image_size(dataset_name)
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_input_channels(dataset_name)
        modules.extend([
            nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
            get_activation(activation),
            nn.BatchNorm2d(widths[l]),
            get_pooling(pooling),
        ])
        size //= 2
    modules.append(nn.Flatten())
    modules.append(nn.Linear(widths[-1]*size*size, num_classes(dataset_name)))
    return nn.Sequential(*modules)

def make_deeplinear(L: int, d: int, seed=8):
    torch.manual_seed(seed)
    layers = []
    for l in range(L):
        layer = nn.Linear(d, d, bias=False)
        nn.init.xavier_normal_(layer.weight)
        layers.append(layer)
    network = nn.Sequential(*layers)
    return network.cuda()

def make_one_layer_linear(d: int, dataset_name: str, seed=8):
    torch.manual_seed(seed)
    layers = [nn.Flatten()]
    layers.append(nn.Linear(num_pixels(dataset_name), d, bias=False))
    layers.append(nn.Linear(d, num_classes(dataset_name), bias=False))
    network = nn.Sequential(*layers)
    return network.cuda()

def make_one_layer_network(h=10, seed=0, activation='tanh', sigma_w=1.9):
    torch.manual_seed(seed)
    network = nn.Sequential(
        nn.Linear(1, h, bias=True),
        get_activation(activation),
        nn.Linear(h, 1, bias=False),
    )
    nn.init.xavier_normal_(network[0].weight, gain=sigma_w)
    nn.init.zeros_(network[0].bias)
    nn.init.xavier_normal_(network[2].weight)
    return network


def load_architecture(arch_id: str, dataset_name: str) -> nn.Module:
    #  ======   fully-connected networks =======
    if arch_id == 'fc-relu':
        return fully_connected_net(dataset_name, [200, 200], 'relu', bias=True)
    elif arch_id == 'fc-elu':
        return fully_connected_net(dataset_name, [200, 200], 'elu', bias=True)
    elif arch_id == 'fc-tanh':
        return fully_connected_net(dataset_name, [200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-hardtanh':
        return fully_connected_net(dataset_name, [200, 200], 'hardtanh', bias=True)
    elif arch_id == 'fc-softplus':
        return fully_connected_net(dataset_name, [200, 200], 'softplus', bias=True)

    #  ======   convolutional networks =======
    elif arch_id == 'cnn-relu':
        return convnet(dataset_name, [32, 32], activation='relu', pooling='max', bias=True)
    elif arch_id == 'cnn-elu':
        return convnet(dataset_name, [32, 32], activation='elu', pooling='max', bias=True)
    elif arch_id == 'cnn-tanh':
        return convnet(dataset_name, [32, 32], activation='tanh', pooling='max', bias=True)
    elif arch_id == 'cnn-avgpool-relu':
        return convnet(dataset_name, [32, 32], activation='relu', pooling='average', bias=True)
    elif arch_id == 'cnn-avgpool-elu':
        return convnet(dataset_name, [32, 32], activation='elu', pooling='average', bias=True)
    elif arch_id == 'cnn-avgpool-tanh':
        return convnet(dataset_name, [32, 32], activation='tanh', pooling='average', bias=True)

    #  ======   convolutional networks with BN =======
    elif arch_id == 'cnn-bn-relu':
        return convnet_bn(dataset_name, [32, 32], activation='relu', pooling='max', bias=True)
    elif arch_id == 'cnn-bn-elu':
        return convnet_bn(dataset_name, [32, 32], activation='elu', pooling='max', bias=True)
    elif arch_id == 'cnn-bn-tanh':
        return convnet_bn(dataset_name, [32, 32], activation='tanh', pooling='max', bias=True)

    #  ======   real networks on CIFAR-10  =======
    elif arch_id == 'resnet32':
        return resnet32()
    elif arch_id == 'vgg11':
        return vgg11_nodropout()
    elif arch_id == 'vgg11-bn':
        return vgg11_nodropout_bn()

    # ====== additional networks ========
    # elif arch_id == 'transformer':
        # return TransformerModelFixed()
    elif arch_id == 'deeplinear':
        return make_deeplinear(20, 50)
    elif arch_id == 'regression':
        return make_one_layer_network(h=100, activation='tanh')
    elif arch_id == 'linear-50':
        return make_one_layer_linear(50, dataset_name)

    # ======= vary depth =======
    elif arch_id == 'fc-tanh-depth1':
        return fully_connected_net(dataset_name, [200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth2':
        return fully_connected_net(dataset_name, [200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth3':
        return fully_connected_net(dataset_name, [200, 200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth4':
        return fully_connected_net(dataset_name, [200, 200, 200, 200], 'tanh', bias=True)

    # ======= vary width =======
    elif arch_id == 'fc-tanh-width2':
        return fully_connected_net(dataset_name, [400, 400], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width3':
        return fully_connected_net(dataset_name, [800, 800], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width4':
        return fully_connected_net(dataset_name, [1200, 1200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width5':
        return fully_connected_net(dataset_name, [50000], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width6':
        return fully_connected_net(dataset_name, [5000], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width7':
        return fully_connected_net(dataset_name, [500], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width8':
        return fully_connected_net(dataset_name, [50], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width9':
        return fully_connected_net(dataset_name, [10], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width10':
        return fully_connected_net(dataset_name, [2000], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-width11':
        return fully_connected_net(dataset_name, [1], 'tanh', bias=True)

    elif arch_id == 'fc-tanh-width8-no-bias':
        return fully_connected_net(dataset_name, [50], 'tanh', bias=False)
    elif arch_id == 'fc-tanh-width11-no-bias':
        return fully_connected_net(dataset_name, [1], 'tanh', bias=False)
    
    
    elif arch_id == 'nfc-relu-width8':
        return nfc_net(dataset_name, [50], 'relu', bias=True)
    elif arch_id == 'nfc-relu-width8-no-bias':
        return nfc_net(dataset_name, [50], 'relu', bias=False)
    elif arch_id == 'nfc-tanh-width8-no-bias':
        return nfc_net(dataset_name, [50], 'tanh', bias=False)
    

    # ======== layer normalization ========
    elif arch_id == 'layer-norm-width6':
        return layer_norm_net(dataset_name, [5000], 'tanh', bias=True)
    elif arch_id == 'layer-norm-width7':
        return layer_norm_net(dataset_name, [500], 'tanh', bias=True)
    elif arch_id == 'layer-norm-width8':
        return layer_norm_net(dataset_name, [50], 'tanh', bias=True)
    elif arch_id == 'layer-norm-width10':
        return layer_norm_net(dataset_name, [2000], 'tanh', bias=True)

    # ======== batch normalization ========
    elif arch_id == 'batch-norm-width6':
        return batch_norm_net(dataset_name, [5000], 'tanh', bias=True)
    elif arch_id == 'batch-norm-width7':
        return batch_norm_net(dataset_name, [500], 'tanh', bias=True)
    elif arch_id == 'batch-norm-width8':
        return batch_norm_net(dataset_name, [50], 'tanh', bias=True)
    elif arch_id == 'batch-norm-width10':
        return batch_norm_net(dataset_name, [2000], 'tanh', bias=True)

    elif arch_id == 'nfc-tanh-width1':
        return nfc_net(dataset_name, [500], 'tanh', bias=True)
    elif arch_id == 'nfc-tanh-width8':
        return nfc_net(dataset_name, [50], 'tanh', bias=True)
    elif arch_id == 'nfc-tanh-width9':
        return nfc_net(dataset_name, [1], 'tanh', bias=True)

    elif arch_id == 'nfc-tanh-width8-regression':
        assert dataset_name.startswith("synthetic-")
        return nfc_net(dataset_name, [5], 'tanh', bias=True)
    