import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import normalize
import scipy
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

device = torch.device("cpu")

#from torch.nn.utils import weight_norm
import torch.optim as optim

from cifar import *

def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])

DATASETS_FOLDER = "../../data/"
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

train_dataset, test_dataset = load_cifar("mse")

dataset_num = 512
train_dataset = take_first(train_dataset, dataset_num)
trainloader = DataLoader(train_dataset, batch_size=512,
                                          shuffle=False)
for X, y in trainloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
num_pixels = 3 * 32** 2

abridged_data = take_first(train_dataset, 4)

class SquaredLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return 0.5 * ((input - target) ** 2).sum()
    
class resnet(nn.Module):
    def __init__(self, dataset_name, widths, init_mode, basis_var, scale):
        super().__init__()
        self.scale = scale
        self.depth = len(widths)
        self.basis_std = np.sqrt(basis_var)
        self.flatten = nn.Flatten()
        self.module_list = nn.ModuleList()
        for l in range(len(widths)-1):
            prev_width = widths[l]
            self.module_list.append(nn.Linear(prev_width, widths[l+1], bias=False))
        self.input_layer = nn.Linear(num_pixels, widths[0], bias=False)
        self.output_layer = nn.Linear(widths[-1],10,bias=False)
        self.__initialize__(widths, init_mode)
    
    def __initialize__(self, widths, init_mode):
        for l in range(len(widths)-1):
            prev_width = widths[l]
            if init_mode == "O(1/sqrt{m})":
                nn.init.normal_(self.module_list[l].weight, mean=0, std=self.basis_std/np.sqrt(prev_width))
            elif init_mode == "O(1)":
                nn.init.normal_(self.module_list[l].weight, mean=0, std=self.basis_std)
            else:
                assert False
        nn.init.normal_(self.input_layer.weight, mean=0, std=1/np.sqrt(num_pixels))
        #self.input_layer.weight.data = torch.nn.functional.normalize(self.output_layer.weight, dim=-1)
        nn.init.normal_(self.output_layer.weight, mean=0, std=1/np.sqrt(widths[-1]))
        self.output_layer.weight.data = torch.nn.functional.normalize(self.output_layer.weight, dim=-1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        for linear in self.module_list:
            width = x.shape[-1]
            #temp_x = torch.FloatTensor(x.detach().numpy())
            #x = torch.nn.functional.normalize(x, dim=-1)
            identity = torch.clone(x)
            x = linear(x)
            #for i in range(x.shape[0]):
            #    x[i] = x[i] / torch.norm(temp_x[i])
            x = self.scale * x / np.sqrt(width) / self.depth
            #x = linear(x)
            x = identity + torch.tanh(x)
        width = x.shape[-1]
        #print(torch.norm(self.output_layer.weight))
        x = self.output_layer(x)# / torch.norm(self.output_layer.weight)
        x = self.scale * x / np.sqrt(width)
        return x
    
def create_key(keys, dict_input):
    cur_dict = dict_input
    for key in keys:
        if key not in cur_dict:
            cur_dict[key] = {}
        cur_dict = cur_dict[key]
    return cur_dict

def compute_jacobian_norm(network: nn.Module, trainloader: DataLoader, num_class: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(trainloader.dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    jacobian = torch.zeros((p, num_class), dtype=torch.float, device='cpu')
    sample_id = 0
    #for (X, _) in iterate_dataset(dataset, 1):
    for _, data in enumerate(trainloader, 0):
        inputs, _ = data
        predictor = torch.sum(network(inputs), dim=0)/n
        assert predictor.shape[0] == num_class
        for i in range(predictor.shape[0]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[i], network.parameters(), retain_graph = True))
            jacobian[:,i] += grads_i
    jacobian_norm =torch.norm(jacobian, dim=0)
    return jacobian_norm

def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    #print(device)
    for (batch_X, batch_y) in loader:
        yield batch_X.to(device), batch_y.to(device)

def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = 512):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        #print(loss)
        #for param in network.parameters():
        #    print(param.data)
        """
        param_list = []
        for param in network.parameters():
            param_list.append(param)
            assert param.requires_grad
        beta = torch.square(param_list[0]) - torch.square(param_list[1])
        loss = 0.25*torch.mean((X@beta-y)**2)
        """
        #loss = loss_fn(network(X), y) / n
        #loss = 0.25 * torch.mean((network(X).squeeze()-y.squeeze())**2)
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp

def compute_gnvp(network: nn.Module, dataset: Dataset, vector: Tensor, num_class: int):
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    gnvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    pred_grad = torch.zeros((p, num_class), dtype=torch.float, device=device)
    for (X, _) in iterate_dataset(dataset, 1):
        predictor = network(X)
        for i in range(predictor.shape[1]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))
            pred_grad[:,i] = grads_i
        gnvp += pred_grad @ (pred_grad.T @ vector) / n
    return gnvp

def compute_hfvp(network: nn.Module, dataset: Dataset, vector: Tensor, num_class: int, index=0, physical_batch_size: int=512):
    assert index < num_class
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    #gnvp = torch.zeros(p, dtype=torch.float, device=device)
    hfvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)
    #pred_grad = torch.zeros((p, num_class), dtype=torch.float, device=device)
    for (X, _) in iterate_dataset(dataset, physical_batch_size):
        predictor = network(X)
        """
        for i in range(predictor.shape[1]):
            grads_i = parameters_to_vector(torch.autograd.grad(predictor[0, i], network.parameters(), retain_graph = True))
            pred_grad[:,i] = grads_i
        """
        output = torch.sum(predictor[:, index])
        print("first backward")
        grads = parameters_to_vector(torch.autograd.grad(output, network.parameters(), create_graph = True))
        print("complete")
        dot = parameters_to_vector(grads).mul(vector).sum()
        print("second backward")
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        print("complete")
        #gnvp += pred_grad @ (pred_grad.T @ vector) / n
        hfvp += parameters_to_vector(grads) / n
    return hfvp

def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(device)
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    l_evals, l_evecs = eigsh(operator, neigs)
    #s_evals, s_evecs= eigsh(operator, neigs, which='SM')
    return torch.from_numpy(np.ascontiguousarray(l_evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(l_evecs, -1)).copy()).float()
           #torch.from_numpy(np.ascontiguousarray(s_evals[::-1]).copy()).float(), \
           #torch.from_numpy(np.ascontiguousarray(np.flip(s_evecs, -1)).copy()).float()


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, lr: float, dataset: Dataset,
                            neigs=6, physical_batch_size=1000, return_smallest = False):
    #vector_test = torch.ones(200)
    #print(compute_hvp(network, loss_fn, dataset, vector_test, physical_batch_size=physical_batch_size))
    #sys.exit()
    """ Compute the leading Hessian eigenvalues. """
    alpha = 4 / lr
    s_evals, s_evecs = 0, 0
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size).detach().cpu()
    if return_smallest == True:
        hvp_delta_small = lambda delta: compute_hvp_smallest(network, loss_fn, alpha, dataset,
                                            delta, physical_batch_size=physical_batch_size).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    if return_smallest == True:
        s_evals, s_evecs = lanczos(hvp_delta_small, nparams, neigs=neigs)
    return l_evals, l_evecs, alpha - s_evals, s_evecs

def get_gauss_newton_eigenvalues(network: nn.Module, dataset: Dataset, neigs=6, num_class=10):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_gnvp(network, dataset, delta, num_class).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return l_evals, l_evecs

def get_hf_eigenvalues(network: nn.Module, dataset: Dataset, neigs=6, num_class=10, physical_batch_size=1000):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hfvp(network, dataset, delta, num_class, index=0, physical_batch_size=physical_batch_size).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    l_evals, l_evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return l_evals, l_evecs

def one_run(grad_loss_ratio, jacob_norm, weight_min, loss_list, heigs_list, gneigs_list, hfeigs_list, model, init_mode, basis_var, scale, lr_mode, basis_lr):
    create_key([model, init_mode, basis_var, scale, lr_mode, basis_lr], grad_loss_ratio)
    create_key([model, init_mode, basis_var, scale, lr_mode, basis_lr], jacob_norm)
    create_key([model, init_mode, basis_var, scale, lr_mode, basis_lr], weight_min)
    create_key([model, init_mode, basis_var, scale, lr_mode, basis_lr], loss_list)
    create_key([model, init_mode, basis_var, scale, lr_mode, basis_lr], heigs_list)
    create_key([model, init_mode, basis_var, scale, lr_mode, basis_lr], gneigs_list)
    create_key([model, init_mode, basis_var, scale, lr_mode, basis_lr], hfeigs_list)
    for width in [512, 1024, 2048]:
        print(width)
        grad_loss_ratio[model][init_mode][basis_var][scale][lr_mode][basis_lr][width], jacob_norm[model][init_mode][basis_var][scale][lr_mode][basis_lr][width], weight_min[model][init_mode][basis_var][scale][lr_mode][basis_lr][width], loss_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width], heigs_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width], gneigs_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width] , hfeigs_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width] = [], [], [], [], [], [], []
        for running_time in range(3):
            print("run: ", running_time)
            grad_loss_ratio[model][init_mode][basis_var][scale][lr_mode][basis_lr][width].append([])
            jacob_norm[model][init_mode][basis_var][scale][lr_mode][basis_lr][width].append([])
            weight_min[model][init_mode][basis_var][scale][lr_mode][basis_lr][width].append([])
            loss_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width].append([])
            heigs_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width].append([])
            gneigs_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width].append([])
            hfeigs_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width].append([])
            if model == "resnet":
                net = resnet("cifar10", [width,width], init_mode, basis_var, scale)
            elif model == "ff":
                net = feed_forward_net("cifar10", [width,width], init_mode, basis_var, scale)
            else:
                assert False
            criterion = SquaredLoss()
            if lr_mode == "constant":
                optimizer = optim.SGD(net.parameters(), lr=basis_lr)
            elif lr_mode == "varying":
                optimizer = optim.SGD(net.parameters(), lr=basis_lr*(width//512))
            else:
                assert False
            for epoch in range(3001):  # loop over the dataset multiple times
                running_loss = 0.0
                optimizer.zero_grad()
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    outputs = net(inputs)
                    loss = criterion(outputs, labels) / dataset_num
                    loss.backward()
                    running_loss += loss.item()
                optimizer.step()
                if epoch % 100 == 0:
                    grad_norm = 0
                    for param in net.parameters():
                        grad_norm += torch.norm(param.grad)**2
                    jacobian_norm = compute_jacobian_norm(net, trainloader, num_class = 10)
                    
                    heigs, _, _, _ = get_hessian_eigenvalues(net, criterion, basis_lr, abridged_data, neigs=1)
                    gneigs, _ = get_gauss_newton_eigenvalues(net, abridged_data, neigs=1)
                    print("into hf")
                    hfeigs, _ = get_hf_eigenvalues(net, abridged_data, neigs=1, num_class=10)
                    
                    norm_min = 1e10
                    for param in net.parameters():
                        if param.shape[0] == 10:
                            print("exit")
                            break
                        norm_min = min(norm_min, torch.min(torch.norm(param, dim=-1)).detach().numpy())
                    print(f'[Epoch: {epoch + 1}] loss: {running_loss:.3f} heigs: {float(heigs[0])} gneigs: {float(gneigs[0])} hfeigs: {float(hfeigs[0])} jacobian norm: {jacobian_norm[0]} norm min: {norm_min} grad_loss_ratio: {grad_norm / running_loss}')
                    grad_loss_ratio[model][init_mode][basis_var][scale][lr_mode][basis_lr][width][-1].append(float(grad_norm / running_loss))
                    jacob_norm[model][init_mode][basis_var][scale][lr_mode][basis_lr][width][-1].append(jacobian_norm[0].item())
                    weight_min[model][init_mode][basis_var][scale][lr_mode][basis_lr][width][-1].append(float(norm_min))
                    loss_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width][-1].append(running_loss)
                    heigs_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width][-1].append(float(heigs[0]))
                    gneigs_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width][-1].append(float(gneigs[0]))
                    hfeigs_list[model][init_mode][basis_var][scale][lr_mode][basis_lr][width][-1].append(float(hfeigs[0]))
    print('Finished Training')

grad_loss_ratio, jacob_norm, weight_min, loss_list, heigs_list, gneigs_list, hfeigs_list = {}, {}, {}, {}, {}, {}, {}

one_run(grad_loss_ratio, jacob_norm, weight_min, loss_list, heigs_list, gneigs_list, hfeigs_list, model="resnet", init_mode="O(1/sqrt{m})", basis_var=5, scale = 1, lr_mode="constant", basis_lr=0.01)