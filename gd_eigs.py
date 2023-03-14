from os import makedirs
import sys

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

import argparse

from archs import load_architecture
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset, get_hessian_eigenvalues_smallest
from data import load_dataset, take_first, DATASETS

from torch.optim.lr_scheduler import LambdaLR

def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, max_steps: int, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0, 
         eos_log: float=-1, param_save: float=-1):
    torch.device('cuda', 1)
    directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss, beta)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)
    makedirs(directory+"/params", exist_ok=True)
    makedirs(directory+"/grads", exist_ok=True)
    makedirs(directory+"/eos", exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta)

    #lambda1 = lambda i: 12500/(i+2500)
    #lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=-1)

    train_loss, test_loss, train_acc, test_acc, lr_iter = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    
    evec_start = 1000
    eigs_top = torch.zeros((max_steps - evec_start)//1000 if eig_freq >= 0 else 0, 20)
    evecs = torch.zeros((max_steps - evec_start)//1000 if eig_freq >= 0 else 0, len(parameters_to_vector(network.parameters())), 20)

    #print(lr.shape)
    #if save_model:
        #torch.save(network.state_dict(), f"{directory}/snapshot_init")
        #param_array = torch.nn.utils.parameters_to_vector(network.parameters())
    enter_eos = False
    count_down = 500
    save_grad_prev = False
    grad_list = []
    loss_eos = []
    l_eigs_eos = []
    s_eigs_eos = []

    for step in range(0, max_steps):
        #print(lr_scheduler.get_lr()[0])
        #lr_iter[step] = lr_scheduler.get_lr()[0]
        #lr_iter[step] = 0
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                           physical_batch_size)
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)
        
        if step >= 2519 and step < 2619:
            grad_vec = []
            for param in network.parameters():
                grad_vec.append(param.grad.view(-1))
            grad_vec = torch.cat(grad_vec).detach().cpu().numpy()
            loss_eos.append(train_loss[step].detach().numpy())
            #np.save(f"{directory}/eos/grad_{step}.npy", grad_vec)
            optimizer.zero_grad()
            s_eig_eos_5, _ = get_hessian_eigenvalues_smallest(network, loss_fn, lr, abridged_train, neigs=5,
                                                                physical_batch_size=physical_batch_size)
            print(s_eig_eos_5)
            optimizer.zero_grad()
            s_eig_eos_20, s_evec_eos = get_hessian_eigenvalues_smallest(network, loss_fn, lr, abridged_train, neigs=8,
                                                                physical_batch_size=physical_batch_size)
            print(s_eig_eos_20)
            s_eigs_eos.append(s_eig_eos_20.detach().float().numpy())
            #np.save(f"{directory}/eos/s_evec_{step}_20.npy", s_evec_eos.detach().float().numpy())
            count_down -= 1
            #np.save(f"{directory}/eos/smallest_eigs_eos_20.npy", s_eigs_eos)
                
        if step == 2618:
            sys.exit()

        if eig_freq != -1 and step % eig_freq == 0:
            eigs[step // eig_freq, :], evec, _, _ = get_hessian_eigenvalues(network, loss_fn, lr, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size)
            """
            if step >= evec_start and step % 1000 == 0:
                eig_top, evec, _, _ = get_hessian_eigenvalues(network, loss_fn, lr, abridged_train, neigs=20,
                                                                physical_batch_size=physical_batch_size)
                eigs_top[(step-evec_start) // 1000] = eig_top
                evecs[(step-evec_start) // 1000] = evec
            """
            print("eigenvalues: ", eigs[step//eig_freq, :])

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
                                   ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
                                   ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step]), ("lr", lr_iter[:step])])
        
        if save_freq != -1 and step >= evec_start and step % 1000 == 0:
            save_files(directory, [("eigs_top", eigs_top[:(step-evec_start)//1000]), ("evecs", evecs[:(step-evec_start)//1000])])

        print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")


        optimizer.zero_grad()
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            loss = loss_fn(network(X.cuda()), y.cuda()) / len(train_dataset)
            loss.backward()

        optimizer.step()
        #lr_scheduler.step()

    save_files_final(directory,
                     [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1]), ("lr", lr_iter[:step + 1])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov"],
                        help="which optimization algorithm to use", default="gd")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)")
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--eos_log", type=int, default=-1,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--param_save", type=int, default=-1,
                        help="if 'true', save model weights at end of training")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta=args.beta,
         nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size,
         seed=args.seed, eos_log=args.eos_log, param_save=args.param_save)
