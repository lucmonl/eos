from os import makedirs
import sys

import scipy
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

import argparse

from data import num_classes
from archs import load_architecture
"""
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset, \
    get_eig_grad, get_gauss_newton_eigenvalues, get_delta_c_eigenvalues, get_delta_c_c_eigenvalues, \
    get_fld_eigenvalues, get_gauss_newton_w_eigenvalues, get_gauss_newton_u_eigenvalues, get_gauss_newton_w_class_eigenvalues, \
    compute_gnvp_w_multiple, compute_gnvp_u_multiple
"""
from utilities import *
from data import load_dataset, take_first, DATASETS

from torch.optim.lr_scheduler import LambdaLR

def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, max_steps: int, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 1000, seed: int = 0, 
         eos_log: float=-1, param_save: float=-1, grad_step = 1, jacobian_sample_interval=1):
    torch.device('cuda', 1)
    directory = get_gd_directory_linear(dataset, lr, arch_id, seed, opt, loss, beta)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)
    makedirs(directory+"/params", exist_ok=True)
    makedirs(directory+"/grads", exist_ok=True)
    makedirs(directory+"/eos", exist_ok=True)
    makedirs(directory+"/model", exist_ok=True)

    num_class = num_classes(dataset)
    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()
    p = len(parameters_to_vector(network.parameters()))
    print("parameter:", p)
    print("class number:", num_class)
    for name, param in network.named_parameters():
        print(name, param.shape)
    w_shape = p - param.shape[0] * param.shape[1]
    #print("first layer parameter:", w_shape)
    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta)

    #lambda1 = lambda i: 12500/(i+2500)
    #lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=-1)

    train_loss, test_loss, train_acc, test_acc, lr_iter = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    loss_dv = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len(abridged_train))
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(parameters_to_vector(network.parameters())))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    evecs = torch.zeros(max_steps//eig_freq if eig_freq >= 0 else 0, len(parameters_to_vector(network.parameters())), neigs)
    weight_norm_l1 = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, 4)
    weight_norm_l2_l1 = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, 4)
    gn_eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    gn_eigs_w = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    gn_eigs_u = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    gn_evecs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len(abridged_train) * num_class, neigs)
    gn_evecs_w = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len(abridged_train) * num_class, neigs)
    gn_evecs_u = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len(abridged_train) * num_class, neigs)
    gn_eigs_w_class = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, num_class, neigs)
    jacobian_norm = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len(abridged_train) * num_class)
    jacobian = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, p, (len(abridged_train) // jacobian_sample_interval )* num_class)
    gn_evecs_top = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, p)
    #delta_c_eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, num_class-1)
    #delta_c_c_eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    #fld_eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, num_class-1)

    eigs_trim = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs, grad_step if grad_step >= 1 else 0)
    eigs_grad = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len(parameters_to_vector(network.parameters())))
    hessian_gradient_product = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len(abridged_train) // jacobian_sample_interval, len(parameters_to_vector(network.parameters())))
    grad_vecs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, len(parameters_to_vector(network.parameters())))
    #evec_start = 1000
    #eigs_top = torch.zeros((max_steps - evec_start)//1000 if eig_freq >= 0 else 0, 20)
    #evecs = torch.zeros((max_steps - evec_start)//1000 if eig_freq >= 0 else 0, len(parameters_to_vector(network.parameters())), 20)

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
        if step % (eig_freq // 5) == 0:
            print(train_loss[step])
        if eos_log != -1 and step > 0 and train_loss[step] > train_loss[step-1] and not save_grad_prev:
            np.save(f"{directory}/eos/grad_prev.npy", np.array(grad_list))
            enter_eos = True
            save_grad_prev = True
        
        if eig_freq != -1 and step % eig_freq == 0:
            #gauss_newton_matrix_u = get_gauss_newton_matrix_u(network, abridged_train, num_class = num_class, w_shape = w_shape)
            #save_files(directory, [("gauss_newton_matrix_u", gauss_newton_matrix_u.cpu())])
            #sys.exit()
            #gn_eigs_u_matrix = torch.FloatTensor(scipy.linalg.eigh(gauss_newton_matrix_u.cpu().numpy(), eigvals=(gauss_newton_matrix_u.shape[0]-neigs,gauss_newton_matrix_u.shape[0]-1))[1]).cuda()
            grad_vecs[step // eig_freq, :], loss_dv[step // eig_freq, :] = get_gradient(network, loss_fn, abridged_train, physical_batch_size)
            eigs[step // eig_freq, :], evec, _, _ = get_hessian_eigenvalues(network, loss_fn, lr, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size, return_smallest=False)
            evecs[step // eig_freq] = evec
            hessian_gradient_product[step // eig_freq] = compute_hessian_grad_product(network, loss_fn, abridged_train, grad_vecs[step // eig_freq],
                                                         physical_batch_size, sample_interval=jacobian_sample_interval)
            #gn_eigs[step // eig_freq, :], gn_evecs = get_gauss_newton_eigenvalues(network, abridged_train, neigs=neigs, num_class = num_class)
            #gn_eigs_w[step // eig_freq, :], gn_eigs_w_evec = get_gauss_newton_w_eigenvalues(network, abridged_train, neigs=neigs, num_class = num_class, w_shape = w_shape)
            #gn_eigs_u[step // eig_freq, :], gn_eigs_u_evec = get_gauss_newton_u_eigenvalues(network, abridged_train, neigs=neigs, num_class = num_class, w_shape = w_shape)
            #jacobian_norm[step // eig_freq, :] = compute_jacobian_norm(network, abridged_train, num_class)
            jacobian[step // eig_freq] = compute_jacobian(network, abridged_train, num_class, sample_interval=jacobian_sample_interval)
            #gn_evecs_top[step // eig_freq, :] = gn_evecs[:, 0].cpu()
            #print(gn_eigs_u[step // eig_freq, :])
            #save_files(directory, [("gn_evec_u_mat_{}".format(step), gn_eigs_u_matrix.cpu())])
            """
            save_files(directory, [("gauss_newton_mat_u_{}".format(step), gauss_newton_matrix_u)])
            save_files(directory, [("gn_evec_u_op_{}".format(step), gn_eigs_u_evec)])
            save_files(directory, [("gn_evec_u_mat_{}".format(step), gn_eigs_u_matrix.cpu())])
            
            def cosine_similarity(u ,v):
                return u@v / torch.norm(u) / torch.norm(v)
            for i in range(10):
                print(i)
                gnvp_u = compute_gnvp_u(network, abridged_train, gn_eigs_u_evec[:,i], num_class, w_shape)
                print(cosine_similarity(gnvp_u, gn_eigs_u_evec[:,i].cuda()))
                gnvp_u = compute_gnvp_u(network, abridged_train, gn_eigs_u_matrix[:,i], num_class, w_shape)
                print(cosine_similarity(gnvp_u, gn_eigs_u_matrix[:,i]))
                gnvp_u = gauss_newton_matrix_u @ gn_eigs_u_evec[:,i].cuda()
                print(cosine_similarity(gnvp_u, gn_eigs_u_evec[:,i].cuda()))
                gnvp_u = gauss_newton_matrix_u @ gn_eigs_u_matrix[:,i]
                print(cosine_similarity(gnvp_u, gn_eigs_u_matrix[:,i]))
            """
            #sys.exit()
            #print(torch.where(torch.abs(gn_eigs_u_evec[:, 0]) <= 1e-5)[0].shape)
            #print(torch.where(torch.abs(gn_eigs_u_evec[:, 5]) <= 1e-5)[0].shape)
            #print(torch.where(torch.abs(gn_eigs_u_evec[:, 7]) <= 1e-5)[0].shape)
            #print(torch.where(torch.abs(gn_eigs_u_evec[:, 9]) <= 1e-5)[0].shape)
            
            #gn_evecs[step // eig_freq] = compute_gnvp_multiple(network, abridged_train, gn_eigs_evec, num_class)
            #gn_evecs_w[step // eig_freq] = compute_gnvp_w_multiple(network, abridged_train, gn_eigs_w_evec, num_class, w_shape)
            #gn_evecs_u[step // eig_freq] = compute_gnvp_u_multiple(network, abridged_train, gn_eigs_u_evec, num_class, w_shape)
            
            #gn_eigs_w_class[step // eig_freq, :, :], _ = get_gauss_newton_w_class_eigenvalues(network, abridged_train, neigs=neigs, num_class = num_class, w_shape = w_shape)
            #delta_c_eigs[step // eig_freq, :], _ = get_delta_c_eigenvalues(network, abridged_train, neigs=num_class-1, num_class = num_class)
            #fld_eigs[step // eig_freq, :], _ = get_fld_eigenvalues(network, abridged_train, neigs=num_class-1, num_class = num_class)
            #delta_c_c_eigs[step // eig_freq, :], _ = get_delta_c_c_eigenvalues(network, abridged_train, neigs=neigs)
            #
            #eigs_grad[step // eig_freq] = get_eig_grad(network, loss_fn, abridged_train, eig_vec = evec[:,0], physical_batch_size=physical_batch_size)
            """
            param_id = 0
            for name, param in network.named_parameters():
                weight_norm_l1[step // eig_freq, param_id] = torch.norm(param, p=1)
                weight_norm_l2_l1[step // eig_freq, param_id] = torch.norm(param, p=2) / torch.norm(param, p=1)
                param_id += 1
            """
            """
            if step >= evec_start and step % 1000 == 0:
                eig_top, evec, _, _ = get_hessian_eigenvalues(network, loss_fn, lr, abridged_train, neigs=20,
                                                                physical_batch_size=physical_batch_size)
                eigs_top[(step-evec_start) // 1000] = eig_top
                evecs[(step-evec_start) // 1000] = evec
            """
            print("hessian eigenvalues: ", eigs[step//eig_freq, :])
            #print("gauss-newton eigenvalues: ", gn_eigs[step//eig_freq, :])
            #print("decomp eigenvalues: ", delta_c_eigs[step//eig_freq, :])
            #print("fld eigenvalues: ", fld_eigs[step//eig_freq, :])
            #print("eigenvalues: ", delta_c_c_eigs[step//eig_freq, :])
            #save_files(directory, [("eig_grad", eigs_grad[:step//eig_freq+1])])
        
        if iterate_freq != -1 and step % iterate_freq == 0:
            #iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())
            iterates[step // iterate_freq, :] = parameters_to_vector(network.parameters()).cpu().detach()

        if save_freq != -1 and step % save_freq == 0:
            print("saving")
            
            save_files(directory, [#("eigs", eigs[:step // eig_freq]), 
                                   #("iterates", iterates[:step // iterate_freq]),
                                   #("evecs", evecs[:step // eig_freq]),
                                   ("loss_derivative", loss_dv[:step // eig_freq]),
                                   ("grad_vecs", grad_vecs[:step // eig_freq]),
                                   #("hessian_grad_product", hessian_gradient_product[:step // eig_freq]),
                                   #("train_loss", train_loss[:step]), #("test_loss", test_loss[:step]),
                                   #("gauss_newton_eigs_w_class", gn_eigs_w_class[:step // eig_freq]),
                                   #("gauss_newton_eigs_w", gn_eigs_w[:step // eig_freq]),
                                   #("gauss_newton_eigs_u", gn_eigs_u[:step // eig_freq]),
                                   #("gauss_newton_eigs", gn_eigs[:step // eig_freq]), 
                                   #("gauss_newton_evecs", gn_evecs[:step // eig_freq]),
                                   #("gauss_newton_evecs_w", gn_evecs_w[:step // eig_freq]),
                                   #("gauss_newton_evecs_u", gn_evecs_u[:step // eig_freq]),
                                   #("jacobian_norm", jacobian_norm[:step // eig_freq]),
                                   #("jacobian_{}".format(eig_freq), jacobian[:step // eig_freq]),
                                   #("gn_evecs_top_{}".format(eig_freq), gn_evecs_top[:step // eig_freq]),
                                   #("delta_c_eigs", delta_c_eigs[:step // eig_freq]),
                                   #("fld_eigs", fld_eigs[:step // eig_freq]),
                                   #("delta_c_c_eigs", delta_c_c_eigs[:step // eig_freq]),
                                   #("weight_norm_l1", weight_norm_l1[:step // eig_freq]),
                                   #("weight_norm_l2_l1", weight_norm_l2_l1[:step // eig_freq]),
                                   #("train_acc", train_acc[:step]), ("test_acc", test_acc[:step]), ("lr", lr_iter[:step])
                                   ])
            
            torch.save(network.state_dict(), f"{directory}/model/network_{step}.pth")
        
        #if save_freq != -1 and step >= evec_start and step % 1000 == 0:
        #    save_files(directory, [("eigs_top", eigs_top[:(step-evec_start)//1000]), ("evecs", evecs[:(step-evec_start)//1000])])

        #print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        #if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
        if (loss_goal != None and train_loss[step] < loss_goal): # or (acc_goal != None and train_acc[step] > acc_goal):
            param_vec = []
            for param in network.parameters():
                param_vec.append(param.view(-1))
            param_vec = torch.cat(param_vec).detach().cpu().numpy()
            np.save(f"{directory}/params/param_final.npy", param_vec)
            break

        optimizer.zero_grad()
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            #loss = loss_fn(network(X.cuda()), y.cuda()) / len(train_dataset)
            loss = compute_loss_linear(network, loss_fn, X.cuda(), y.cuda()) / len(train_dataset)
            #print(loss_1, loss)
            if grad_step != -1 and eig_freq != -1 and step % eig_freq == 0:
                (loss / grad_step).backward()
            else:
                loss.backward()
        #print(loss)
        #for name, param in network.named_parameters():
        #    print(name, param.grad)

        #print("=========")
        """
        optimizer.zero_grad()
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            loss = loss_fn(network(X.cuda()), y.cuda()) / len(train_dataset)
            #loss = compute_loss_linear(network, loss_fn, X.cuda(), y.cuda()) / len(train_dataset)
            #print(loss_1, loss)
            if grad_step != -1 and eig_freq != -1 and step % eig_freq == 0:
                (loss / grad_step).backward()
            else:
                loss.backward()
        print(loss)
        #for name, param in network.named_parameters():
        #    print(name, param.grad)
        """
        
        if step > 0 and eig_freq != -1 and step % eig_freq == 0:
            grad_vec = []
            for param in network.parameters():
                grad_vec.append(param.grad.view(-1))
            grad_vec = torch.cat(grad_vec).detach().cpu().numpy()
            loss_eos.append(train_loss[step].detach().numpy())
            np.save(f"{directory}/eos/grad_{step}.npy", grad_vec)
        
        if eos_log != -1 and count_down != 0 and not enter_eos:
            grad_vec = []
            for param in network.parameters():
                grad_vec.append(param.grad.view(-1))
            grad_vec = torch.cat(grad_vec).detach().cpu().numpy()
            if len(grad_list) > 40:
                grad_list = grad_list[1:]
                loss_eos = loss_eos[1:]
            grad_list.append(grad_vec)
            loss_eos.append(train_loss[step].detach().numpy())
        
        if eos_log != -1 and enter_eos and count_down > 0:
            grad_vec = []
            for param in network.parameters():
                grad_vec.append(param.grad.view(-1))
            grad_vec = torch.cat(grad_vec).detach().cpu().numpy()
            loss_eos.append(train_loss[step].detach().numpy())
            np.save(f"{directory}/eos/grad_{step}.npy", grad_vec)

            l_eig_eos, l_evec_eos, s_eig_eos, s_evec_eos = get_hessian_eigenvalues(network, loss_fn, lr, abridged_train, neigs=5,
                                                                physical_batch_size=physical_batch_size)
            l_eigs_eos.append(l_eig_eos.detach().float().numpy())
            s_eigs_eos.append(s_eig_eos.detach().float().numpy())
            np.save(f"{directory}/eos/l_evec_{step}.npy", l_evec_eos.detach().float().numpy())
            np.save(f"{directory}/eos/s_evec_{step}.npy", s_evec_eos.detach().float().numpy())
            count_down -= 1

            np.save(f"{directory}/eos/largest_eigs_eos.npy", l_eigs_eos)
            np.save(f"{directory}/eos/smallest_eigs_eos.npy", s_eigs_eos)
            np.save(f"{directory}/eos/loss_eos.npy", loss_eos)
        
        if eos_log != -1 and enter_eos and count_down == 0:
            np.save(f"{directory}/eos/largest_eigs_eos.npy", l_eigs_eos)
            np.save(f"{directory}/eos/smallest_eigs_eos.npy", s_eigs_eos)
            np.save(f"{directory}/eos/loss_eos.npy", loss_eos)
            enter_eos = False
            sys.exit()

        if param_save != -1 and step % eig_freq ==  0:
            param_vec = []
            grad_vec = []
            for param in network.parameters():
                param_vec.append(param.view(-1))
                grad_vec.append(param.grad.view(-1))
            param_vec = torch.cat(param_vec).detach().cpu().numpy()
            grad_vec = torch.cat(grad_vec).detach().cpu().numpy()
            np.save(f"{directory}/params/param_{step}.npy", param_vec)
            np.save(f"{directory}/grads/grad_{step}.npy", grad_vec)

        if grad_step != -1 and eig_freq != -1 and step % eig_freq == 0:
            for j in range(grad_step):
                eigs_trim[step // eig_freq, :, j], _, _, _ = get_hessian_eigenvalues(network, loss_fn, lr, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size, return_smallest=False)
                optimizer.step()
            save_files_final(directory, [("eigs_trim", eigs_trim[:(step + 1) // eig_freq])])
        else:
            optimizer.step()
        #lr_scheduler.step()
    """
    save_files_final(directory,
                     [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1]), ("lr", lr_iter[:step + 1])])
    """
    
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
    parser.add_argument("--abridged_size", type=int, default=1000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--eos_log", type=int, default=-1,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--param_save", type=int, default=-1,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--grad_step", type=int, default=-1,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--jacobian_interval", type=int, default=1,
                        help="Sample interval (on data points) for jacobian matrix")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta=args.beta,
         nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size,
         seed=args.seed, eos_log=args.eos_log, param_save=args.param_save, grad_step = args.grad_step, 
         jacobian_sample_interval=args.jacobian_interval)
