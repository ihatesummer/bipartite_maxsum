# %%
from maxsum_mp import check_pairing_validity
from maxsum_rl import *
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

N_TEST = 10000
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
plt.style.use('seaborn-deep')


def main():
    check_cuda_device()
    _, _, w_ds = get_datasets()
    # D_mp = get_reference_mp(w_ds, n_iter=MAX_TIMESTEP)
    # np.save("d_mp_test.npy", D_mp)
    D_mp = np.load("d_mp_test.npy")
    D_hungarian = get_reference_hungarian(w_ds)

    pi_alpha, pi_rho = (Pi(2*N_NODE**2, N_NODE**2),
                        Pi(2*N_NODE**2, N_NODE**2))
    pi_alpha.eval()
    pi_rho.eval()

    if os.path.exists(FILENAMES["model_alpha"] and FILENAMES["model_rho"]):
        ckpt_alpha = torch.load(FILENAMES["model_alpha"])
        ckpt_rho = torch.load(FILENAMES["model_rho"])
        ckpt_general = torch.load(FILENAMES["general_info"])
        pi_alpha.model.load_state_dict(ckpt_alpha['model_state_dict'])
        pi_rho.model.load_state_dict(ckpt_rho['model_state_dict'])
        repeat_no_ckpt = ckpt_general['repeat_no'] + 1
        failure_count_mp = ckpt_general['failure_count_mp']
        print(f"Checkpoint up to repeat#{repeat_no_ckpt-1} loaded.")
    else:
        print("ERROR: no trained model found.")
        return

    ham_dist_list = np.zeros(N_TEST, dtype=int)
    sumrates = {"mp": np.zeros(N_TEST),
                "rl": np.zeros(N_TEST),
                "hungarian": np.zeros(N_TEST)}
    for i in range(N_TEST):
        # if i != 343:
        #     continue
        data_no = i + N_TRAIN
        # print(data_no)
        w = reshape_to_square(w_ds[data_no], N_NODE)

        sumrates["hungarian"][i] = np.sum(D_hungarian[data_no]*w)

        if check_pairing_validity(D_mp[data_no]):
            sumrates["mp"][i] = np.sum(D_mp[data_no]*w)
        else:
            sumrates["mp"][i] = 0
        
        # print(D_hungarian[data_no])
        alpha_rl, rho_rl = test_pi(pi_alpha, pi_rho, w_ds[data_no])
        # print(alpha_rl + rho_rl)
        D_rl = get_pairing_matrix_argmax(alpha_rl, rho_rl, N_NODE)
        # print(D_rl)
        if not check_pairing_validity(D_rl):
            D_rl = greedy_CA(alpha_rl+rho_rl, D_rl)
            # print("after greedy CA:")
            # print(D_rl)
        ham_dist = get_hamming_distance(D_rl, D_mp[data_no])
        # print(ham_dist)
        ham_dist_list[i] = ham_dist
        sumrates["rl"][i] = np.sum(D_rl*w)
    # print("Test set evaluation:")
    unique, counts = np.unique(ham_dist_list, return_counts=True)
    # print(dict(zip(unique, counts)))

    # plt.plot(sumrates["hungarian"][:1000], label="Hungarian", color='r', alpha=0.5)
    # plt.plot(sumrates["mp"][:1000], label="Ising", color='b', alpha=0.5)
    # plt.plot(sumrates["rl"][:1000], label="RL", color='g', alpha=0.5)
    bins = np.linspace(0, 5, 26)
    plt.hist([sumrates["hungarian"], sumrates["hungarian"], sumrates["rl"]],
             bins, label=["Hungarian", "Ising", "RL"])
    plt.legend()
    plt.xlim(0, 5)
    plt.title("Test set evaluations")
    plt.xlabel("sum-rate [bps]")
    plt.ylabel("number of samples")
    plt.savefig("tmp.png")


def test_pi(pi_alpha, pi_rho, w):
    alpha = torch.zeros((N_NODE**2))
    rho = torch.zeros((N_NODE**2))
    w_tensor = torch.from_numpy(w.astype(np.float32))
    for t in range(1, MAX_TIMESTEP):
        alpha_act, rho_act = (
            pi_alpha.act(torch.cat((rho, w_tensor))),
            pi_rho.act(torch.cat((alpha, w_tensor))))
        alpha, rho = alpha + alpha_act, rho + rho_act
        alpha = torch.clamp(alpha, -1, 1)
        rho = torch.clamp(rho, -1, 1)
    alpha_rl = reshape_to_square(alpha.detach().numpy(), N_NODE)
    rho_rl = reshape_to_square(rho.detach().numpy(), N_NODE)
    return alpha_rl, rho_rl


def greedy_CA(D_float, D):
    idx_taken = []
    for row in range(N_NODE):
        selection = np.argmax(D[row])
        if selection in idx_taken:
            taken_by = np.reshape(
                np.argwhere(D[:, selection]==1), (-1))[0]
            if D_float[row, selection] > D_float[taken_by, selection]:
                D[row, selection] = 1
                choices = np.delete(D_float[taken_by], idx_taken)
                chosen_idx = np.argwhere(D_float[taken_by]==np.max(choices))[0, 0]
                D[taken_by, selection] = 0
                D[taken_by, chosen_idx] = 1
            else:
                D[row, selection] = 0
                choices = np.delete(D_float[row], idx_taken)
                chosen_idx = np.argwhere(D_float[row]==np.max(choices))[0, 0]
                D[row, chosen_idx] = 1
            idx_taken.append(chosen_idx)
        else:
            idx_taken.append(selection)
    return D


if __name__=="__main__":
    main()
