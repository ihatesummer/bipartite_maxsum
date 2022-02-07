# %%
from maxsum_mp import check_pairing_validity
from maxsum_rl import *
import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt

N_TEST = 10
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
plt.style.use('seaborn-deep')


def main():
    check_cuda_device()
    _, _, w_ds = get_datasets()
    D_mp = np.load(FILENAMES["D_mp"])
    alpha_mp = np.load(FILENAMES["alpha_mp"])
    rho_mp = np.load(FILENAMES["rho_mp"])
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
    rl_times = np.zeros(N_TEST)
    for i in range(N_TEST):
        # if i != 1237:
        #     continue
        data_no = i + N_TRAIN
        w = reshape_to_square(w_ds[data_no], N_NODE)

        sumrates["hungarian"][i] = np.sum(D_hungarian[data_no]*w)

        if not check_pairing_validity(D_mp[data_no]):
            D_mp[data_no] = greedy_CA(alpha_mp[data_no]+rho_mp[data_no], D_mp[data_no])
            # D_mp[data_no] = exclude_collision(D_mp[data_no])
        sumrates["mp"][i] = np.sum(D_mp[data_no]*w)

        tic_rl = time.time()
        alpha_rl, rho_rl = test_pi(pi_alpha, pi_rho, w_ds[data_no])
        D_rl = get_pairing_matrix_argmax(alpha_rl, rho_rl, N_NODE)
        rl_times[i] = time.time() - tic_rl
        if not check_pairing_validity(D_rl):
            D_rl = greedy_CA(alpha_rl+rho_rl, D_rl)
            # D_rl = exclude_collision(D_rl)
        sumrates["rl"][i] = np.sum(D_rl*w)
        ham_dist = get_hamming_distance(D_rl, D_mp[data_no])
        ham_dist_list[i] = ham_dist
    unique, counts = np.unique(ham_dist_list, return_counts=True)
    print("MP-RL hamming distance:")
    print(dict(zip(unique, counts)))
    print(f"Avg sum rate (Hungarian): {np.mean(sumrates['hungarian'])}")
    print(f"Avg sum rate (MP): {np.mean(sumrates['mp'])}")
    print(f"Avg sum rate (RL): {np.mean(sumrates['rl'])}")
    print(f"Avg time (RL): {np.mean(rl_times)}")

    # bins = np.linspace(0, 5, 26)
    # plt.hist([sumrates["hungarian"], sumrates["mp"], sumrates["rl"]],
    #          bins, label=["Hungarian", "Ising", "RL"])
    # plt.legend()
    # plt.xlim(0, 5)
    # plt.title("Test set evaluations")
    # plt.xlabel("sum-rate [bps]")
    # plt.ylabel("number of samples")
    # plt.savefig("tmp.png")


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
    random_order = np.random.permutation(N_NODE)
    for row in random_order:
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


def exclude_collision(D):
    idx_taken = []
    random_order = np.random.permutation(N_NODE)
    for row in random_order:
        selection = np.argmax(D[row])
        if selection in idx_taken:
            D[row, selection] = 0
        idx_taken.append(selection)
    return D



if __name__=="__main__":
    main()
