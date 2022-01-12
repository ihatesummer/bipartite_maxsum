# %%
from maxsum import check_validity, get_pairing_matrix_argmax, get_pairing_matrix_sign, reshape_to_flat, reshape_to_square
from maxsum_ul import get_datasets, forward_pass, decompose_dataset, FILENAMES, N_DATASET, N_NODE, SEED_W
from maxsum_rl import Pi, plot_positions
import numpy as np
import torch

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
GAMMA = 1.0
N_TRAIN = 1
N_TEST = 1
MAX_TIMESTEP = 5
FILENAMES["nn_weight_alpha"] = "weights_rl_alpha.h5"
FILENAMES["nn_weight_rho"] = "weights_rl_rho.h5"


def main():
    pos_bs, pos_user, w, alpha_star, rho_star = get_datasets(
        FILENAMES, N_DATASET, SEED_W, N_NODE, "geographic")
    dim_in = 2*N_NODE**2
    dim_out = N_NODE**2
    pi_alpha = Pi(dim_in, dim_out)
    pi_rho = Pi(dim_in, dim_out)
    pi_alpha.model.load_state_dict(torch.load(FILENAMES["nn_weight_alpha"]))
    pi_rho.model.load_state_dict(torch.load(FILENAMES["nn_weight_rho"]))
    print("Trained model found.")
    for data_no in range(N_TEST):
        print(f"Train set {data_no} weights:\n{reshape_to_square(w[data_no], N_NODE)}")
        plot_positions(pos_bs[data_no], pos_user[data_no], data_no)
        D_mp = get_pairing_matrix_argmax(
            reshape_to_square(alpha_star[data_no], N_NODE),
            reshape_to_square(rho_star[data_no], N_NODE), N_NODE)
        D_rl = evaluate_policy(pi_alpha, pi_rho, w[data_no])
        compare_pairings(D_rl, D_mp)


def evaluate_policy(pi_alpha, pi_rho, w):
    alpha = torch.zeros((N_NODE**2))
    rho = torch.zeros((N_NODE**2))
    w = torch.tensor(w).float()
    for t in range(MAX_TIMESTEP):
        action_alpha = pi_alpha.act(torch.cat((alpha, w)))
        action_rho = pi_rho.act(torch.cat((rho, w)))
        alpha = alpha + action_alpha
        rho = rho + action_rho
    pi_alpha.onpolicy_reset()
    pi_rho.onpolicy_reset()
    alpha_rl = reshape_to_square(alpha.detach().numpy(), N_NODE)
    rho_rl = reshape_to_square(rho.detach().numpy(), N_NODE)
    D_rl = get_pairing_matrix_argmax(alpha_rl, rho_rl, N_NODE)
    print(f"alpha_rl:\n{alpha_rl}")
    print(f"rho_rl:\n{rho_rl}")
    print(f"alpha+rho (RL):\n{alpha_rl+rho_rl}")
    print(f"D (RL):\n{D_rl}")
    print(f"Pairing validity (RL): {check_validity(D_rl)}")
    return D_rl


def compare_pairings(D_rl, D_mp):
    diff_count=int(np.sum(abs(D_rl - D_mp))/2)
    print(f"{diff_count} pairings are different than the optimum.")


if __name__ == '__main__':
    main()
