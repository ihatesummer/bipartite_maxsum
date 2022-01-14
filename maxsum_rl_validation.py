# %%
from maxsum import check_validity, get_pairing_matrix_argmax, reshape_to_square
from maxsum_ul import get_datasets, forward_pass, FILENAMES, N_DATASET, N_NODE, SEED_W
from maxsum_rl import Pi, plot_positions, plot_trajectories, get_hamming_distance
import numpy as np
import torch
import os

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
MAX_TIMESTEP = 10
FILENAMES["model_alpha"] = "model_alpha.h5"
FILENAMES["model_rho"] = "model_rho.h5"


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    pos_bs, pos_user, w, alpha_star, rho_star = get_datasets(
        FILENAMES, N_DATASET, SEED_W, N_NODE, "geographic")
    dim_in = 2*N_NODE**2
    dim_out = N_NODE**2
    pi_alpha = Pi(dim_in, dim_out)
    pi_rho = Pi(dim_in, dim_out)

    if os.path.exists(FILENAMES["model_alpha"] and FILENAMES["model_rho"]):
        print("Loading train checkpoint...")
        ckpt_alpha = torch.load(FILENAMES["model_alpha"])
        ckpt_rho = torch.load(FILENAMES["model_rho"])
        pi_alpha.model.load_state_dict(ckpt_alpha['model_state_dict'])
        pi_rho.model.load_state_dict(ckpt_rho['model_state_dict'])
        data_no_ckpt = ckpt_rho['data_no']
        print(f"Checkpoint up to data {data_no_ckpt} loaded.")

    # idx to check
    data_no = 1
    print(f"Train set {data_no} weights:\n{reshape_to_square(w[data_no], N_NODE)}")
    plot_positions(pos_bs[data_no], pos_user[data_no], data_no)
    D_mp = get_pairing_matrix_argmax(
        reshape_to_square(alpha_star[data_no], N_NODE),
        reshape_to_square(rho_star[data_no], N_NODE), N_NODE)
    print(f"D(mp):\n{D_mp}")

    alpha = torch.zeros((N_NODE**2))
    rho = torch.zeros((N_NODE**2))
    w_tensor = torch.from_numpy(w[data_no].astype(np.float32))

    alpha_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    rho_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    alpha_target_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    rho_target_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    alpha_hist[0] = alpha.detach().numpy()
    rho_hist[0] = rho.detach().numpy()
    alpha_target_hist[0] = alpha.detach().numpy()
    rho_target_hist[0] = rho.detach().numpy()
    for t in range(MAX_TIMESTEP):
        alpha_rho = torch.cat((alpha, rho)).detach().numpy().reshape(1, -1)
        alpha_rho_target = forward_pass(w[data_no], alpha_rho)
        alpha_target, rho_target = np.array_split(alpha_rho_target, 2)
        alpha_target = torch.from_numpy(alpha_target.astype(np.float32))
        rho_target = torch.from_numpy(rho_target.astype(np.float32))

        alpha_target_hist[t] = alpha_target.detach().numpy()
        rho_target_hist[t] = rho_target.detach().numpy()

        action_alpha = pi_alpha.act(torch.cat((rho, w_tensor)))
        action_rho = pi_rho.act(torch.cat((alpha, w_tensor)))

        action_alpha = pi_alpha.act(torch.cat((alpha, w_tensor)))
        action_rho = pi_rho.act(torch.cat((rho, w_tensor)))

        alpha = alpha + action_alpha
        rho = rho + action_rho

        alpha_hist[t] = alpha.detach().numpy()
        rho_hist[t] = rho.detach().numpy()

        alpha = alpha.detach()
        rho = rho.detach()
    
    pi_alpha.onpolicy_reset()
    pi_rho.onpolicy_reset()
    plot_trajectories(alpha_hist, alpha_target_hist, rho_hist, rho_target_hist, data_no, "test")
    alpha_rl = reshape_to_square(alpha.detach().numpy(), N_NODE)
    rho_rl = reshape_to_square(rho.detach().numpy(), N_NODE)
    D_rl = get_pairing_matrix_argmax(alpha_rl, rho_rl, N_NODE)
    print(f"alpha (RL):\n{alpha_rl}")
    print(f"rho (RL):\n{rho_rl}")
    print(f"alpha+rho (RL):\n{alpha_rl+rho_rl}")
    print(f"D (RL):\n{D_rl}")
    print(f"Pairing validity (RL): {check_validity(D_rl)}")


if __name__ == '__main__':
    main()
