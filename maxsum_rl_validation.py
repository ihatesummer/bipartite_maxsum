# %%
from maxsum_rl import Pi, get_datasets, get_targets, show_rl_traj, get_hamming_distance, FILENAMES, N_DATASET, N_NODE, SEED_W, MAX_TIMESTEP
from maxsum_mp import check_pairing_validity, get_pairing_matrix_argmax, reshape_to_square
import numpy as np
import torch
import os

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def main(idx = 0):
    print("CUDA available:", torch.cuda.is_available())
    print("Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    pos_bs, pos_user, w, alpha_star, rho_star = get_datasets(
        FILENAMES, N_DATASET, SEED_W, N_NODE, "geographic")
    dim_in = 2*N_NODE**2
    dim_out = N_NODE**2
    pi_alpha = Pi(dim_in, dim_out)
    pi_rho = Pi(dim_in, dim_out)
    pi_alpha.eval()
    pi_rho.eval()

    if os.path.exists(FILENAMES["model_alpha"] and FILENAMES["model_rho"]):
        ckpt_alpha = torch.load(FILENAMES["model_alpha"])
        ckpt_rho = torch.load(FILENAMES["model_rho"])
        pi_alpha.model.load_state_dict(ckpt_alpha['model_state_dict'])
        pi_rho.model.load_state_dict(ckpt_rho['model_state_dict'])
        print("Train checkpoint loaded.")
    else:
        print("No trained model found.")
        return
    
    print(f"Data {idx} weights:\n{reshape_to_square(w[idx], N_NODE)}")
    D_mp = get_pairing_matrix_argmax(
        reshape_to_square(alpha_star[idx], N_NODE),
        reshape_to_square(rho_star[idx], N_NODE), N_NODE)
    print(f"D(mp):\n{D_mp}")

    alpha = torch.zeros((N_NODE**2))
    rho = torch.zeros((N_NODE**2))
    w_tensor = torch.from_numpy(w[idx].astype(np.float32))

    alpha_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    rho_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    alpha_target_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    rho_target_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    alpha_hist[0] = alpha.detach().numpy()
    rho_hist[0] = rho.detach().numpy()
    alpha_target_hist[0] = alpha.detach().numpy()
    rho_target_hist[0] = rho.detach().numpy()
    for t in range(1, MAX_TIMESTEP):
        alpha_target, rho_target = get_targets(alpha, rho, w[idx])
        alpha_act, rho_act = (pi_alpha.act(torch.cat((rho, w_tensor))),
                              pi_rho.act(torch.cat((alpha, w_tensor))))
        alpha, rho = alpha + alpha_act, rho + rho_act

        alpha_hist[t] = alpha.detach().numpy()
        rho_hist[t] = rho.detach().numpy()
        alpha_target_hist[t] = alpha_target.detach().numpy()
        rho_target_hist[t] = rho_target.detach().numpy()

        alpha, rho = alpha.detach(), rho.detach()
    
    pi_alpha.onpolicy_reset()
    pi_rho.onpolicy_reset()
    show_rl_traj(alpha_hist, alpha_target_hist,
                      rho_hist, rho_target_hist, idx, "_valid", 0)
    alpha_rl = reshape_to_square(alpha.detach().numpy(), N_NODE)
    rho_rl = reshape_to_square(rho.detach().numpy(), N_NODE)
    D_rl = get_pairing_matrix_argmax(alpha_rl, rho_rl, N_NODE)
    # print(f"alpha (RL):\n{alpha_rl}")
    # print(f"rho (RL):\n{rho_rl}")
    print(f"alpha+rho (RL):\n{alpha_rl+rho_rl}")
    print(f"D (RL):\n{D_rl}")
    print(f"Pairing validity: {check_pairing_validity(D_rl)} || "
          f"Ham. dist.: {get_hamming_distance(D_rl, D_mp)}")


if __name__ == '__main__':
    main()
