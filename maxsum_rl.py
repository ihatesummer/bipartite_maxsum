# %%
from maxsum_mp import get_pairing_matrix_argmax, reshape_to_square, reshape_to_flat, update_alpha, update_rho, iterate_maxsum_mp
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
import itertools
import os

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
matplotlib.use('Agg')
GAMMA = 1.0
SEED_W = 0
N_NODE = 5
N_DATASET = 100000
N_TRAIN = 90000
N_REPEAT = 50
MAX_TIMESTEP = 15
FILENAMES = {
    "w": f"{N_NODE}x{N_NODE}_w.npy",
    "pos_bs": f"{N_NODE}x{N_NODE}_bs_pos.npy",
    "pos_user": f"{N_NODE}x{N_NODE}_user_pos.npy",
    "model_alpha": "ckpt_alpha.h5",
    "model_rho": "ckpt_rho.h5",
    "general_info": "ckpt_general.h5"
}


def main():
    check_cuda_device()
    _, _, w_ds = get_datasets()
    D_mp = get_reference_mp(w_ds, n_iter=MAX_TIMESTEP)
    D_hungarian= get_reference_hungarian(w_ds)

    learning_rate = 1e-5
    pi_alpha, pi_rho = (Pi(2*N_NODE**2, N_NODE**2),
                        Pi(2*N_NODE**2, N_NODE**2))
    optim_alpha, optim_rho = (optim.Adam(pi_alpha.parameters(),
                                         lr=learning_rate,
                                         weight_decay=1e-6),
                              optim.Adam(pi_rho.parameters(), 
                                         lr=learning_rate,
                                         weight_decay=1e-6))
    loss_fn = nn.MSELoss(reduction='mean')

    if os.path.exists(FILENAMES["model_alpha"] and FILENAMES["model_rho"]):
        ckpt_alpha = torch.load(FILENAMES["model_alpha"])
        ckpt_rho = torch.load(FILENAMES["model_rho"])
        ckpt_general = torch.load(FILENAMES["general_info"])
        pi_alpha.model.load_state_dict(ckpt_alpha['model_state_dict'])
        pi_rho.model.load_state_dict(ckpt_rho['model_state_dict'])
        optim_alpha.load_state_dict(ckpt_alpha['optimizer_state_dict'])
        optim_rho.load_state_dict(ckpt_rho['optimizer_state_dict'])
        repeat_no_ckpt = ckpt_general['repeat_no'] + 1
        trained_idx = ckpt_general['trained_idx']
        failure_count_mp = ckpt_general['failure_count_mp']
        print(f"Checkpoint up to repeat#{repeat_no_ckpt-1} loaded.")
    else:
        print(f"No checkpoint found.")
        repeat_no_ckpt = 0
        failure_count_mp = 0
        trained_idx = np.array([])

    for repeat_no in range(repeat_no_ckpt, N_REPEAT):
        train_count = 0
        trained_idx = np.array([])
        dataset_indices = np.linspace(0, N_TRAIN-1, N_TRAIN, dtype=int)
        random_order = np.random.permutation(dataset_indices)
        for data_no in random_order:
            if data_no in trained_idx:
                continue
            else:
                trained_idx = np.append(trained_idx, data_no)
            # print(f"Train set {data_no} weights:\n{reshape_to_square(w_ds[data_no], N_NODE)}")
            if (D_mp[data_no] != D_hungarian[data_no]).any():
                if repeat_no == 0:
                    failure_count_mp += 1
            train_count += 1
            init_alpha, init_rho = (torch.zeros((N_NODE**2)),
                                    torch.zeros((N_NODE**2)))
            w_tensor = torch.from_numpy(w_ds[data_no].astype(np.float32))

            alpha, rho = init_alpha, init_rho
            (alpha_hist, rho_hist,
             alpha_target_hist,
             rho_target_hist) = (np.zeros((MAX_TIMESTEP, N_NODE**2)),
                                 np.zeros((MAX_TIMESTEP, N_NODE**2)),
                                 np.zeros((MAX_TIMESTEP, N_NODE**2)),
                                 np.zeros((MAX_TIMESTEP, N_NODE**2)))
            alpha_hist[0], rho_hist[0] = (alpha.detach().numpy(),
                                          rho.detach().numpy())
            alpha_target_hist[0], rho_target_hist[0] = (alpha.detach().numpy(),
                                                        rho.detach().numpy())
            for t in range(1, MAX_TIMESTEP):
                alpha_target, rho_target = get_targets(alpha, rho, w_ds[data_no])
                alpha_act, rho_act = (pi_alpha.act(torch.cat((rho, w_tensor))),
                                        pi_rho.act(torch.cat((alpha, w_tensor))))
                alpha, rho = alpha + alpha_act, rho + rho_act

                td_loss_alpha, td_loss_rho = (loss_fn(alpha, alpha_target),
                                              loss_fn(rho, rho_target))
                alpha_clp = torch.clamp(alpha, -1, 1)
                rho_clp = torch.clamp(rho, -1, 1)
                if (alpha_clp != alpha).any():
                    td_loss_alpha += 10
                if (rho_clp != rho).any():
                    td_loss_rho += 10
                alpha, rho = alpha_clp, rho_clp
                pi_alpha.rewards.append(-td_loss_alpha)
                pi_rho.rewards.append(-td_loss_rho)

                alpha_hist[t] = alpha.detach().numpy()
                rho_hist[t] = rho.detach().numpy()
                alpha_target_hist[t] = alpha_target.detach().numpy()
                rho_target_hist[t] = rho_target.detach().numpy()

                optim_alpha.zero_grad()
                td_loss_alpha.backward(retain_graph=True)
                optim_alpha.step()

                optim_rho.zero_grad()
                td_loss_rho.backward(retain_graph=False)
                optim_rho.step()

                alpha, rho = alpha.detach(), rho.detach()
            
            # epi_loss_alpha = on_policy_train(pi_alpha, optim_alpha, True)
            # epi_loss_rho = on_policy_train(pi_rho, optim_rho, False)

            total_reward_alpha = sum(pi_alpha.rewards)
            total_reward_rho = sum(pi_rho.rewards)
            pi_alpha.onpolicy_reset()
            pi_rho.onpolicy_reset()
            
            if train_count % 1000 == 0:
                D_rl_valid = validate_pi(pi_alpha, pi_rho, w_ds[data_no],
                                            data_no, repeat_no)
                ham_dist = get_hamming_distance(D_rl_valid, D_mp[data_no])
                print(f"(Repeat{repeat_no} data{data_no}) "
                    f"reward_sum(alpha): {total_reward_alpha:.4f}, "
                    f"reward_sum(rho): {total_reward_rho:.4f}, "
                    f"RL-MP Ham.Dist.: {ham_dist}")
                # print(f"Validation pairing:\n{D_rl_valid}")

                torch.save({'model_state_dict': pi_alpha.model.state_dict(),
                            'optimizer_state_dict': optim_alpha.state_dict()},
                        FILENAMES["model_alpha"])
                torch.save({'model_state_dict': pi_rho.model.state_dict(),
                            'optimizer_state_dict': optim_rho.state_dict()},
                            FILENAMES["model_rho"])
                torch.save({'repeat_no': repeat_no,
                            'trained_idx': trained_idx,
                            'failure_count_mp': failure_count_mp},
                            FILENAMES["general_info"])
        trained_idx = np.array([])
            # epi_no for-loop indent
        # data_no for-loop indent
    # repeat_no for-loop indent


class Pi(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Pi, self).__init__()
        layers=[
            nn.Linear(dim_in, 75),
            nn.LeakyReLU(),
            nn.Linear(75, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 30),
            nn.LeakyReLU(),
            nn.Linear(30, dim_out),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # set training mode

    def onpolicy_reset(self):
        self.rewards = []

    def act(self, state):
        action = self.model(state)
        return action


def validate_pi(pi_alpha, pi_rho, w, idx, repeat_no):
    pi_alpha.eval()
    pi_rho.eval()
    alpha = torch.zeros((N_NODE**2))
    rho = torch.zeros((N_NODE**2))
    w_tensor = torch.from_numpy(w.astype(np.float32))

    alpha_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    rho_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    alpha_hist[0] = alpha.detach().numpy()
    rho_hist[0] = rho.detach().numpy()
    alpha_target_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    rho_target_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
    alpha_target_hist[0] = alpha.detach().numpy()
    rho_target_hist[0] = rho.detach().numpy()

    for t in range(1, MAX_TIMESTEP):
        alpha_target, rho_target = get_targets(alpha, rho, w)
        alpha_act, rho_act = (
            pi_alpha.act(torch.cat((rho, w_tensor))),
            pi_rho.act(torch.cat((alpha, w_tensor))))
        alpha, rho = alpha + alpha_act, rho + rho_act
        alpha = torch.clamp(alpha, -1, 1)
        rho = torch.clamp(rho, -1, 1)
        alpha_hist[t] = alpha.detach().numpy()
        rho_hist[t] = rho.detach().numpy()
        alpha_target_hist[t] = alpha_target.detach().numpy()
        rho_target_hist[t] = rho_target.detach().numpy()
    
    pi_alpha.onpolicy_reset()
    pi_rho.onpolicy_reset()
    pi_alpha.train()
    pi_rho.train()

    show_rl_traj(alpha_hist, alpha_target_hist,
                 rho_hist, rho_target_hist,
                 idx, f"{repeat_no}_valid")
    alpha_rl = reshape_to_square(alpha.detach().numpy(), N_NODE)
    rho_rl = reshape_to_square(rho.detach().numpy(), N_NODE)
    D_rl = get_pairing_matrix_argmax(alpha_rl, rho_rl, N_NODE)
    # print(f"alpha+rho (RL):\n{alpha_rl+rho_rl}")
    return D_rl


def check_cuda_device():
    print("CUDA available:", torch.cuda.is_available())
    print("Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))


def get_datasets():
    w = np.load(FILENAMES["w"])
    pos_bs = np.load(FILENAMES["pos_bs"])
    pos_user = np.load(FILENAMES["pos_user"])
    return pos_bs, pos_user, w


def get_reference_mp(w_ds, n_iter):
    D_mp = np.zeros((N_DATASET, N_NODE, N_NODE), dtype=int)
    for i in range(N_DATASET):
        alpha_hist, rho_hist = iterate_maxsum_mp(w_ds[i], n_iter, N_NODE)
        alpha, rho = (reshape_to_square(alpha_hist[-1], N_NODE),
                      reshape_to_square(rho_hist[-1], N_NODE))
        D_mp[i] = get_pairing_matrix_argmax(alpha, rho, N_NODE)
    
    return D_mp


def get_reference_hungarian(w_ds):
    D_hungarian = np.zeros((N_DATASET, N_NODE, N_NODE), dtype=int)
    for i in range(N_DATASET):
        w_i = reshape_to_square(w_ds[i], N_NODE)
        row_idx, col_idx = linear_sum_assignment(w_i, maximize=True)
        for j in range(N_NODE):
            D_hungarian[i, row_idx[j], col_idx[j]] = 1
    return D_hungarian


def get_targets(alpha, rho, w):
    alpha_sqr = reshape_to_square(alpha.detach().numpy(), N_NODE)
    rho_sqr = reshape_to_square(rho.detach().numpy(), N_NODE)
    w_sqr = reshape_to_square(w, N_NODE)

    alpha_target_sqr = update_alpha(alpha_sqr, rho_sqr, w_sqr, bLogSumExp=False)
    rho_target_sqr = update_rho(alpha_sqr, rho_sqr, w_sqr, bLogSumExp=False)

    alpha_target = reshape_to_flat(alpha_target_sqr, N_NODE)
    rho_target = reshape_to_flat(rho_target_sqr, N_NODE)

    alpha_target = torch.from_numpy(alpha_target.astype(np.float32))
    rho_target = torch.from_numpy(rho_target.astype(np.float32))
    return alpha_target, rho_target


def on_policy_train(pi, optimizer, bRetainGraph):
    T=len(pi.rewards)
    returns=torch.zeros(T)
    future_returns=0.0
    for t in reversed(range(T)):
        future_returns=pi.rewards[t] + GAMMA * future_returns
        returns[t]=future_returns
    loss=-returns
    loss=torch.sum(loss)
    optimizer.zero_grad()
    loss.backward(retain_graph=bRetainGraph)
    optimizer.step()
    return loss


def get_hamming_distance(D_rl, D_mp):
    hamming_dist = int(np.sum(abs(D_rl - D_mp)))
    return hamming_dist


def show_rl_traj(alpha_hist, alpha_target_hist,
                 rho_hist, rho_target_hist,
                 idx, repeat):
    _, axes=plt.subplots(nrows=N_NODE, ncols=2,
                         figsize=(10, 12),
                         tight_layout=True)
    axes[0, 0].set_title(r"$\alpha$")
    axes[0, 1].set_title(r"$\rho$")
    n_steps = np.size(alpha_hist, axis=0)
    t = np.linspace(0, n_steps-1, n_steps)
    marker = itertools.cycle(("$1$", "$2$", "$3$", "$4$", "$5$"))
    for i in range(N_NODE):
        for j in range(N_NODE):
            mk = next(marker)
            # mk = None
            axes[i, 0].plot(t, alpha_hist[:, N_NODE*i+j])
            axes[i, 0].plot(t, alpha_target_hist[:, N_NODE*i+j],
                            marker=mk, color='black', linewidth=0)
            axes[i, 1].plot(t, rho_hist[:, N_NODE*i+j])
            axes[i, 1].plot(t, rho_target_hist[:, N_NODE*i+j],
                            marker=mk, color='black', linewidth=0)
        axes[i, 0].set_xlim(xmin=0, xmax=n_steps-1)
        axes[i, 1].set_xlim(xmin=0, xmax=n_steps-1)
        # axes[i, 0].set_ylim(ymin=-3, ymax=3)
        # axes[i, 1].set_ylim(ymin=-3, ymax=3)
    plt.savefig(f"training/data{idx}_rep{repeat}.png")
    plt.close('all')


if __name__ == '__main__':
    main()
