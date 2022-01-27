# %%
from maxsum_mp import get_pairing_matrix_argmax, reshape_to_square, reshape_to_flat, update_alpha, update_rho, iterate_maxsum_mp
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import itertools
import os

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
GAMMA = 1.0
SEED_W = 0
N_NODE = 5
N_DATASET = 100
N_TRAIN = 2
N_EPISODE = 1
N_REPEAT = 1001
MAX_TIMESTEP = 10
FILENAMES = {
    "w": f"{N_NODE}x{N_NODE}_w.npy",
    "pos_bs": f"{N_NODE}x{N_NODE}_bs_pos.npy",
    "pos_user": f"{N_NODE}x{N_NODE}_user_pos.npy",
    "model_alpha": "model_alpha.h5",
    "model_rho": "model_rho.h5"
}


def main():
    check_cuda_device()
    _, _, w_ds = get_datasets()
    D_mp = get_reference_mp(w_ds, n_iter=MAX_TIMESTEP)
    D_hungarian= get_reference_hungarian(w_ds)

    learning_rate = 1e-4
    pi_alpha, pi_rho = (Pi(2*N_NODE**2, N_NODE**2),
                        Pi(2*N_NODE**2, N_NODE**2))
    optim_alpha, optim_rho = (optim.Adam(pi_alpha.parameters(),
                                         lr=learning_rate),
                              optim.Adam(pi_rho.parameters(), 
                                         lr=learning_rate))
    loss_fn = nn.MSELoss(reduction='sum')

    if os.path.exists(FILENAMES["model_alpha"] and FILENAMES["model_rho"]):
        ckpt_alpha = torch.load(FILENAMES["model_alpha"])
        ckpt_rho = torch.load(FILENAMES["model_rho"])
        pi_alpha.model.load_state_dict(ckpt_alpha['model_state_dict'])
        pi_rho.model.load_state_dict(ckpt_rho['model_state_dict'])
        optim_alpha.load_state_dict(ckpt_alpha['optimizer_state_dict'])
        optim_rho.load_state_dict(ckpt_rho['optimizer_state_dict'])
        repeat_no_ckpt = ckpt_rho['repeat_no'] + 1
        print(f"Checkpoint up to repeat#{repeat_no_ckpt-1} loaded.")
    else:
        repeat_no_ckpt = 0
        print(f"No checkpoint found.")

    for repeat_no in range(repeat_no_ckpt, N_REPEAT):
        dataset_indices = np.linspace(0, N_TRAIN-1, N_TRAIN, dtype=int)
        reward_history = np.zeros((N_REPEAT, N_TRAIN))
        solved_history = np.zeros((N_REPEAT, N_TRAIN), dtype=bool)
        ham_dist_history = np.zeros((N_REPEAT, N_TRAIN))
        for data_no in np.random.permutation(dataset_indices):
            # print(f"Train set {data_no} weights:\n{reshape_to_square(w[data_no], N_NODE)}")
            # print(f"D(mp):\n{D_mp[data_no]}")
            init_alpha, init_rho = (torch.zeros((N_NODE**2)),
                                    torch.zeros((N_NODE**2)))
            w_tensor = torch.from_numpy(w_ds[data_no].astype(np.float32))
            for epi_no in range(N_EPISODE):
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
                    if (alpha_clp != alpha).all():
                        td_loss_alpha += 1
                    if (rho_clp != rho).all():
                        td_loss_rho += 1
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
                total_reward_alpha = sum(pi_alpha.rewards)
                total_reward_rho = sum(pi_rho.rewards)
                pi_alpha.onpolicy_reset()
                pi_rho.onpolicy_reset()
            D_rl = get_pairing_matrix_argmax(
                reshape_to_square(alpha.detach().numpy(), N_NODE),
                reshape_to_square(rho.detach().numpy(), N_NODE),
                N_NODE)
            ham_dist = get_hamming_distance(D_rl, D_mp[data_no])
            ham_dist_history[repeat_no, data_no] = ham_dist
            solved_history[repeat_no, data_no] = (D_rl == D_mp[data_no]).all()
            reward_history[repeat_no, data_no] = total_reward_alpha + total_reward_rho

            if repeat_no % 100 == 0:
                print(f"(Repeat{repeat_no} data{data_no}) "
                    f"reward_sum(alpha): {total_reward_alpha:.2f}, "
                    f"reward_sum(rho): {total_reward_rho:.2f}, "
                    f"RL-MP Ham.Dist.: {ham_dist}")
                show_rl_traj(alpha_hist, alpha_target_hist,
                                  rho_hist, rho_target_hist, data_no, epi_no, repeat_no)
                # alpha_repeat = reshape_to_square(alpha.detach().numpy(), N_NODE)
                # rho_repeat = reshape_to_square(rho.detach().numpy(), N_NODE)
                # print(f"alpha+rho (RL):\n{alpha_epi+rho_epi}")
                # np.save(f"training/ds{data_no}_repeat{repeat_no}_alpha_hist.npy", alpha_hist)
                # np.save(f"training/ds{data_no}_repeat{repeat_no}_alpha_target_hist.npy", alpha_target_hist)
                # np.save(f"training/ds{data_no}_repeat{repeat_no}_rho_hist.npy", rho_hist)
                # np.save(f"training/ds{data_no}_repeat{repeat_no}_rho_target_hist.npy", rho_target_hist)

            # plot_reward(reward_history, solved_history, data_no, repeat_no)
            torch.save({'data_no': data_no,
                        'repeat_no': repeat_no,
                        'model_state_dict': pi_alpha.model.state_dict(),
                        'optimizer_state_dict': optim_alpha.state_dict()},
                       FILENAMES["model_alpha"])
            torch.save({'data_no': data_no,
                        'repeat_no': repeat_no,
                        'model_state_dict': pi_rho.model.state_dict(),
                        'optimizer_state_dict': optim_rho.state_dict()},
                        FILENAMES["model_rho"])
            # epi_no for-loop indent
        # data_no for-loop indent
    # repeat_no for-loop indent


class Pi(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Pi, self).__init__()
        layers=[
            nn.Linear(dim_in, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 150),
            nn.LeakyReLU(),
            nn.Linear(150, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 75),
            nn.LeakyReLU(),
            nn.Linear(75, 40),
            nn.LeakyReLU(),
            nn.Linear(40, dim_out),
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


def check_cuda_device():
    print("CUDA available:", torch.cuda.is_available())
    print("Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))


def get_datasets():
    w = np.load(FILENAMES["w"])
    pos_bs = np.load(FILENAMES["pos_bs"])
    pos_user = np.load(FILENAMES["pos_user"])
    return pos_bs, pos_user, w


def get_reference_mp(w_ds, n_iter):
    D_mp = np.zeros((N_TRAIN, N_NODE, N_NODE), dtype=int)
    for i in range(N_TRAIN):
        alpha_hist, rho_hist = iterate_maxsum_mp(w_ds[i], n_iter, N_NODE)
        alpha, rho = (reshape_to_square(alpha_hist[-1], N_NODE),
                      reshape_to_square(rho_hist[-1], N_NODE))
        D_mp[i] = get_pairing_matrix_argmax(alpha, rho, N_NODE)
    
    return D_mp


def get_reference_hungarian(w_ds):
    D_hungarian = np.zeros((N_TRAIN, N_NODE, N_NODE), dtype=int)
    for i in range(N_TRAIN):
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
    hamming_dist=int(np.sum(abs(D_rl - D_mp)))
    return hamming_dist


def plot_reward(reward_history, solved_history, ds_idx, repeat_idx):
    reward_history *= -1
    plt.title('Neg. Reward')
    epi=np.linspace(0, len(reward_history)-1, len(reward_history))

    reward_history_solved=np.copy(reward_history)
    reward_history_solved[[not solved for solved in solved_history]]=np.nan

    plt.semilogy(epi, reward_history,
                 "-", color='black', alpha=0.5)
    plt.semilogy(epi, reward_history_solved,
                 "*", color='green', markersize=3)
    plt.xlim(xmin=0)
    plt.savefig(f"training/data{ds_idx}_rep{repeat_idx}_reward.png")
    plt.close('all')


def show_rl_traj(alpha_hist, alpha_target_hist,
                 rho_hist, rho_target_hist,
                 idx, episode, repeat):
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
            axes[i, 0].plot(t, alpha_hist[:, N_NODE*i+j],
                            marker=mk)
            axes[i, 0].plot(t, alpha_target_hist[:, N_NODE*i+j],
                            marker=mk, color='black', linewidth=0)
            axes[i, 1].plot(t, rho_hist[:, N_NODE*i+j],
                            marker=mk)
            axes[i, 1].plot(t, rho_target_hist[:, N_NODE*i+j],
                            marker=mk, color='black', linewidth=0)
        axes[i, 0].set_xlim(xmin=0, xmax=n_steps-1)
        axes[i, 1].set_xlim(xmin=0, xmax=n_steps-1)
        # axes[i, 0].set_ylim(ymin=-3, ymax=3)
        # axes[i, 1].set_ylim(ymin=-3, ymax=3)
    plt.savefig(f"training/data{idx}_epi{episode}_rep{repeat}.png")
    plt.close('all')


if __name__ == '__main__':
    main()
