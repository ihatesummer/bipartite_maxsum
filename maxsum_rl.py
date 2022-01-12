# %%
from maxsum import check_validity, get_pairing_matrix_argmax, get_pairing_matrix_sign, reshape_to_flat, reshape_to_square
from maxsum_ul import get_datasets, forward_pass, decompose_dataset, FILENAMES, N_DATASET, N_NODE, SEED_W
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import itertools

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
GAMMA = 1.0
N_TRAIN = 1
N_TEST = 1
N_EPISODE = 100000
MAX_TIMESTEP = 5
FILENAMES["nn_weight_alpha"] = "weights_rl_alpha.h5"
FILENAMES["nn_weight_rho"] = "weights_rl_rho.h5"


def main():
    pos_bs, pos_user, w, alpha_star, rho_star = get_datasets(
        FILENAMES, N_DATASET, SEED_W, N_NODE, "geographic")
    dim_in = 2*N_NODE**2
    dim_out = N_NODE**2
    learning_rate = 1e-3
    pi_alpha = Pi(dim_in, dim_out)
    pi_rho = Pi(dim_in, dim_out)
    optim_alpha = optim.Adam(pi_alpha.parameters(), lr=learning_rate)
    optim_rho = optim.Adam(pi_rho.parameters(), lr=learning_rate)

    loss_history = []
    reward_history = []
    solved_history = []
    loss_fn = nn.MSELoss(reduction='sum')
    for data_no in range(N_TRAIN):
        print(f"Train set {data_no} weights:\n{reshape_to_square(w[data_no], N_NODE)}")
        plot_positions(pos_bs[data_no], pos_user[data_no], data_no)
        D_mp = get_pairing_matrix_argmax(reshape_to_square(alpha_star[data_no], N_NODE),
                                         reshape_to_square(rho_star[data_no], N_NODE),
                                         N_NODE)
        print(f"D(mp):\n{D_mp}")
        init_alpha = torch.zeros((N_NODE**2))
        init_rho = torch.zeros((N_NODE**2))
        w_tensor = torch.from_numpy(w[data_no].astype(np.float32))

        # init_alpha, init_rho = cheat_start(init_alpha, init_rho,
        #                                     w[data_no], n_cheat_step=5)

        for epi_no in range(N_EPISODE):
            alpha = init_alpha
            rho = init_rho
            alpha_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
            rho_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
            alpha_target_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
            rho_target_hist = np.zeros((MAX_TIMESTEP, N_NODE**2))
            alpha_hist[0] = alpha.detach().numpy()
            rho_hist[0] = rho.detach().numpy()
            alpha_target_hist[0] = alpha.detach().numpy()
            rho_target_hist[0] = rho.detach().numpy()
            
            for t in range(1, MAX_TIMESTEP):
                # print(f"====timestep: {t}====")
                # print(f"Prev. alpha: \n{reshape_to_square(alpha.detach().numpy(), N_NODE)}")
                # print(f"Prev rho: \n{reshape_to_square(rho.detach().numpy(), N_NODE)}")
                alpha_rho = torch.cat((alpha, rho)).detach().numpy().reshape(1, -1)
                alpha_rho_target = forward_pass(w[data_no], alpha_rho)
                alpha_target, rho_target = np.array_split(alpha_rho_target, 2)
                # print(f"Target alpha: \n{reshape_to_square(alpha_target, N_NODE)}")
                # print(f"Target rho: \n{reshape_to_square(rho_target, N_NODE)}")
                alpha_target = torch.from_numpy(alpha_target.astype(np.float32))
                rho_target = torch.from_numpy(rho_target.astype(np.float32))

                alpha_target = torch.ones(dim_out)
                rho_target = torch.ones(dim_out)

                alpha_target_hist[t] = alpha_target.detach().numpy()
                rho_target_hist[t] = rho_target.detach().numpy()

                action_alpha = pi_alpha.act(torch.cat((rho, w_tensor)))
                action_rho = pi_rho.act(torch.cat((alpha, w_tensor)))
                # print(f"Action (alpha): \n{reshape_to_square(action_alpha.detach().numpy(), N_NODE)}")
                # print(f"Action (rho): \n{reshape_to_square(action_rho.detach().numpy(), N_NODE)}")
                alpha = alpha + action_alpha
                rho = rho + action_rho

                # print(f"New alpha: \n{reshape_to_square(alpha.detach().numpy(), N_NODE)}")
                # print(f"New rho: \n{reshape_to_square(rho.detach().numpy(), N_NODE)}")
                # n_up_alpha = 0
                # n_down_alpha = 0
                # n_up_rho = 0
                # n_down_rho = 0
                # if (torch.max(alpha) > 1 or torch.min(alpha) < -1):
                #     n_up_alpha = torch.count_nonzero(alpha > 1)
                #     n_down_alpha = torch.count_nonzero(alpha < -1)
                #     reward_alpha -= 10 * \
                #         (n_up_alpha.item() + n_down_alpha.item())
                #     alpha = torch.clamp(alpha, -1, 1)
                # if (torch.max(rho) > 1 or torch.min(rho) < -1):
                #     n_up_rho = torch.count_nonzero(rho > 1)
                #     n_down_rho = torch.count_nonzero(rho < -1)
                #     reward_rho -= 10*(n_up_rho.item() + n_down_rho.item())
                #     rho = torch.clamp(rho, -1, 1)
                reward_alpha = -loss_fn(alpha, alpha_target)
                reward_rho = -loss_fn(rho, rho_target)
                # print(f"Reward (alpha): {reward_alpha}")
                # print(f"Reward (rho): {reward_rho}")
                pi_alpha.rewards.append(reward_alpha)
                pi_rho.rewards.append(reward_rho)

                alpha_hist[t] = alpha.detach().numpy()
                rho_hist[t] = rho.detach().numpy()

            loss_sum_alpha = train(pi_alpha, optim_alpha)
            loss_sum_rho = train(pi_rho, optim_rho)
            loss_history.append(loss_sum_alpha+loss_sum_rho)
            total_reward_alpha = sum(pi_alpha.rewards)
            total_reward_rho = sum(pi_rho.rewards)
            reward_history.append(total_reward_alpha + total_reward_rho)

            if epi_no % 1000 == 0:
                print(f"(Dataset{data_no} episode{epi_no}) "
                    f"total_reward: {total_reward_alpha+total_reward_rho:.2f}, "
                    f"sum_log_prob: {np.sum(pi_alpha.log_probs) + np.sum(pi_rho.log_probs):.2f}, "
                    f"loss: {loss_sum_alpha+loss_sum_rho:.2f}")
                compare_with_target(alpha_hist, alpha_target_hist, rho_hist, rho_target_hist,
                                    data_no, epi_no)
            pi_alpha.onpolicy_reset()
            pi_rho.onpolicy_reset()
    torch.save(pi_alpha.model.state_dict(), FILENAMES["nn_weight_alpha"])
    torch.save(pi_rho.model.state_dict(), FILENAMES["nn_weight_rho"])
    plot_reward_loss_vs_epoch(
        reward_history, loss_history, solved_history, data_no)


class Pi(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Pi, self).__init__()
        layers=[
            nn.Linear(dim_in, 70),
            nn.LeakyReLU(),
            nn.Linear(70, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 30),
            nn.LeakyReLU(),
            nn.Linear(30, dim_out),
            nn.Tanh()
        ]
        self.model=nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # set training mode

    def onpolicy_reset(self):
        self.log_probs=[]
        self.rewards=[]

    def act(self, state):
        means = self.model(state)
        pd = Normal(loc=means, scale=0.001)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        log_prob_joint = torch.sum(log_prob)
        self.log_probs.append(log_prob_joint)  # store for training
        return action


def cheat_start(init_alpha, init_rho, w, n_cheat_step):
    for n in range(n_cheat_step):
        init_alpha_rho = (torch.cat((init_alpha, init_rho))).detach().numpy().reshape(1,-1)
        init_alpha_rho = forward_pass(w, init_alpha_rho)
        init_alpha, init_rho = np.array_split(init_alpha_rho, 2)
        init_alpha = torch.from_numpy(init_alpha.astype(np.float32))
        init_rho = torch.from_numpy(init_rho.astype(np.float32))
    return init_alpha, init_rho


def train(pi, optimizer):
    T = len(pi.rewards)
    returns = np.empty(T, dtype=np.float32)
    future_returns = 0.0
    for t in reversed(range(T)):
        future_returns = pi.rewards[t] + GAMMA * future_returns
        returns[t] = future_returns
    returns = torch.tensor(returns)
    return_baseline = torch.mean(returns)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * returns
    # loss = - log_probs * (returns-return_baseline)
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()  # backpropagate, compute gradients
    optimizer.step()  # gradient-ascent, update the weights
    return loss


def compare_pairings(D_rl, D_mp):
    diff_count=int(np.sum(abs(D_rl - D_mp))/2)
    print(f"{diff_count} pairings are different than the optimum.")


def plot_positions(bs, user, idx):
    _, ax = plt.subplots()
    ax.set_title("BS and user positions")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    for i in range(N_NODE):
        ax.plot(bs[i, 0], bs[i, 1], 'b',
                marker=f"$b{i}$", markersize=16)
        ax.plot(user[i, 0], user[i, 1], 'r',
                marker=f"$u{i}$", markersize=16)
    plt.savefig(f"training/dataset{idx}_pos.png")
    plt.close('all')


def plot_reward_loss_vs_epoch(reward_history,
                              loss_history,
                              solved_history, idx):
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    axes[0].set_title('reward')
    axes[1].set_title('loss')
    epi = np.linspace(0, len(reward_history)-1, len(reward_history))

    reward_history_solved = np.asfarray(reward_history)
    reward_history_solved[[not solved for solved in solved_history]] = np.nan
    reward_history_unsolved = np.asfarray(reward_history)
    reward_history_unsolved[solved_history] = np.nan
    axes[0].plot(epi, reward_history_solved,
                 "*", color='green', markersize=1)
    axes[0].plot(epi, reward_history_unsolved,
                 "o", color='red', markersize=1)
    axes[0].plot(epi, reward_history,
                 "-", color='black', alpha=0.2)
    axes[0].set_xlim(xmin=0)

    loss_history_solved = np.asfarray(loss_history)
    loss_history_solved[[not solved for solved in solved_history]]=np.nan
    loss_history_unsolved=np.asfarray(loss_history)
    loss_history_unsolved[solved_history]=np.nan
    axes[1].plot(epi, loss_history_solved,
                 "o", color='green', markersize=2)
    axes[1].plot(epi, loss_history_unsolved,
                 "o", color='red', markersize=2)
    axes[1].plot(epi, loss_history,
                 "-", color='black', alpha=0.2)
    axes[1].set_xlim(xmin=0)
    plt.savefig(f"training/dataset{idx}.png")
    plt.close('all')


def compare_with_target(alpha_hist, alpha_target_hist, rho_hist, rho_target_hist, idx, episode):
    _, axes = plt.subplots(nrows=N_NODE, ncols=2, figsize=(12,24))
    axes[0, 0].set_title("alpha")
    axes[0, 1].set_title("rho")
    t = np.linspace(0, MAX_TIMESTEP-1, MAX_TIMESTEP)
    marker = itertools.cycle(("$1$", "$2$", "$3$", "$4$", "$5$"))
    for i in range(N_NODE):
        for j in range (N_NODE):
            mk = next(marker)
            axes[i, 0].plot(t, alpha_hist[:, N_NODE*i+j],
                        marker=mk,
                        color='green')
            axes[i, 0].plot(t, alpha_target_hist[:, N_NODE*i+j],
                        marker=mk,
                        color='red', linewidth=0)
            axes[i, 1].plot(t, rho_hist[:, N_NODE*i+j],
                            marker=mk,
                            color='green')
            axes[i, 1].plot(t, rho_target_hist[:, N_NODE*i+j],
                            marker=mk,
                            color='red', linewidth=0)
        axes[i, 0].set_xlim(xmin=0, xmax=MAX_TIMESTEP-1)
        axes[i, 1].set_xlim(xmin=0, xmax=MAX_TIMESTEP-1)
        # axes[i, 0].set_ylim(ymin=-3, ymax=3)
        # axes[i, 1].set_ylim(ymin=-3, ymax=3)
    # axes[0].legend()
    # axes[1].legend()
    plt.savefig(f"training/dataset{idx}_epi{episode}.png")
    plt.close('all')

if __name__ == '__main__':
    main()
