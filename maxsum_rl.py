# %%
from maxsum import get_pairing_matrix_argmax, reshape_to_square, check_validity
from maxsum_ul import get_datasets, forward_pass, FILENAMES, N_DATASET, N_NODE, SEED_W
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import itertools
import os

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
GAMMA = 1.0
N_TRAIN = 50
N_EPISODE = 100
N_REPEAT = 500
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
    learning_rate = 1e-4
    pi_alpha, pi_rho= Pi(dim_in, dim_out), Pi(dim_in, dim_out)
    optim_alpha = optim.SGD(pi_alpha.parameters(), lr=learning_rate)
    optim_rho = optim.SGD(pi_rho.parameters(), lr=learning_rate)
    scheduler_alpha = optim.lr_scheduler.ReduceLROnPlateau(optim_alpha, 'min')
    scheduler_rho = optim.lr_scheduler.ReduceLROnPlateau(optim_alpha, 'min')
    loss_fn = nn.L1Loss(reduction='sum')

    if os.path.exists(FILENAMES["model_alpha"] and FILENAMES["model_rho"]):
        print("Loading train checkpoint...")
        ckpt_alpha = torch.load(FILENAMES["model_alpha"])
        ckpt_rho = torch.load(FILENAMES["model_rho"])
        pi_alpha.model.load_state_dict(ckpt_alpha['model_state_dict'])
        pi_rho.model.load_state_dict(ckpt_rho['model_state_dict'])
        optim_alpha.load_state_dict(ckpt_alpha['optimizer_state_dict'])
        optim_rho.load_state_dict(ckpt_rho['optimizer_state_dict'])
        data_no_ckpt = ckpt_rho['data_no']
        repeat_no_ckpt = ckpt_rho['repeat_no']
        print(f"Checkpoint up to data {data_no_ckpt} loaded.")
        dataset_start_idx = data_no_ckpt + 1
    else:
        dataset_start_idx = 0
        repeat_no_ckpt = 0

    for repeat_no in range(repeat_no_ckpt, N_REPEAT):
        for data_no in range(dataset_start_idx, N_TRAIN):
            D_mp = get_pairing_matrix_argmax(
                reshape_to_square(alpha_star[data_no], N_NODE),
                reshape_to_square(rho_star[data_no], N_NODE), N_NODE)
            # print(f"Train set {data_no} weights:\n{reshape_to_square(w[data_no], N_NODE)}")
            # plot_positions(pos_bs[data_no], pos_user[data_no], data_no)
            # print(f"D(mp):\n{D_mp}")

            init_alpha = torch.zeros((N_NODE**2))
            init_rho = torch.zeros((N_NODE**2))
            w_tensor = torch.from_numpy(w[data_no].astype(np.float32))
            reward_history = np.zeros(N_EPISODE)
            solved_history = np.zeros(N_EPISODE, dtype=bool)
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
                    alpha_target, rho_target = get_targets(alpha, rho, w[data_no])
                    alpha_act = pi_alpha.act(torch.cat((rho, w_tensor)))
                    rho_act = pi_rho.act(torch.cat((alpha, w_tensor)))
                    alpha = alpha + alpha_act
                    rho = rho + rho_act
                    td_loss_alpha = loss_fn(alpha, alpha_target)
                    td_loss_rho = loss_fn(rho, rho_target)
                    pi_alpha.rewards.append(-td_loss_alpha)
                    pi_rho.rewards.append(-td_loss_rho)

                    alpha_target_hist[t] = alpha_target.detach().numpy()
                    rho_target_hist[t] = rho_target.detach().numpy()
                    alpha_hist[t] = alpha.detach().numpy()
                    rho_hist[t] = rho.detach().numpy()

                    optim_alpha.zero_grad()
                    td_loss_alpha.backward(retain_graph=True)
                    optim_alpha.step()

                    optim_rho.zero_grad()
                    td_loss_rho.backward(retain_graph=False)
                    optim_rho.step()
                    
                    alpha = alpha.detach()
                    rho = rho.detach()

                # epi_loss_alpha = train(pi_alpha, optim_alpha, True)
                # epi_loss_rho = train(pi_rho, optim_rho, False)
                # scheduler_alpha.step(epi_loss_alpha)
                # scheduler_rho.step(epi_loss_rho)
                # epi_loss = epi_loss_alpha + epi_loss_rho

                D_rl = get_pairing_matrix_argmax(
                    reshape_to_square(alpha.detach().numpy(), N_NODE),
                    reshape_to_square(rho.detach().numpy(), N_NODE), N_NODE)
                solved_history[epi_no] = (D_rl==D_mp).all()
                total_reward_alpha = sum(pi_alpha.rewards)
                total_reward_rho = sum(pi_rho.rewards)
                reward_history[epi_no] = total_reward_alpha + total_reward_rho

                if epi_no % 100 == 0:
                    print(f"(Repeat{repeat_no} dataset{data_no} episode{epi_no}) "
                        f"total_reward: {total_reward_alpha+total_reward_rho:.2f}, "
                        # f"loss: {epi_loss:.2f}, "
                        f"RL-MP Ham.Dist.: {get_hamming_distance(D_rl, D_mp)}")
                    # plot_trajectories(alpha_hist, alpha_target_hist,
                    #                   rho_hist, rho_target_hist, data_no, epi_no)
                    np.save(f"training/ds{data_no}_epi{epi_no}_alpha_hist.npy", alpha_hist)
                    np.save(f"training/ds{data_no}_epi{epi_no}_alpha_target_hist.npy", alpha_target_hist)
                    np.save(f"training/ds{data_no}_epi{epi_no}_rho_hist.npy", rho_hist)
                    np.save(f"training/ds{data_no}_epi{epi_no}_rho_target_hist.npy", rho_target_hist)
                pi_alpha.onpolicy_reset()
                pi_rho.onpolicy_reset()
            # plot_reward(reward_history, solved_history, data_no)
            solved_percent = np.count_nonzero(solved_history) / len(solved_history) * 100
            print(f"{solved_percent}% of episodes succeeded.")
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
        self.rewards=[]

    def act(self, state):
        action = self.model(state)
        return action


def cheat_start(init_alpha, init_rho, w, n_cheat_step):
    for n in range(n_cheat_step):
        init_alpha_rho = (torch.cat((init_alpha, init_rho))).detach().numpy().reshape(1,-1)
        init_alpha_rho = forward_pass(w, init_alpha_rho)
        init_alpha, init_rho = np.array_split(init_alpha_rho, 2)
        init_alpha = torch.from_numpy(init_alpha.astype(np.float32))
        init_rho = torch.from_numpy(init_rho.astype(np.float32))
    return init_alpha, init_rho


def get_targets(alpha, rho, w):
    alpha_rho = torch.cat((alpha, rho)).detach().numpy().reshape(1, -1)
    alpha_rho_target = forward_pass(w, alpha_rho)
    alpha_target, rho_target = np.array_split(alpha_rho_target, 2)
    alpha_target = torch.from_numpy(alpha_target.astype(np.float32))
    rho_target = torch.from_numpy(rho_target.astype(np.float32))
    return alpha_target, rho_target


def train(pi, optimizer, bRetainGraph):
    T = len(pi.rewards)
    returns = torch.zeros(T)
    future_returns = 0.0
    for t in reversed(range(T)):
        future_returns = pi.rewards[t] + GAMMA * future_returns
        returns[t] = future_returns
    loss = -returns
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward(retain_graph=bRetainGraph)
    optimizer.step()
    return loss


def get_hamming_distance(D_rl, D_mp):
    hamming_dist = int(np.sum(abs(D_rl - D_mp)))
    return hamming_dist


def plot_positions(bs, user, idx):
    plt.title("BS and user positions")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    for i in range(N_NODE):
        plt.plot(bs[i, 0], bs[i, 1], 'b',
                marker=f"$b{i}$", markersize=16)
        plt.plot(user[i, 0], user[i, 1], 'r',
                marker=f"$u{i}$", markersize=16)
    plt.savefig(f"training/dataset{idx}_pos.png")
    plt.clf()
    plt.close("all")


def plot_reward(reward_history, solved_history, idx):
    reward_history *= -1
    plt.title('Neg. Reward')
    epi = np.linspace(0, len(reward_history)-1, len(reward_history))

    reward_history_solved = np.copy(reward_history)
    reward_history_solved[[not solved for solved in solved_history]] = np.nan

    plt.semilogy(epi, reward_history,
                 "-", color='black', alpha=0.5)
    plt.semilogy(epi, reward_history_solved,
                 "*", color='green', markersize=3)
    plt.xlim(xmin=0)
    plt.savefig(f"training/dataset{idx}_reward.png")
    plt.close('all')


def plot_trajectories(alpha_hist, alpha_target_hist, rho_hist, rho_target_hist, idx, episode):
    _, axes = plt.subplots(nrows=N_NODE, ncols=2, figsize=(10,20))
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
    plt.savefig(f"training/dataset{idx}_epi{episode}.png")
    plt.close('all')

if __name__ == '__main__':
    main()
