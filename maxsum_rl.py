from maxsum import check_validity, get_pairing_matrix_argmax, get_pairing_matrix_sign, reshape_to_flat, reshape_to_square
from maxsum_ul import get_datasets, forward_pass, decompose_dataset, FILENAMES, N_DATASET, N_NODE, SEED_W
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
GAMMA = 1.0
N_EPISODE = 10000
MAX_TIMESTEP = 10
FILENAMES["nn_weight"] = "weights_rl.h5"


def main():
    pos_bs, pos_user, w, alpha_star, rho_star = get_datasets(
        FILENAMES, N_DATASET, SEED_W, N_NODE, "unif_easy")
    dataset_idx = 0
    D_mp = get_pairing_matrix_argmax(
        reshape_to_square(alpha_star[dataset_idx], N_NODE),
        reshape_to_square(rho_star[dataset_idx], N_NODE), N_NODE)
    print(f"D (mp):\n{D_mp}")
    dim_in = 2*(N_NODE**2)
    dim_out = (N_NODE**2)
    pi = Pi(dim_in, dim_out)
    try:
        pi.model.load_state_dict(torch.load(FILENAMES["nn_weight"]))
        print("Trained model found.")
        D_rl = evaluate_policy(pi, w[dataset_idx])
        compare_pairings(D_rl, D_mp)
    except:
        print("No trained model found. Training...")
        optimizer = optim.Adam(pi.parameters(), lr=0.0001)
        # mae_fxn = nn.L1Loss(reduction='mean')
        loss_history = []
        reward_history = []
        solved_history = []
        print(f"Weights:\n{reshape_to_square(w[dataset_idx], N_NODE)}")
        for episode in range(N_EPISODE):
            # Cheat start
            init_alpha = reshape_to_square(w[dataset_idx]/2, N_NODE)
            init_rho = reshape_to_square(w[dataset_idx]/2, N_NODE)
            init_state = get_pairing_matrix_argmax(
                init_alpha, init_rho, N_NODE) - 0.5
            init_state = reshape_to_flat(init_state, N_NODE)
            state_MP = (np.concatenate((init_state, init_state))).reshape((1,-1))
            state=torch.tensor(state_MP, requires_grad=True).float()

            # # Zero start
            # state = torch.zeros((1, dim_in),
            #                     requires_grad=True)
            # state_MP = np.zeros((1, dim_in))
            # print(state_MP)
            # print(state)

            for t in range(MAX_TIMESTEP):
                action_half=pi.act(state)
                action = torch.cat((action_half[0], action_half[0]))
                state_new = state + action

                state_np = state.detach().numpy()
                state_new_np = state_new.detach().numpy()
                # state_passed_np = forward_pass(w[dataset_idx], state_np)
                state_passed_np = forward_pass(w[dataset_idx], state_MP)

                alpha_mp_now, rho_mp_now = decompose_dataset(
                    state_passed_np, 'output')
                alpha_rl_now, rho_rl_now = decompose_dataset(
                    state_new_np[0], 'output')
                reward = get_reward(alpha_mp_now, rho_mp_now,
                                    alpha_rl_now, rho_rl_now, N_NODE)
                D_rl_now = get_pairing_matrix_sign(alpha_rl_now, rho_rl_now)
                D_mp_now = get_pairing_matrix_argmax(alpha_mp_now, rho_mp_now, N_NODE)
                if ((D_rl_now == D_mp_now).all()):
                    pi.rewards.append(reward*2)
                    break
                pi.rewards.append(reward)

                # print(f"====timestep: {t}====")
                # print(f"State: \n{state_np}")
                # print(f"Action: \n{action.detach().numpy()}")
                # print(f"Next_state: \n{state_new_np}")
                # print(f"State_passed: \n{state_passed_np}")
                # print(f"alpha_rl_now: \n{alpha_rl_now}")
                # print(f"alpha_rl_now_max: \n{get_pairing_matrix_sign(alpha_rl_now, alpha_rl_now)}")
                # print(f"alpha_mp_now: \n{alpha_mp_now}")
                # print(f"alpha_mp_now_max: \n{get_pairing_matrix_sign(alpha_mp_now, alpha_mp_now)}")
                # print(f"rho_rl_now: \n{rho_rl_now}")
                # print(f"rho_rl_now_max: \n{get_pairing_matrix_sign(rho_rl_now, rho_rl_now)}")
                # print(f"rho_mp_now: \n{rho_mp_now}")
                # print(f"rho_mp_now_max: \n{get_pairing_matrix_sign(rho_mp_now, rho_mp_now)}")
                # print(f"D_rl_now: \n{get_pairing_matrix_sign(alpha_rl_now, rho_rl_now)}")
                # print(f"D_mp_now: \n{get_pairing_matrix_sign(alpha_mp_now, rho_mp_now)}")
                # print(f"Difference: \n{state_np - state_passed_np}")
                # print(f"Reward: {reward}")

                state_MP = np.array([state_passed_np])
                state = state_new

            loss_sum = train(pi, optimizer)  # train per episode
            total_reward = sum(pi.rewards)
            D_rl_epi = get_pairing_matrix_sign(alpha_rl_now, rho_rl_now)
            D_mp_epi = get_pairing_matrix_argmax(alpha_mp_now, rho_mp_now, N_NODE)
            solved = np.all(D_rl_epi == D_mp_epi)
            reward_history.append(total_reward)
            loss_history.append(loss_sum)
            solved_history.append(solved)
            if episode % 100 == 0:
                print(f"(Episode {episode}) "
                    f"total_reward: {total_reward:.2f}, "
                    f"sum_log_prob: {np.sum(pi.log_probs):.2f}, "
                    f"loss: {loss_sum:.2f}, "
                    f"solved: {solved}\n",
                    f"D_mp:\n{D_mp_epi}\n",
                    f"D_rl:\n{D_rl_epi}")
            pi.onpolicy_reset()  # onpolicy: clear memory after training
        torch.save(pi.model.state_dict(), FILENAMES["nn_weight"])
        plot_reward_loss_vs_epoch(
            reward_history, loss_history, solved_history)


class Pi(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Pi, self).__init__()
        layers=[
            nn.Linear(dim_in, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 80),
            nn.LeakyReLU(),
            nn.Linear(80, dim_out),
            nn.Tanh()
        ]
        self.model=nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # set training mode

    def onpolicy_reset(self):
        self.log_probs=[]
        self.rewards=[]

    def forward(self, x):
        pdparam=self.model(x)
        return pdparam

    def act(self, state):
        means=self.forward(state)
        pd=Normal(loc=means, scale=0.15)
        action=pd.sample()
        log_prob=pd.log_prob(action)
        log_prob_joint=torch.sum(log_prob)
        self.log_probs.append(log_prob_joint)  # store for training
        return action


def train(pi, optimizer):
    # Inner gradient-ascent loop of REINFORCE algorithm
    T=len(pi.rewards)
    returns=np.empty(T, dtype=np.float32)
    future_returns=0.0
    # compute the returns efficiently
    for t in reversed(range(T)):
        future_returns=pi.rewards[t] + GAMMA * future_returns
        returns[t]=future_returns
    returns=torch.tensor(returns)
    return_baseline=torch.mean(returns)
    log_probs_sum=torch.stack(pi.log_probs)
    loss=- log_probs_sum * returns
    # loss = - log_probs_sum * (returns-return_baseline)
    loss_sum=torch.sum(loss)
    optimizer.zero_grad()
    loss_sum.backward()  # backpropagate, compute gradients
    optimizer.step()  # gradient-ascent, update the weights
    # print(f"returns: {returns}")
    # print(f"log_probs: {pi.log_probs}")
    # print(f"log_probs (sum): {log_probs_sum}")
    # print(f"loss: {loss}")
    # print(f"loss (sum): {loss_sum}")
    return loss_sum


def get_reward(alpha_mp, rho_mp,
               alpha_rl, rho_rl,
               n_node):
    alpha_plus_rho_rl = alpha_rl + rho_rl
    alpha_plus_rho_mp = alpha_mp + rho_mp
    D_rl = get_pairing_matrix_sign(alpha_rl, rho_rl)
    D_mp = get_pairing_matrix_sign(alpha_mp, rho_mp)

    reward = np.sum(alpha_plus_rho_mp * D_rl)
    
    colsum = np.sum(D_rl, axis=0)
    rowsum = np.sum(D_rl, axis=1)
    if not ((rowsum == 1).all() and (colsum == 1).all()):
        n_mismatch = np.sum(np.abs(rowsum - np.ones(n_node)))
        n_mismatch += np.sum(np.abs(colsum - np.ones(n_node)))
        reward -= n_mismatch

    return reward


def evaluate_policy(pi, w):
    state=torch.tensor(np.array([np.concatenate((w/2, - w/2))]),
                         requires_grad=True).float()
    for t in range(MAX_TIMESTEP):
        action_half=pi.act(state)
        action=torch.cat((action_half[0], action_half[0]))
        state=state + action
    pi.onpolicy_reset()
    state = state.detach().numpy()[0]
    alpha_rl, rho_rl = decompose_dataset(
        state, 'output')
    alpha_rl = reshape_to_square(alpha_rl, N_NODE)
    rho_rl = reshape_to_square(rho_rl, N_NODE)
    D_rl = get_pairing_matrix_sign(alpha_rl, rho_rl)
    # print(f"alpha_rl:\n{alpha_rl}")
    # print(f"rho_rl:\n{rho_rl}")
    # print(f"alpha+rho (RL):\n{alpha_rl+rho_rl}")
    print(f"D (RL):\n{D_rl}")
    print(f"Pairing validity (RL): {check_validity(D_rl)}")
    return D_rl


def compare_pairings(D_rl, D_mp):
    diff_count=int(np.sum(abs(D_rl - D_mp))/2)
    print(f"{diff_count} pairings are different than the optimum.")


def plot_reward_loss_vs_epoch(reward_history,
                              loss_history,
                              solved_history):
    _, axes=plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axes[0].set_title('reward')
    axes[1].set_title('loss')
    epi=np.linspace(0, len(reward_history)-1, len(reward_history))

    reward_history_solved=np.asfarray(reward_history)
    reward_history_solved[[not solved for solved in solved_history]]=np.nan
    reward_history_unsolved=np.asfarray(reward_history)
    reward_history_unsolved[solved_history]=np.nan
    axes[0].plot(epi, reward_history_solved,
                 "*", color='green', markersize=1)
    axes[0].plot(epi, reward_history_unsolved,
                 "o", color='red', markersize=1)
    axes[0].plot(epi, reward_history,
                 "-", color='black', alpha=0.2)
    axes[0].set_xlim(xmin=0)

    loss_history_solved=np.asfarray(loss_history)
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
    plt.show()


if __name__ == '__main__':
    main()
