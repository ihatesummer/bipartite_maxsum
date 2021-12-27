from maxsum import check_validity, reshape_to_flat, reshape_to_square
from maxsum_condensed import get_pairing_matrix
from maxsum_ml_unsupervised import(fetch_dataset, forward_pass, decompose_dataset)
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
GAMMA = 0.99
N_EPISODE = 5000
MAX_TIMESTEP = 20
N_NODE = 5  # number of nodes per group
DIM_IN = 2*(N_NODE**2)
DIM_OUT = 2*(N_NODE**2)
FILENAMES = {
    "w": f"{N_NODE}-by-{N_NODE} - w.csv",
    "alpha_star": f"{N_NODE}-by-{N_NODE} - alpha_star.csv",
    "rho_star": f"{N_NODE}-by-{N_NODE} - rho_star.csv"
}
FILENAME_NN_WEIGHT = "weights_rl.h5"
SEED_W = 0


def main():
    (w, alpha_star, rho_star) = fetch_dataset()
    alpha_rho_star = np.concatenate((alpha_star, rho_star),
                                    axis=1)
    D_mp = get_pairing(alpha_rho_star[0])
    print(f"D (mp):\n{D_mp}")

    pi = Pi(DIM_IN, DIM_OUT)
    try:
        pi.model.load_state_dict(torch.load(FILENAME_NN_WEIGHT))
        print("Trained model found.")
        D_rl = evaluate_policy(pi)
        compare_pairings(D_rl, D_mp)
    except:
        print("No trained model found. Training...")
        optimizer = optim.Adam(pi.parameters(), lr=0.001)
        mae_fxn = nn.L1Loss(reduction='mean')
        loss_history = []
        reward_history = []
        solved_history = []
        print(f"Weights:\n{reshape_to_square(w[0], N_NODE)}")
        for episode in range(N_EPISODE):
            state = torch.tensor(
                np.array(
                    [np.concatenate((w[0]/2, -w[0]/2))]),
                requires_grad=True).float()
            # state = torch.zeros((1, DIM_IN),
            #                     requires_grad=True)
            for t in range(MAX_TIMESTEP):
                state_np = state.detach().numpy()
                action = pi.act(state)
                state_new = state + action
                state_new_np = state_new.detach().numpy()
                state_passed_np = forward_pass(w[0], state_np)
                state_passed = torch.tensor(np.array([state_passed_np]))
                reward = -mae_fxn(state_new, state_passed)
                pi.rewards.append(reward)
                # print(f"====timestep: {t}====")
                # print(f"State: \n{state_np}")
                # print(f"Action: \n{action.detach().numpy()}")
                # print(f"Next_state: \n{state_new_np}")
                # print(f"State_passed: \n{state_passed_np}")
                # print(f"Difference: \n{state_np - state_passed_np}")
                # print(f"Reward: {reward}")
                state = state_new
            loss = train(pi, optimizer)  # train per episode
            total_reward = sum(pi.rewards)
            reward_history.append(total_reward)
            loss_history.append(loss)
            if episode % 100 == 0:
                print(f"(Episode {episode}) "
                    f"total_reward: {total_reward:.2f}, "
                    f"sum_log_prob: {np.sum(pi.log_probs):.2f}, "
                    f"loss: {loss:.2f}")
            pi.onpolicy_reset()  # onpolicy: clear memory after training
        torch.save(pi.model.state_dict(), FILENAME_NN_WEIGHT)
        plot_reward_loss_vs_epoch(reward_history, loss_history)


class Pi_alpha(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(dim_in, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 80),
            nn.LeakyReLU(),
            nn.Linear(80, dim_out),
            nn.Tanh()
        ]

class Pi_rho(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Pi, self).__init__()


class Pi(nn.Module):
    def __init__(self, DIM_IN, DIM_OUT):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(DIM_IN, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, DIM_OUT),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # set training mode

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        means = self.forward(state)
        pd = Normal(loc=means, scale=0.5) # probability distribution
        action = pd.sample() # pi(a|s) in action via pd
        log_prob = torch.sum(pd.log_prob(action)) # log_prob of pi(a|s)
        self.log_probs.append(log_prob) # store for training
        return action


def train(pi, optimizer):
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    returns = np.empty(T, dtype=np.float32)
    future_returns = 0.0
    # compute the returns efficiently
    for t in reversed(range(T)):
        future_returns = pi.rewards[t] + GAMMA * future_returns
        returns[t] = future_returns
    returns = torch.tensor(returns)
    return_baseline = torch.mean(returns)
    log_probs_sum = torch.stack(pi.log_probs)
    # loss = - log_probs_sum * (returns-return_baseline)
    loss = log_probs_sum * (returns)
    loss_sum = torch.sum(loss)
    optimizer.zero_grad()
    loss_sum.backward()  # backpropagate, compute gradients
    optimizer.step()  # gradient-ascent, update the weights
    # print(f"returns: {returns}")
    # print(f"log_probs: {pi.log_probs}")
    # print(f"log_probs (sum): {log_probs_sum}")
    # print(f"loss: {loss}")
    # print(f"loss (sum): {loss_sum}")
    return loss_sum


def get_pairing(alpha_rho):
    alpha, rho = decompose_dataset(
        alpha_rho, 'output')
    alpha = reshape_to_square(alpha, N_NODE)
    rho = reshape_to_square(rho, N_NODE)
    # print("MP:")
    # print(f"alpha:\n{alpha}")
    # print(f"rho:\n{rho}")
    # print(f"sum:\n{alpha+rho}")
    return get_pairing_matrix(alpha, rho)


def evaluate_policy(pi):
    state = torch.zeros((1, DIM_IN),
                        requires_grad=True)
    for t in range(MAX_TIMESTEP):
        action = pi.act(state)
        state = state + action
    pi.onpolicy_reset()
    state = state.detach().numpy()[0]
    alpha_rl, rho_rl = decompose_dataset(
        state, 'output')
    alpha_rl = reshape_to_square(alpha_rl, N_NODE)
    rho_rl = reshape_to_square(rho_rl, N_NODE)
    D_rl = get_pairing_matrix(alpha_rl, rho_rl)
    print(f"alpha_rl:\n{alpha_rl}")
    print(f"rho_rl:\n{rho_rl}")
    print(f"alpha+rho (RL):\n{alpha_rl+rho_rl}")
    print(f"D (RL):\n{D_rl}")
    print(f"Pairing validity (RL): {check_validity(D_rl)}")
    return D_rl


def compare_pairings(D_rl, D_mp):
    diff_count = int(np.sum(abs(D_rl - D_mp))/2)
    print(f"{diff_count} pairings are different than the optimum.")


def plot_reward_loss_vs_epoch(reward_history, loss_history):
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    axes[0].set_title('reward')
    axes[1].set_title('loss')
    epi = np.linspace(0, len(reward_history)-1, len(reward_history))
    axes[0].plot(epi, reward_history,
                "-", color='red', alpha=0.2)
    axes[0].set_xlim(xmin=0)
    axes[1].plot(epi, loss_history,
                "-", color='green', alpha=0.2)
    axes[1].set_xlim(xmin=0)
    plt.show()


if __name__ == '__main__':
    main()
