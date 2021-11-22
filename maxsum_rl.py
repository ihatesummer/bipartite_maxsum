from maxsum_ml_unsupervised import(fetch_dataset, forward_pass)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

np.set_printoptions(precision=2)
gamma = 0.99
n_episode = 10
max_timestep = 300
N_NODE = 5  # number of nodes per group
N_ITER = N_NODE*10
N_DATASET = 10000
filenames = {
    "w": f"{N_NODE}-by-{N_NODE} - w.csv",
    "alpha_star": f"{N_NODE}-by-{N_NODE} - alpha_star.csv",
    "rho_star": f"{N_NODE}-by-{N_NODE} - rho_star.csv"
}
FILENAME_NN_WEIGHT = "weights_rl.h5"
SEED_W = 0


def main():
    (w, alpha_star, rho_star) = fetch_dataset()
    alpha_rho_star = np.concatenate((alpha_star,
                                     rho_star),
                                    axis=1)
    in_dim = 2*(N_NODE**2)
    out_dim = 2*(N_NODE**2)
    pi = Pi(in_dim, out_dim)  # policy
    optimizer = optim.Adam(pi.model.parameters(), lr=0.1)
    loss_fxn = nn.MSELoss(reduction='mean')
    for episode in range(n_episode):
        # state = np.array([np.zeros(in_dim)])
        # state = torch.from_numpy(state.astype(np.float32))
        state = torch.zeros((1, in_dim),
                            requires_grad=True)
        for t in range(max_timestep):
            # print(f"====timestep: {t}====")
            action = pi.act(state)
            state_now = state.clone().detach() + action
            state_np = state_now.detach().numpy()
            # print(f"next_state: \n{state}")
            state_passed = forward_pass(w[0],
                                        state_np)
            # print(f"state_passed: \n{state_passed}")
            reward = -np.mean(np.square(state_np - state_passed))
            # print(f"reward: {reward}")
            pi.rewards.append(reward)
            state = state_now
        loss = train(pi, optimizer)  # train per episode
        total_reward = sum(pi.rewards)
        print(f"Episode {episode}: loss: {loss:.2f}, "
              f"total_reward: {total_reward:.2f}")
        pi.onpolicy_reset()  # onpolicy: clear memory after training


class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # set training mode

    def onpolicy_reset(self):
        self.actions = []
        self.rewards = []

    def act(self, state):
        action = self.model(state)
        self.actions.append(action)  # store for training
        return action


def train(pi, optimizer):
    optimizer.zero_grad()
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    returns = np.empty(T, dtype=np.float32)
    future_returns = 0.0
    # compute the returns efficiently
    for t in (range(T)):
        returns[t] = pi.rewards[t] * (gamma**t)
    loss = torch.tensor(np.abs(returns))  # gradient term; PyTorch optimizer minimizes this
    loss = torch.sum(loss)
    loss.backward()  # backpropagate, compute gradients
    optimizer.step()  # gradient-ascent, update the weights
    return loss


if __name__ == '__main__':
    main()
