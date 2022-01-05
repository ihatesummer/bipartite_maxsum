from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

gamma = 0.99

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train() # set training mode

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32)) # to tensor
        pdparam = self.forward(x) # forward pass
        pd = Categorical(logits=pdparam) # probability distribution
        action = pd.sample() # pi(a|s) in action via pd
        log_prob = pd.log_prob(action) # log_prob of pi(a|s)
        self.log_probs.append(log_prob) # store for training
        return action.item()

def train(pi, optimizer):
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32) # the returns
    future_ret = 0.0
    # compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    # print(f"returns: {rets}")
    # print(f"log_probs: {pi.log_probs}")
    log_probs = torch.stack(pi.log_probs)
    # print(f"log_probs (stack): {pi.log_probs}")
    loss = - log_probs * rets # gradient term; Negative for maximizing
    # print(f"loss: {loss}")
    loss = torch.sum(loss)
    # print(f"loss (sum): {loss}")
    optimizer.zero_grad()
    loss.backward() # backpropagate, compute gradients
    optimizer.step() # gradient-ascent, update the weights
    return loss

def main():
    env = gym.make('CartPole-v0')
    in_dim = env.observation_space.shape[0] # 4
    out_dim = env.action_space.n # 2
    pi = Pi(in_dim, out_dim) # policy pi_theta for REINFORCE
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    loss_history = []
    reward_history = []
    solved_history = []
    for epi in range(2000):
        state = env.reset()
        for t in range(200): # cartpole max timestep is 200
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            # env.render()
            if done:
                break
        loss = train(pi, optimizer) # train per episode
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset() # onpolicy: clear memory after training
        print(f'Episode {epi}, loss: {loss}, \
        total_reward: {total_reward}, solved: {solved}')
        reward_history.append(total_reward)
        loss_history.append(loss)
        solved_history.append(solved)


    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    axes[0].set_title('reward')
    axes[1].set_title('loss')
    epi = np.linspace(0, len(reward_history)-1, len(reward_history))

    reward_history_solved = np.asfarray(reward_history)
    reward_history_solved[[not solved for solved in solved_history]] = np.nan
    reward_history_unsolved = np.asfarray(reward_history)
    reward_history_unsolved[solved_history] = np.nan
    axes[0].plot(epi, reward_history_solved,
                 "o", color='green', markersize=2)
    axes[0].plot(epi, reward_history_unsolved,
                 "o", color='red', markersize=2)
    axes[0].plot(epi, reward_history,
                 "-", color='black', alpha=0.2)
    axes[0].set_xlim(xmin=0)

    loss_history_solved = np.asfarray(loss_history)
    loss_history_solved[[not solved for solved in solved_history]] = np.nan
    loss_history_unsolved = np.asfarray(loss_history)
    loss_history_unsolved[solved_history] = np.nan
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
