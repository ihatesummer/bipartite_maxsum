import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
GAMMA = 1.0

def main():
    dim_in = 1
    dim_out = 1
    pi = Pi(dim_in, dim_out)
    learning_rate = 1e-2
    optimizer = optim.Adam(pi.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss(reduction='sum')
    for epi in range(100):
        print(f"\nepi: {epi}")
        print(f"weight: {pi.model[0].weight}")
        print(f"bias: {pi.model[0].bias}")
        state = torch.ones(1)
        for t in range(1):
            print(f"t: {t}")
            action = pi.act(state)
            state = state+action
            print(f"new state: {state}")
            target = torch.ones(1)*2
            reward = -loss_fn(state , target)
            pi.rewards.append(reward)
        loss = train(pi, optimizer)
        pi.onpolicy_reset()
            


class Pi(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(dim_in, dim_out)
        ]
        self.model=nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # set training mode

    def onpolicy_reset(self):
        self.log_probs=[]
        self.rewards=[]

    def act(self, state):
        means = self.model(state)
        print(f"means: {means}")
        # pd = Normal(loc=means, scale=0.001)
        # action = pd.sample()
        # log_prob = pd.log_prob(action)
        # log_prob_joint = torch.sum(log_prob)
        # self.log_probs.append(log_prob_joint)  # store for training
        # print(f"action: {action}")
        return means

def train(pi, optimizer):
    T = len(pi.rewards)
    returns = torch.zeros(T)
    future_returns = 0.0
    for t in reversed(range(T)):
        future_returns = pi.rewards[t] + GAMMA * future_returns
        returns[t] = future_returns
    # log_probs = torch.stack(pi.log_probs)
    # loss = - log_probs * returns
    loss = - returns
    print(f"rewards: {pi.rewards}")
    print(f"returns: {returns}")
    # print(f"log_probs: {log_probs}")
    # loss = - log_probs * (returns-return_baseline)
    loss = torch.sum(loss)
    print(f"loss: {loss}")
    optimizer.zero_grad()
    loss.backward()  # backpropagate, compute gradients
    optimizer.step()  # gradient-ascent, update the weights
    return loss

if __name__ == '__main__':
    main()
