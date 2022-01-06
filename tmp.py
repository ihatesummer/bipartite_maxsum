import torch
from torch.distributions import Normal, Categorical

nn_probs = (torch.tensor([0.4, 0.2, 0.2, 0.1, 0.1]))
pd = Categorical(probs=nn_probs)
print(f"pd: {pd}")
action = pd.sample()
print(f"action: {action}")
log_prob = pd.log_prob(action)
print(f"log_prob: {log_prob}")