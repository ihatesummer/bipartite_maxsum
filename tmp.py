import torch
from torch.distributions import Normal
print(torch.cuda.is_available())

means = (torch.zeros((1, 5), requires_grad=True))
print(f"means: {means}")
pd = Normal(loc=means, scale=1)
print(f"pd: {pd}")
action = pd.sample()
print(f"action: {action}")
log_prob = pd.log_prob(action)
print(f"log_prob: {log_prob}")
