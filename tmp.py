import torch
from torch.distributions import Normal
print(torch.cuda.is_available())

means = (torch.ones((1, 50), requires_grad=True))
print(f"means: {means}")
pd = Normal(loc=means, scale=0.5)
print(f"pd: {pd}")
action = pd.sample()
print(f"action: {action}")
log_prob = pd.log_prob(action)
print(f"log_prob: {log_prob}")
print(f"sum: {torch.sum(log_prob)}")
print(f"sum*30: {torch.sum(log_prob)*30}")
