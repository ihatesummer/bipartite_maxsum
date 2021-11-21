import numpy as np
import matplotlib.pyplot as plt

ub = np.linspace(0, 1, 21)
acc = np.ones(shape=np.shape(ub))

idx_080 = np.where(ub==0.80)[0][0]
idx_085 = idx_080 + 1
idx_090 = idx_085 + 1
idx_095 = idx_090 + 1
idx_100 = idx_095 + 1

acc[idx_080] = 0.99
acc[idx_085] = 0.9575
acc[idx_090] = 0.9055
acc[idx_095] = 0.7795
acc[idx_100] = 0.6210

acc_baseline = np.ones(shape=np.shape(ub))
acc_baseline[-1] = 0.995

fig, ax = plt.subplots()
markers = ['.', 'v', '^', '<', '>',
           '1', '2', '3', '4', 'p',
           'P', 'h', '+', 'x', 'D']

ax.plot(ub, acc_baseline*100,
        'v',
        alpha=1, color='black',
        markersize=5,
        label="Iterative search")
ax.plot(ub, acc*100,
        '+',
        alpha=1, color='red',
        markersize=10,
        label="Ultra-low latency")
# plt.axhline(1, alpha=0.7, linewidth=2, color='black', label="Iterative")

ax.set_xlabel("Relative link power")
ax.set_ylabel("Probability of finding optima")
ax.set_xlim(xmin=0, xmax=1.05)
ax.set_ylim(ymin=60, ymax=105)
ax.set_xticks(np.linspace(0,1,11))
ax.set_yticks(np.linspace(50,100,11))
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig("accuracy by problem difficulty.png")
