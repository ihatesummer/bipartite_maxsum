from matplotlib.pyplot import (
    axhline, show, subplots, xticks)
from numpy import (array, append, sqrt, shape,
                   mean, linspace, loadtxt)
from os.path import join
from os import getcwd
from math import factorial, floor


def n_perm_r(n, r):
    return int(factorial(n)/factorial(n-r))


def get_idx_vehicles(i):
    if i >= DIM_RIGHT/2:
        i = int(i - DIM_RIGHT/2)
    v1 = floor(i/(N_VEHICLE-1))
    v2 = i % (N_VEHICLE-1)+1*(i % (N_VEHICLE-1) >= v1)
    return array([v1, v2])


def get_anchor_idx(N_VEHICLE, DIM_LEFT, DIM_RIGHT):
    # Anchor in the middle
    pos_idx = [int(N_VEHICLE/2),
               int(DIM_LEFT/2+int(N_VEHICLE/2))]
    dist_idx = array([], dtype=int)
    for i in range(DIM_RIGHT):
        rx_veh, tx_veh = get_idx_vehicles_copy(i,
                                               DIM_RIGHT,
                                               N_VEHICLE)
        if tx_veh in pos_idx:
            dist_idx = append(dist_idx, i)
    return pos_idx, dist_idx


def get_idx_vehicles_copy(i, DIM_RIGHT, N_VEHICLE):  # tmp fix
    if i >= DIM_RIGHT/2:
        i = int(i - DIM_RIGHT/2)
    v1 = floor(i/(N_VEHICLE-1))
    v2 = i % (N_VEHICLE-1)+1*(i % (N_VEHICLE-1) >= v1)
    return array([v1, v2])


N_VEHICLE = 6
DIM_LEFT = int(N_VEHICLE*2)
DIM_RIGHT = int(n_perm_r(N_VEHICLE, 2) * 2)
(IDX_ANCHOR_POS,
    IDX_ANCHOR_DIST) = get_anchor_idx(N_VEHICLE,
                                      DIM_LEFT,
                                      DIM_RIGHT)

z_real = loadtxt(join(getcwd(), "z_real.csv"), delimiter=',')
z_mu = loadtxt(join(getcwd(), "z_mu.csv"), delimiter=',')
nrow, ncol = shape(z_mu)
xdiff = z_real[:int(nrow/2), :]-z_mu[:int(nrow/2), :]
ydiff = z_real[int(nrow/2):, :]-z_mu[int(nrow/2):, :]
l2_diff = sqrt(xdiff**2+ydiff**2)
print(nrow)
fig, ax = subplots()
markers = ['.', 'v', '^', '<', '>',
           '1', '2', '3', '4', 'p',
           'P', 'h', '+', 'x', 'D']
for row in range(0, int(nrow/2)):
    idx = get_idx_vehicles(row)
    ax.semilogy(linspace(0, ncol-1, ncol),
            l2_diff[row, :], markers[row%15], alpha=1,
            linewidth=1)

axhline(0.3, alpha=0.7, linewidth=5, color='black', label="Objective (30cm)")

l2_diff_avg = mean(l2_diff, axis=0)
ax.semilogy(linspace(0, ncol-1, ncol), l2_diff_avg, 's', markersize=10,
        color='red', linewidth=5, alpha=0.7, label="average")

ax.set_xlabel("iteration")
ax.set_ylabel("Localization error [m]")
ax.set_xlim(xmin=0, xmax=ncol-1)
ax.set_ylim(ymin=0, ymax=8)
xticks(linspace(0, ncol-1, ncol))
ax.grid()
ax.legend()
show()
