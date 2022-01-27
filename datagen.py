import numpy as np
import matplotlib.pyplot as plt
from maxsum_rl import FILENAMES, N_DATASET, N_NODE, SEED_W


def main():
    map_size = 1
    pos_bs = generate_positions(SEED_W, map_size)
    pos_user = generate_positions(SEED_W+N_DATASET, map_size)
    w = get_w(pos_bs, pos_user)
    np.save(FILENAMES["pos_bs"], pos_bs)
    np.save(FILENAMES["pos_user"], pos_user)
    np.save(FILENAMES['w'], w)
    for i in range(N_DATASET):
        plot_positions(pos_bs[i], pos_user[i], i, map_size)


def generate_positions(random_seed, map_size):
    rng = np.random.default_rng(random_seed)
    pos = rng.uniform(0, map_size, (N_DATASET, N_NODE, 2))
    return pos


def get_w(pos_bs, pos_user):
    datarate = np.zeros((N_DATASET, N_NODE**2))
    for n in range(N_DATASET):
        for i in range(N_NODE):
            for j in range(N_NODE):
                dist = np.linalg.norm(
                    pos_bs[n, i] - pos_user[n, j], 2)
                datarate[n, N_NODE*i+j] = np.log2(1+dist**-3)
        datarate[n] /= np.max(datarate[n])
    return datarate


def plot_positions(bs_i, user_i, idx, map_size):
    plt.title("BS and user positions")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim(0, map_size)
    plt.ylim(0, map_size)
    for i in range(N_NODE):
        plt.plot(bs_i[i, 0], bs_i[i, 1], 'b',
                 marker=f"${i+1}$", markersize=12)
        plt.plot(user_i[i, 0], user_i[i, 1], 'r',
                 marker=f"${i+1}$", markersize=12)
    plt.savefig(f"training/data{idx}_pos.png")
    plt.clf()
    plt.close("all")


if __name__=="__main__":
    main()
