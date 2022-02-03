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
                datarate[n, N_NODE*i+j] = np.log2(1+1/(4*np.pi*dist**3))
        datarate[n] /= np.max(datarate[n])
    return datarate


if __name__=="__main__":
    main()
