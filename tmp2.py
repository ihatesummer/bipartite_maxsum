import numpy as np
from maxsum import reshape_to_square
from maxsum_ul import forward_pass, N_NODE, get_datasets, FILENAMES, N_DATASET, SEED_W, decompose_dataset


D_rl_now = np.array([[0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1]])
D_mp_now = np.array([[0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1]])
print((D_rl_now == D_mp_now).all())
