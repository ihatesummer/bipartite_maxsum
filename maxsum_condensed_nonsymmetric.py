import numpy as np
import matplotlib.pyplot as plt
import time
from maxsum import reshape_to_flat, check_validity, get_pairing_matrix_argmax, show_match
from maxsum_condensed import show_msg_changes_2, log_sum_exp

INF = 10**60  # infinity
DAMP = 0  # between 0 and 1 (0 for fastest change)
N_NODE = 5  # number of nodes per group
N_ITER = N_NODE*10
bLogSumExp = False
np.set_printoptions(precision=2)


def main():
    rng = np.random.default_rng(0)
    w = rng.uniform(0, 1, (N_NODE, N_NODE))
    print(f"weights:\n{w}")
    alpha = np.zeros((N_NODE, N_NODE))
    rho = np.zeros((N_NODE, N_NODE))
    alpha_history = np.zeros((N_NODE**2, N_ITER))
    rho_history = np.zeros((N_NODE**2, N_ITER))

    tic = time.time()
    for i in range(N_ITER):
        alpha = update_alpha(alpha, rho, w, bLogSumExp)
        rho = update_rho(alpha, rho, w, bLogSumExp)
        alpha_history[:, i] = reshape_to_flat(alpha)
        rho_history[:, i] = reshape_to_flat(rho)
    
    show_msg_changes_2(alpha_history, rho_history)

    print(f"alpha + rho: \n{alpha+rho}")
    D = get_pairing_matrix_argmax(alpha, rho, N_NODE)
    is_valid = check_validity(D)
    toc = time.time()
    print(f"matching time: {(toc - tic)*1000}ms")
    if is_valid:
        print("Successful bipartite matching " +
              f"with pairings as \n{D}")
        # show_match(w, D)
    else:
        print("Pairing unsucessful.")


def update_alpha(alpha, rho, w, bLogSumExp):
    old = alpha
    new = np.zeros(((N_NODE, N_NODE)))
    for i in range(N_NODE):
        for j in range(N_NODE):
            tmp = np.copy(rho)
            if bLogSumExp:
                tmp_ith_row_except_ij = np.delete(tmp[i, :], j)
                new[i, j] = -log_sum_exp(tmp_ith_row_except_ij)
            else:
                tmp[i, j] = -INF
                new[i, j] = -max(tmp[i, :])
    return new*(1-DAMP) + old*(DAMP)


def update_rho(alpha, rho, w, bLogSumExp):
    old = rho
    new = np.zeros(((N_NODE, N_NODE)))
    for i in range(N_NODE):
        for j in range(N_NODE):
            tmp = alpha + w
            if bLogSumExp:
                tmp_jth_col_except_ij = np.delete(tmp[:, j], i)
                new[i, j] = w[i, j] - log_sum_exp(tmp_jth_col_except_ij)
            else:
                tmp[i, j] = -INF
                new[i, j] = w[i, j] - max(tmp[:, j])
    return new*(1-DAMP) + old*(DAMP)


if __name__=="__main__":
    main()
