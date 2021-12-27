import numpy as np
import matplotlib.pyplot as plt
import time
from maxsum import reshape_to_flat, reshape_to_square, check_validity, show_match, get_pairing_matrix

INF = 10**60  # infinity
DAMP = 0  # between 0 and 1 (0 for fastest change)
N_NODE = 5  # number of nodes per group
N_ITER = 20
bLogSumExp = False
np.set_printoptions(precision=2)


def main():
    rng = np.random.default_rng(1)
    w = rng.uniform(0, 1, (N_NODE, N_NODE))
    print(f"weights:\n{w}")
    # alpha = np.zeros((N_NODE, N_NODE))
    # rho = np.zeros((N_NODE, N_NODE))
    alpha = w/2
    rho = -w/2
    alpha_history = np.zeros((N_NODE**2, N_ITER))
    rho_history = np.zeros((N_NODE**2, N_ITER))

    tic = time.time()
    for i in range(N_ITER):
        alpha = update_alpha(alpha, rho, w, bLogSumExp)
        rho = update_rho(alpha, rho, w, bLogSumExp)
        alpha_history[:, i] = reshape_to_flat(alpha, N_NODE)
        rho_history[:, i] = reshape_to_flat(rho, N_NODE)
        if i>0:
            print(f"\n{i}th iteration:")
            print(f"alpha diff:\n{reshape_to_square(alpha_history[:,i]-alpha_history[:,i-1], N_NODE)}")
            print(f"rho diff:\n{reshape_to_square(rho_history[:,i]-rho_history[:,i-1], N_NODE)}")
            print(f"alpha + rho: \n{alpha+rho}")
    
    show_msg_changes_2(alpha_history, rho_history)

    print(f"alpha + rho: \n{alpha+rho}")
    D = get_pairing_matrix(alpha, rho)
    is_valid = check_validity(D)
    toc = time.time()
    print(f"matching time: {(toc - tic)*1000}ms")
    if is_valid:
        print("Successful bipartite matching " +
              f"with pairings as \n{D}")
        # show_match(w, D)
    else:
        print("Pairing unsucessful.")


def log_sum_exp(input_array):
    return np.log(np.sum(np.exp(input_array)))


def update_alpha(alpha, rho, w, bLogSumExp):
    old = alpha
    new = np.zeros(((N_NODE, N_NODE)))
    for i in range(N_NODE):
        for j in range(N_NODE):
            tmp = rho + w/2
            if bLogSumExp:
                tmp_ith_row_except_ij = np.delete(tmp[i, :], j)
                new[i, j] = w[i, j]/2 - log_sum_exp(tmp_ith_row_except_ij)
            else:
                tmp[i, j] = -INF
                new[i, j] = w[i, j]/2 - max(tmp[i, :])
                
    return new*(1-DAMP) + old*(DAMP)


def update_rho(alpha, rho, w, bLogSumExp):
    old = rho
    new = np.zeros(((N_NODE, N_NODE)))
    for i in range(N_NODE):
        for j in range(N_NODE):
            tmp = alpha + w/2
            if bLogSumExp:
                tmp_jth_col_except_ij = np.delete(tmp[:, j], i)
                new[i, j] = w[i, j]/2 - log_sum_exp(tmp_jth_col_except_ij)
            else:
                tmp[i, j] = -INF
                new[i, j] = w[i, j]/2 - max(tmp[:, j])
    return new*(1-DAMP) + old*(DAMP)


def show_msg_changes_2(alpha_history,
                       rho_history):
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    axes[0].set_title('alpha')
    axes[1].set_title('rho')
    x = np.linspace(0, N_ITER-1, N_ITER)
    for node_ij in range(N_NODE**2):
        axes[0].plot(x, alpha_history[node_ij, :],
                     "-o", markersize=3,
                     color='red', alpha=0.2)
        axes[0].set_xlim(xmin=0, xmax=N_ITER)
        axes[1].plot(x, rho_history[node_ij, :],
                     "-o", markersize=3,
                     color='green', alpha=0.2)
        axes[1].set_xlim(xmin=0, xmax=N_ITER)
    plt.show()


if __name__=="__main__":
    main()
