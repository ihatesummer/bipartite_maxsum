import numpy as np
import matplotlib.pyplot as plt

INF = 10^60  # infinity
DAMP = 0.0  # between 0 and 1. 0 for fastest change.
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

    for i in range(N_ITER):
        alpha = update_alpha(alpha, rho, w)
        rho = update_rho(alpha, rho, w)

    D = rho + alpha
    print(f"alpha:\n{alpha}")
    print(f"rho:\n{rho}")
    print(f"D:\n{D}")
    for row in range(N_NODE):
        idx_max = np.argmax(D[row, :])
        D[row, :] = 0
        D[row, idx_max] = 1

    is_valid = check_validity(D)
    if is_valid:
        print("Successful bipartite matching " + 
        f"with pairings as \n{D}")
        show_match(w, D)
    else:
        print("Pairing unsucessful.")


def log_sum_exp(input_array):
    return np.log(np.sum(np.exp(input_array)))


def update_alpha(alpha, rho, w):
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
                new[i, j] = w[i, j]/2 -max(tmp[i, :])
    return new*(1-DAMP) + old*(DAMP)


def update_rho(alpha, rho, w):
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
                new[i, j] = w[i, j]/2 -max(tmp[:, j])
    return new*(1-DAMP) + old*(DAMP)


def check_validity(D):
    rowsum = np.sum(D, axis=0)
    colsum = np.sum(D, axis=0)
    if np.all(rowsum==1) and np.all(colsum==1):
        return True
    else:
        return False


def show_match(w, D):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('Preferences')
    plt.imshow(w, origin='lower', cmap='gray')
    plt.colorbar(orientation='vertical')
    x = np.linspace(0, N_NODE-1, N_NODE)
    y = np.argmax(D, axis=1)
    plt.scatter(y, x,
                marker='d',
                color='red')
    plt.show()


if __name__=="__main__":
    main()
