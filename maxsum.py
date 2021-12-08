import numpy as np
import matplotlib.pyplot as plt

INF = 10**60  # infinity
DAMP = 0.0  # between 0 and 1. 0 for fastest change.
N_NODE = 5  # number of nodes per group
# N_ITER = N_NODE*10
N_ITER = 50
np.set_printoptions(precision=2)


def main():
    rng = np.random.default_rng(0)
    w = rng.uniform(0, 1, (N_NODE, N_NODE))
    # w = np.array(
    #     [[2, 4, 2, 2, 2],
    #      [4, 2, 2, 2, 2],
    #      [2, 2, 4, 2, 2],
    #      [2, 2, 2, 4, 2],
    #      [2, 2, 2, 2, 4]])
    print(f"weights:\n{w}")
    alpha = np.zeros((N_NODE, N_NODE))
    eta = np.zeros((N_NODE, N_NODE))
    rho = np.zeros((N_NODE, N_NODE))
    beta = np.zeros((N_NODE, N_NODE))
    # alpha = init_normal(rng)
    # eta = init_normal(rng)
    # rho = init_normal(rng)
    # beta = init_normal(rng)

    alpha_history = alloc_history_arr()
    eta_history = alloc_history_arr()
    rho_history = alloc_history_arr()
    beta_history = alloc_history_arr()

    for i in range(N_ITER):
        print("="*10 + f"iter {i}" + "="*10)
        D_old = eta + alpha + w
        alpha_old = alpha
        alpha = update_alpha(alpha, rho)
        eta = update_eta(eta, beta)
        rho = update_rho(rho, eta, w)
        beta = update_beta(beta, alpha, w)
        print(f"alpha:\n{alpha}")
        # print(f"eta:\n{eta}")
        # print(f"rho:\n{rho}")
        # print(f"beta:\n{beta}")
        D_now = eta + alpha + w
        alpha_history[:, i] = reshape_to_flat(alpha)
        eta_history[:, i] = reshape_to_flat(eta)
        rho_history[:, i] = reshape_to_flat(rho)
        beta_history[:, i] = reshape_to_flat(beta)

        print("mean(alpha_new-alpha_old): ",
              np.mean(np.abs(alpha_history[:, i]
               - alpha_history[:, i-1])))
    
    show_msg_changes(alpha_history,
                     eta_history,
                     rho_history,
                     beta_history)

    D_final = eta + alpha + w
    print(f"D:\n{D_final}")
    for row in range(N_NODE):
        idx_max = np.argmax(D_final[row, :])
        D_final[row, :] = 0
        D_final[row, idx_max] = 1

    is_valid = check_validity(D_final)
    if is_valid:
        print("Successful bipartite matching " + 
        f"with pairings as \n{D_final}")
        # show_match(w, D)
    else:
        print("Pairing unsucessful.")


def alloc_history_arr():
    return np.zeros((N_NODE**2, N_ITER))


def init_normal(rng):
    return rng.normal(size=(N_NODE, N_NODE))


def init_zeros():
    return np.zeros(N_NODE, N_NODE)


def reshape_to_flat(square_array):
    return np.reshape(square_array, N_NODE**2)


def update_beta(beta, alpha, w):
    old = beta
    new = w + alpha
    return new*(1-DAMP) + old*(DAMP)


def update_rho(rho, eta, w):
    old = rho
    new = w + eta
    return new*(1-DAMP) + old*(DAMP)


def update_alpha(alpha, rho):
    old = alpha
    new = np.zeros(((N_NODE, N_NODE)))
    for i in range(N_NODE):
        for j in range(N_NODE):
            tmp = np.copy(rho)
            tmp[i, j] = -INF
            new[i, j] = -max(tmp[i, :])
    return new*(1-DAMP) + old*(DAMP)


def update_eta(eta, beta):
    old = eta
    new = np.zeros(((N_NODE, N_NODE)))
    for i in range(N_NODE):
        for j in range(N_NODE):
            tmp = np.copy(beta)
            tmp[i, j] = -INF
            new[i, j] = -max(tmp[:, j])
    return new*(1-DAMP) + old*(DAMP)


def show_msg_changes(alpha_history,
                     eta_history,
                     rho_history,
                     beta_history):
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
    axes[0, 0].set_title('alpha')
    axes[0, 1].set_title('eta')
    axes[1, 0].set_title('rho')
    axes[1, 1].set_title('beta')
    x = np.linspace(0, N_ITER-1, N_ITER)
    for node_ij in range(N_NODE**2):
        axes[0, 0].plot(x, alpha_history[node_ij, :],
                     "-o", color='red', alpha=0.2)
        axes[0, 0].set_xlim(xmin=0, xmax=N_ITER)
        axes[0, 1].plot(x, eta_history[node_ij, :],
                     "-o", color='blue', alpha=0.2)
        axes[0, 1].set_xlim(xmin=0, xmax=N_ITER)
        axes[1, 0].plot(x, rho_history[node_ij, :],
                     "-o", color='green', alpha=0.2)
        axes[1, 0].set_xlim(xmin=0, xmax=N_ITER)
        axes[1, 1].plot(x, beta_history[node_ij, :],
                     "-o", color='brown', alpha=0.2)
        axes[1, 1].set_xlim(xmin=0, xmax=N_ITER)
    plt.show()


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
