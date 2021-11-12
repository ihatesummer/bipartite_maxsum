import numpy as np
import matplotlib.pyplot as plt

INF = 10^60  # infinity
DAMP = 0.0  # between 0 and 1. 0 for fastest change.
N_NODE = 30  # number of nodes per group
N_ITER = N_NODE*10
np.set_printoptions(precision=2)


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
    plt.imshow(w, origin='lower', cmap='jet')
    plt.colorbar(orientation='vertical')
    x = np.linspace(0, N_NODE-1, N_NODE)
    y = np.argmax(D, axis=1)
    plt.scatter(y, x,
                marker='o',
                color='black')
    plt.show()


def main():
    rng = np.random.default_rng(0)
    w = rng.uniform(0, 1, (N_NODE, N_NODE))
    print(f"weights:\n{w}")
    beta = np.zeros((N_NODE, N_NODE))
    eta = np.zeros((N_NODE, N_NODE))
    rho = np.zeros((N_NODE, N_NODE))
    alpha = np.zeros((N_NODE, N_NODE))

    for i in range(N_ITER):
        beta = update_beta(beta, alpha, w)
        rho = update_rho(rho, eta, w)
        alpha = update_alpha(alpha, rho)
        eta = update_eta(eta, beta)

    D = eta + alpha + w
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


if __name__=="__main__":
    main()
