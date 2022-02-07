# %%
from re import I
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import time

INF = 10**60
DAMP = 0.5  # between 0 and 1 (0 for fastest change)
bLogSumExp = False
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
matplotlib.use('Agg')

def main():
    idx = np.linspace(0, 10, 11, dtype=int)
    n_node = 10
    n_iter = 15
    w_ds = np.load(f"{n_node}x{n_node}_w.npy")
    pos_bs = np.load(f"{n_node}x{n_node}_bs_pos.npy")
    pos_user = np.load(f"{n_node}x{n_node}_user_pos.npy")

    # print(f"Data{idx} weights:\n"
    #       f"{reshape_to_square(w_ds[idx], n_node)}")
    times = []
    for i in idx:
        tic = time.time()
        alpha_hist, rho_hist = iterate_maxsum_mp(w_ds[i], n_iter, n_node)

        alpha, rho = (reshape_to_square(alpha_hist[-1], n_node),
                    reshape_to_square(rho_hist[-1], n_node))

        D_mp = get_pairing_matrix_argmax(alpha, rho, n_node)
        times.append(time.time()-tic)
        is_valid = check_pairing_validity(D_mp)
        if is_valid:
            print(f"Pairing success:\n{D_mp}")
        else:
            print("Pairing failed.")

        show_mp_traj(alpha_hist, rho_hist, i, n_node)
        show_mp_error(alpha_hist, rho_hist, i, n_node)
        plot_positions(pos_bs[i], pos_user[i], i, n_node, map_size=1)
    print("Avg time:", np.mean(times))


def iterate_maxsum_mp(w, n_iter, n_node):
    w = reshape_to_square(w, n_node)
    alpha = np.zeros((n_node, n_node))
    rho = np.zeros((n_node, n_node))
    alpha_history = np.zeros((n_iter, n_node**2))
    rho_history = np.zeros((n_iter, n_node**2))

    for i in range(n_iter):
        alpha_history[i] = reshape_to_flat(alpha, n_node)
        rho_history[i] = reshape_to_flat(rho, n_node)
        alpha_next = update_alpha(alpha, rho, w, bLogSumExp=False)
        rho_next = update_rho(alpha, rho, w, bLogSumExp=False)
        alpha, rho = alpha_next, rho_next
        # print_msg(alpha, rho, i)

    return alpha_history, rho_history


def reshape_to_flat(square_array, n_node):
    return np.reshape(square_array, n_node**2)


def reshape_to_square(flat_array, n_node):
    return np.reshape(flat_array, (n_node, n_node))


def log_sum_exp(input_array):
    return np.log(np.sum(np.exp(input_array)))


def update_alpha(alpha, rho, w, bLogSumExp):
    n_node = np.size(alpha, axis=0)
    old = alpha
    new = np.zeros(((n_node, n_node)))
    for i in range(n_node):
        for j in range(n_node):
            tmp = rho + w/2
            if bLogSumExp:
                tmp_ith_row_except_ij = np.delete(tmp[i, :], j)
                new[i, j] = w[i, j]/2 - log_sum_exp(tmp_ith_row_except_ij)
            else:
                tmp[i, j] = -INF
                new[i, j] = w[i, j]/2 - max(tmp[i, :])      
    return new*(1-DAMP) + old*(DAMP)


def update_rho(alpha, rho, w, bLogSumExp):
    n_node = np.size(alpha, axis=0)
    old = rho
    new = np.zeros(((n_node, n_node)))
    for i in range(n_node):
        for j in range(n_node):
            tmp = alpha + w/2
            if bLogSumExp:
                tmp_jth_col_except_ij = np.delete(tmp[:, j], i)
                new[i, j] = w[i, j]/2 - log_sum_exp(tmp_jth_col_except_ij)
            else:
                tmp[i, j] = -INF
                new[i, j] = w[i, j]/2 - max(tmp[:, j])
    return new*(1-DAMP) + old*(DAMP)


def print_msg(alpha, rho, i):
    print(f"\n{i}th iteration:")
    print(f"alpha:\n{alpha}")
    print(f"rho:\n{rho}")
    print(f"alpha + rho: \n{alpha+rho}")


def get_pairing_matrix_argmax(alpha, rho, n_node):
    pair_matrix = rho + alpha
    for row in range(n_node):
        idx_max = np.argmax(pair_matrix[row])
        pair_matrix[row, :] = 0
        pair_matrix[row, idx_max] = 1
    return pair_matrix.astype(int)


def get_pairing_matrix_sign(alpha, rho):
    sum = rho + alpha
    pair_matrix = (sum>0).astype(int)
    return pair_matrix


def check_pairing_validity(D):
    rowsum = np.sum(D, axis=0)
    colsum = np.sum(D, axis=1)
    if np.all(rowsum==1) and np.all(colsum==1):
        return True
    else:
        return False


def plot_positions(bs_i, user_i, idx, n_node, map_size):
    plt.title("BS and user positions")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim(0, map_size)
    plt.ylim(0, map_size)
    for i in range(n_node):
        plt.plot(bs_i[i, 0], bs_i[i, 1], 'b',
                 marker=f"${i+1}$", markersize=12)
        plt.plot(user_i[i, 0], user_i[i, 1], 'r',
                 marker=f"${i+1}$", markersize=12)
    plt.savefig(f"training/data{idx}_pos.png")
    plt.clf()
    plt.close("all")


def show_mp_traj(alpha_hist, rho_hist, idx, n_node):
    _, axes = plt.subplots(nrows=n_node, ncols=2,
                           figsize=(10, 12),
                           tight_layout=True)
    axes[0, 0].set_title(r"$\alpha$")
    axes[0, 1].set_title(r"$\rho$")
    n_iter = np.size(alpha_hist, axis=0)
    t = np.linspace(0, n_iter-1, n_iter)
    marker = itertools.cycle(("$1$", "$2$", "$3$", "$4$", "$5$"))
    for i in range(n_node):
        for j in range (n_node):
            mk = next(marker)
            axes[i, 0].plot(t, alpha_hist[:, n_node*i+j],
                            marker=mk)
            axes[i, 1].plot(t, rho_hist[:, n_node*i+j],
                            marker=mk)
        axes[i, 0].set_xlim(xmin=0, xmax=n_iter-1)
        axes[i, 1].set_xlim(xmin=0, xmax=n_iter-1)
    plt.savefig(f"training/data{idx}_mp_traj.png")
    plt.close('all')


def show_mp_error(alpha_hist, rho_hist, idx, n_node):
    alpha_hist_shift = np.vstack((alpha_hist[1:], np.zeros(n_node**2)))
    rho_hist_shift = np.vstack((rho_hist[1:], np.zeros(n_node**2)))
    alpha_diff_hist = np.abs((alpha_hist_shift - alpha_hist)[:-1])
    rho_diff_hist = np.abs((rho_hist_shift - rho_hist)[:-1])

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6),
                           tight_layout=True)
    axes[0].set_title(r"$\|f(\rho, w) - \alpha\|$")
    axes[1].set_title(r"$\|g(\alpha, w) - \rho\|$")
    t_max = np.size(alpha_diff_hist, axis=0)
    t = np.linspace(0, t_max-1, t_max)
    for node_ij in range(n_node**2):
        axes[0].semilogy(t, alpha_diff_hist[:, node_ij],
                         "-*", linewidth=1, color='red', alpha=0.5)
        axes[0].set_xlim(xmin=0, xmax=t_max-1)
        axes[1].semilogy(t, rho_diff_hist[:, node_ij],
                         "-*", linewidth=1, color='green', alpha=0.5)
        axes[1].set_xlim(xmin=0, xmax=t_max-1)
    plt.savefig(f"training/data{idx}_mp_err.png")
    plt.close('all')


if __name__=="__main__":
    main()
