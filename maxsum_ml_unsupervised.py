import tensorflow as tf
import numpy as np
import time
import os as os
from sklearn.model_selection import train_test_split
from maxsum_condensed import (update_alpha, update_rho,
                              conclude_update,
                              check_validity,
                              show_match)

np.set_printoptions(precision=2)
N_NODE = 5  # number of nodes per group
N_ITER = N_NODE*10
N_DATASET = 100000
bLogSumExp = False
filenames = {
    "w": f"{N_NODE}-by-{N_NODE} - w.csv",
    "alpha_star": f"{N_NODE}-by-{N_NODE} - alpha_star - LogSumExp={bLogSumExp}.csv",
    "rho_star": f"{N_NODE}-by-{N_NODE} - rho_star - LogSumExp={bLogSumExp}.csv"
}
FILENAME_NN_WEIGHT = "weights_unsupervised.h5"

SEED_OFFSET = 10000000
SEED_W = 0
SEED_ALPHA = SEED_W + SEED_OFFSET
SEED_RHO = SEED_ALPHA + SEED_OFFSET



def main():
    (w, alpha_star, rho_star) = fetch_dataset()
    dataset_w = w
    dataset_alpha_rho_star = np.concatenate((alpha_star, rho_star), axis=1)
    # print_dataset_dimensions(w, alpha_star, rho_star,
    #                          dataset_w, dataset_alpha_rho_star)
    (w_train, w_test,
     alpha_rho_star_train,
     alpha_rho_star_test) = train_test_split(dataset_w, dataset_alpha_rho_star,
                                             test_size=0.2, shuffle=False)
    model = initialize_model()
    loss(model, w_test, alpha_rho_star_test)
    # _, mse = model.evaluate(x_test, y_test, verbose=0)
    # print(f"Test set MSE: {mse}")

    # run_test_matching(x_test, y_test_star, model)
 

def fetch_dataset():
    bDatasetAvailable = check_dataset_availability()
    if not bDatasetAvailable:
        (w, alpha_star, rho_star) = generate_and_write_dataset()
        print("Dataset generated.")
    else:
        (w, alpha_star, rho_star) = read_dataset()
        print("Dataset loaded.")
    return (w, alpha_star, rho_star)


def check_dataset_availability():
    bAvailable = []
    for entry in filenames:
        filename = filenames[entry]
        bAvailable.append(os.path.exists(filename))
    return all(bAvailable)


def generate_and_write_dataset():
    w = generate_dataset_input()
    np.savetxt(filenames['w'], w, delimiter=',')
    alpha_star, rho_star = generate_dataset_output(w)
    np.savetxt(filenames['alpha_star'], alpha_star, delimiter=',')
    np.savetxt(filenames['rho_star'], rho_star, delimiter=',')
    return (w, alpha_star, rho_star)


def generate_dataset_input():
    for i in range(N_DATASET):
        rng = np.random.default_rng(SEED_W+i)
        w_instance = rng.uniform(0, 1, (1, N_NODE**2))
        if i==0:
            w = w_instance
        else:
            w = np.append(w, w_instance, axis=0)
    return w


def generate_dataset_output(w):
    alpha_star = np.zeros(np.shape(w))
    rho_star = np.zeros(np.shape(w))
    for i in range(N_DATASET):
        w_now = reshape_to_square(w[i])
        alpha_tmp = np.zeros(np.shape(w_now))
        rho_tmp = np.zeros(np.shape(w_now))
        for j in range(N_ITER):
            alpha_tmp = update_alpha(
                alpha_tmp, rho_tmp, w_now, bLogSumExp=False)
            rho_tmp = update_rho(
                alpha_tmp, rho_tmp, w_now, bLogSumExp=False)
        alpha_star[i] = reshape_to_flat(alpha_tmp)
        rho_star[i] = reshape_to_flat(rho_tmp)
    return alpha_star, rho_star


def read_dataset():
    w = np.loadtxt(filenames['w'], dtype=float, delimiter=',')
    alpha_star = np.loadtxt(filenames['alpha_star'], dtype=float, delimiter=',')
    rho_star = np.loadtxt(filenames['rho_star'], dtype=float, delimiter=',')
    return w, alpha_star, rho_star


def reshape_to_square(flat_array):
    try:
        return np.reshape(flat_array, (N_NODE, N_NODE))
    except Exception as e:
        print(f"ERROR: array reshaping failed: {e}")


def reshape_to_flat(square_array):
    try:
        return np.reshape(square_array, N_NODE**2)
    except Exception as e:
        print(f"ERROR: array reshaping failed: {e}")
    

def decompose_dataset(arr, mode):
    if mode == 'input':
        w, alpha, rho = np.array_split(arr, 3)
        for _ in [w, alpha, rho]:
            w = reshape_to_square(w)
            alpha = reshape_to_square(alpha)
            rho = reshape_to_square(rho)
        return w, alpha, rho
    elif mode == 'output':
        alpha, rho = np.array_split(arr, 2)
        for _ in [alpha, rho]:
            alpha = reshape_to_square(alpha)
            rho = reshape_to_square(rho)
        return alpha, rho
    else:
        pass


def print_dataset_dimensions(w, alpha_star, rho_star,
                             dataset_w, dataset_alpha_rho_star):
    print("Shapes of ...")
    print(f"weights:\n{np.shape(w)}")
    print(f"alpha_star:\n{np.shape(alpha_star)}")
    print(f"rho_star:\n{np.shape(rho_star)}")
    print(f"dataset_x:\n{np.shape(dataset_w)}")
    print(f"dataset_y:\n{np.shape(dataset_alpha_rho_star)}")


def initialize_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2*(N_NODE**2))
    ])
    return model

    
def loss(model, w, alpha_rho_star):
    alpha_rho = model(w)
    nData = np.size(alpha_rho, axis=0)
    # l1_errors = np.zeros(nData)
    l1_errors = np.zeros(5)
    # for i in range(nData):
    for i in range(5):
        alpha, rho = decompose_dataset(
            alpha_rho[i], 'output')
        alpha_next = update_alpha(alpha, rho,
                                  reshape_to_square(w[i]),
                                  bLogSumExp=True)
        rho_next = update_rho(alpha, rho,
                              reshape_to_square(w[i]),
                              bLogSumExp=True)

        alpha_next = reshape_to_flat(alpha_next)
        rho_next = reshape_to_flat(rho_next)
        alpha_rho_next = np.concatenate((alpha_next, rho_next))
        l1_errors[i] = np.sum(
            np.abs(alpha_rho_star[i] - alpha_rho_next))
        
    print(l1_errors)
    print(np.mean(l1_errors))


def remove_invalid_samples(arr, idx_invalid):
    return np.delete(arr, idx_invalid, axis=0)


def construct_nn_input(w, alpha_rho):
    w_alpha_rho = np.array([
        np.concatenate((w, alpha_rho))
    ])
    return w_alpha_rho


def run_test_matching(x_test, y_test_star, model):
    n_test_samples = np.size(x_test, axis=0)
    D_msgPassing, D_msgPassing_validity = get_D_msgPassing(
        n_test_samples, y_test_star)

    idx_invalid_samples = np.where(D_msgPassing_validity==False)[0]
    
    D_msgPassing = remove_invalid_samples(D_msgPassing,
                                          idx_invalid_samples)
    
    D_ann, D_ann_validity = get_D_ann(n_test_samples, x_test, model)
    print("test accuracy[%]: ", np.count_nonzero(D_ann_validity)/n_test_samples*100)
    print(f"i.e. {np.count_nonzero(D_ann_validity)} out of {n_test_samples}")


def get_D_ann(n_test_samples, x_test, model):
    D_ann = np.zeros((n_test_samples, N_NODE**2), dtype=int)
    D_ann_validity = np.zeros(n_test_samples, dtype=bool)
    for i in range(n_test_samples):
        w, _, _ = decompose_dataset(x_test[i], 'input')
        w = reshape_to_flat(w)
        alpha_rho = np.repeat(np.zeros(np.shape(w)), 2)
        w_alpha_rho = construct_nn_input(w, alpha_rho)
        for j in range(N_ITER):
            alpha_rho = model(w_alpha_rho).numpy()[0]
            w_alpha_rho = construct_nn_input(w, alpha_rho)
        alpha_converged, rho_converged = decompose_dataset(
            alpha_rho, 'output')        
        D_real = conclude_update(alpha_converged, rho_converged)
        is_valid = check_validity(D_real)
        D_ann[i] = reshape_to_flat(D_real)
        D_ann_validity[i] = is_valid
    return D_ann, D_ann_validity


def get_D_msgPassing(n_test_samples, y_test_star):
    D_msgPassing = np.zeros((n_test_samples, N_NODE**2), dtype=int)
    D_msgPassing_validity = np.zeros((n_test_samples, 1), dtype=bool)
    for i in range(n_test_samples):
        alpha_star, rho_star = decompose_dataset(y_test_star[i], 'output')
        D_real = conclude_update(alpha_star, rho_star)
        is_valid = check_validity(D_real)
        D_msgPassing[i] = reshape_to_flat(D_real)
        D_msgPassing_validity[i] = is_valid
    return D_msgPassing, D_msgPassing_validity


if __name__=="__main__":
    tic = time.time()
    main()
    toc = time.time()
    print(f"Runtime: {toc-tic}sec.")
