from re import A
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os as os
from sklearn.model_selection import train_test_split
from maxsum_condensed import (update_alpha, update_rho,
                              conclude_update,
                              check_validity)

np.set_printoptions(precision=2)
N_NODE = 5  # number of nodes per group
N_ITER = N_NODE*10
N_DATASET = 100000
bLogSumExp = True
filenames = {
    "w": f"{N_NODE}-by-{N_NODE} - w.csv",
    "alpha_in": f"{N_NODE}-by-{N_NODE} - alpha_in.csv",
    "rho_in": f"{N_NODE}-by-{N_NODE} - rho_in.csv",
    "alpha_out": f"{N_NODE}-by-{N_NODE} - alpha_out - LogSumExp={bLogSumExp}.csv",
    "rho_out": f"{N_NODE}-by-{N_NODE} - rho_out - LogSumExp={bLogSumExp}.csv"
}

SEED_OFFSET = 10000000
SEED_W = 0
SEED_ALPHA = SEED_W + SEED_OFFSET
SEED_RHO = SEED_ALPHA + SEED_OFFSET


def check_dataset_availability():
    bAvailable = []
    for entry in filenames:
        filename = filenames[entry]
        bAvailable.append(os.path.exists(filename))
    return all(bAvailable)


def load_dataset():
    w = np.loadtxt(filenames['w'], dtype=float, delimiter=',')
    alpha_in = np.loadtxt(filenames['alpha_in'], dtype=float, delimiter=',')
    rho_in = np.loadtxt(filenames['rho_in'], dtype=float, delimiter=',')
    alpha_out = np.loadtxt(filenames['alpha_out'], dtype=float, delimiter=',')
    rho_out = np.loadtxt(filenames['rho_out'], dtype=float, delimiter=',')
    return w, alpha_in, rho_in, alpha_out, rho_out


def generate_and_save_dataset():
    w, alpha_in, rho_in = generate_dataset_input()
    np.savetxt(filenames['w'], w, delimiter=',')
    np.savetxt(filenames['alpha_in'], alpha_in, delimiter=',')
    np.savetxt(filenames['rho_in'], rho_in, delimiter=',')

    alpha_out, rho_out = generate_dataset_output(alpha_in, rho_in, w)
    np.savetxt(filenames['alpha_out'], alpha_out, delimiter=',')
    np.savetxt(filenames['rho_out'], rho_out, delimiter=',')
    return w, alpha_in, rho_in, alpha_out, rho_out


def generate_dataset_input():
    for i in range(N_DATASET):
        rng = np.random.default_rng(SEED_W+i)
        w_instance = rng.uniform(0, 1, (1, N_NODE**2))
        rng = np.random.default_rng(SEED_ALPHA+i)
        alpha_instance = rng.uniform(0, 1, (1, N_NODE**2))
        rng = np.random.default_rng(SEED_RHO+i)
        rho_instance = rng.uniform(0, 1, (1, N_NODE**2))
        if i==0:
            w = w_instance
            alpha = alpha_instance
            rho = rho_instance
        else:
            w = np.append(w, w_instance, axis=0)
            alpha = np.append(alpha, alpha_instance, axis=0)
            rho = np.append(rho, rho_instance, axis=0)
    return w, alpha, rho


def generate_dataset_output(alpha_in, rho_in, w):
    alpha_next = np.zeros(np.shape(alpha_in))
    rho_next = np.zeros(np.shape(rho_in))
    for i in range(N_DATASET):
        w_now = reshape_to_square(w[i])
        alpha_now = reshape_to_square(alpha_in[i])
        rho_now = reshape_to_square(rho_in[i])

        alpha_next[i] = np.reshape(update_alpha(alpha_now, rho_now, w_now, bLogSumExp),
                                   (1, N_NODE**2))
        rho_next[i] = np.reshape(update_rho(alpha_now, rho_now, w_now, bLogSumExp),
                                 (1, N_NODE**2))
    return alpha_next, rho_next


def reshape_to_square(flat_array):
    try:
        return np.reshape(flat_array, (N_NODE, N_NODE))
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


def print_train_inputs():
    print(f"weights:\n{w}")
    print(f"alpha_in:\n{alpha_in}")
    print(f"rho_in:\n{rho_in}")
    print(f"shapes:\n{np.shape(rho_in)}")
    print(f"collective:\n{dataset_x}")
    print(f"shape:\n{np.shape(dataset_x)}")


def print_train_outputs():
    print(f"alpha_out:\n{alpha_out}")
    print(f"rho_out:\n{rho_out}")
    print(f"shapes:\n{np.shape(rho_out)}")
    print(f"collective:\n{dataset_y}")
    print(f"shape:\n{np.shape(dataset_y)}")


def lol_just_trying_one_test():
    sample_input = np.array([dataset_x[0]])
    print(f"sample_input: {sample_input}")
    w, alpha, rho = decompose_dataset(sample_input[0], 'input')
    print(f"w: {w}")
    # print(f"alpha_in: {alpha}")
    # print(f"rho_in: {rho}")

    print("\nIterating NN...")
    # sample_prediction = model(sample_input).numpy()
    # alpha_out_pred, rho_out_pred = decompose_dataset(sample_prediction[0], 'output')
    # print(f"alpha_out_pred: {alpha_out_pred}")
    # print(f"rho_out_pred: {rho_out_pred}")
    w_flat = np.reshape(w, N_NODE**2)
    alpha_flat = np.reshape(alpha, N_NODE**2)
    rho_flat = np.reshape(rho, N_NODE**2)
    sample_prediction_and_w = np.array([
        np.concatenate((w_flat, alpha_flat, rho_flat))])
    for i in range(N_ITER):
        sample_prediction = model(sample_prediction_and_w)
        alpha_and_rho = sample_prediction.numpy()[0]
        sample_prediction_and_w = np.array([
            np.concatenate((w_flat, alpha_and_rho))
        ])

    alpha_final_pred, rho_final_pred = decompose_dataset(alpha_and_rho, 'output')
    print(f"alpha_pred: {alpha_final_pred}")
    print(f"rho_pred: {rho_final_pred}")
    D = conclude_update(alpha_final_pred, rho_final_pred)
    check_validity(D)


tic = time.time()


bDatasetAvailable = check_dataset_availability()

if not bDatasetAvailable:
    w, alpha_in, rho_in, alpha_out, rho_out = generate_and_save_dataset()
    print("Dataset generated.")
else:
    w, alpha_in, rho_in, alpha_out, rho_out = load_dataset()
    print("Dataset loaded.")

dataset_x = np.concatenate((w, alpha_in, rho_in), axis=1)
dataset_y = np.concatenate((alpha_out, rho_out), axis=1)
# print_train_inputs()
# print_train_outputs()

# x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.1, random_state=0)
# toc = time.time() - tic

# print(f"Dataset generation time: {toc}")

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(64, activation='relu'),
#   tf.keras.layers.Dense(2*(N_NODE**2))
# ])

# loss_fn = tf.keras.losses.MeanSquaredError()

# model.compile(optimizer='adam',
#               loss=loss_fn)
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test, verbose=2)

# lol_just_trying_one_test()
