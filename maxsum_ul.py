import tensorflow as tf
import numpy as np
import time
import os as os
from sklearn.model_selection import train_test_split
from maxsum_condensed import (update_alpha, update_rho,
                              get_pairing_matrix,
                              check_validity)

np.set_printoptions(precision=2)
N_NODE = 5  # number of nodes per group
N_ITER = 20
N_DATASET = 10000
bLogSumExp = False
filenames = {
    "w": f"{N_NODE}-by-{N_NODE} - w.csv",
    "alpha_star": f"{N_NODE}-by-{N_NODE} - alpha_star.csv",
    "rho_star": f"{N_NODE}-by-{N_NODE} - rho_star.csv"
}
FILENAME_NN_WEIGHT = "weights_unsupervised.h5"
# FILENAME_NN_WEIGHT = "weights_unsupervised_gyh_30.h5"
SEED_W = 0


def main():
    (w, alpha_star, rho_star) = fetch_dataset()
    dataset_w = w
    dataset_alpha_rho_star = np.concatenate((alpha_star,
                                             rho_star),
                                            axis=1)

    (w_train, w_test,
     alpha_rho_star_train,
     alpha_rho_star_test) = train_test_split(
         dataset_w[:N_DATASET, :],
         dataset_alpha_rho_star[:N_DATASET, :],
         test_size=0.2, shuffle=True)

    w_train_tensor = tf.data.Dataset.from_tensor_slices((w_train))
    model = initialize_model()
    try:
        model.load_weights(FILENAME_NN_WEIGHT)
        print("Trained NN weights loaded.")
        print(model.summary())
    except:
        print("Training NN.")
        optimizer = tf.keras.optimizers.Nadam()
        loss_fxn = tf.keras.losses.MeanAbsoluteError()
        n_epochs = 20
        batch_size = 2000

        w_train_minibatches = w_train_tensor.batch(batch_size)

        for epoch in range(n_epochs):
            for batch_no, w_minibatch in enumerate(w_train_minibatches):
                with tf.GradientTape() as tape:
                    alpha_rho_minibatch = np.array([])
                    alpha_rho_passed_minibatch = np.array([])
                    for w_sample in w_minibatch:
                        w_in = construct_nn_input(w_sample.numpy())
                        alpha_rho = model(w_in, training=True)
                        alpha_rho_minibatch = np.append(
                            alpha_rho_minibatch,
                            alpha_rho)
                        alpha_rho_passed = forward_pass(w_sample, alpha_rho)
                        alpha_rho_passed_minibatch = np.append(
                            alpha_rho_passed_minibatch,
                            alpha_rho_passed)
                    
                    alpha_rho_minibatch = np.reshape(
                        alpha_rho_minibatch,
                        (batch_size, (N_NODE**2)*2))
                    alpha_rho_passed_minibatch = np.reshape(
                        alpha_rho_passed_minibatch,
                        (batch_size, (N_NODE**2)*2))
                    loss_value = loss_fxn(alpha_rho, alpha_rho_passed)
                grads = tape.gradient(loss_value,
                                      model.trainable_weights)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_weights))
                print(f"Epoch {epoch}, batch#{batch_no}: loss={loss_value}")
            print("\n")
        model.save_weights(FILENAME_NN_WEIGHT)
    run_test_matching(w_test, alpha_rho_star_test, model)


def fetch_dataset():
    bAvailable = check_dataset_availability()
    if not bAvailable:
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
    w = generate_dataset_input_easy(unif_ub=1.0)
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


def generate_dataset_input_easy(unif_ub):
    w = np.zeros((N_DATASET, N_NODE**2))
    for row in range(N_DATASET):
        w[row, :] = np.random.uniform(0, unif_ub, N_NODE**2)
        idx_taken = np.array([], dtype=int)
        for i in range(N_NODE):
            idx_picked = np.random.randint(0, N_NODE)
            while idx_picked in idx_taken:
                idx_picked = np.random.randint(0, N_NODE)
            w[row, N_NODE*i + idx_picked] = 1
            idx_taken = np.append(idx_taken, idx_picked)
        if row==0:
            print(f"Generating {N_DATASET} sets"
                  " of input weights such as...")
            print(reshape_to_square(w[row,:]))
    return w


def generate_dataset_input_quantized():
    for i in range(N_DATASET):
        rng = np.random.default_rng(SEED_W+i)
        w_instance = rng.integers(0, 10, (1, N_NODE**2))
        if i==0:
            w = w_instance/10
        else:
            w = np.append(w, w_instance/10, axis=0)
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


def print_dataset_dimensions(arr_list):
    for arr in arr_list:
        print(f"Shape: {np.shape(arr)}")


def initialize_model():
    inputs = tf.keras.layers.Input(shape=(N_NODE**2,))
    x1 = tf.keras.layers.Dense(50, activation="relu")(inputs)
    x2 = tf.keras.layers.Dense(100, activation="relu")(x1)
    x3 = tf.keras.layers.Dense(50, activation="relu")(x2)
    outputs = tf.keras.layers.Dense(50, name="predictions")(x3)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def forward_pass(w, alpha_rho):
    alpha, rho = decompose_dataset(alpha_rho[0], 'output')
    alpha_next = update_alpha(alpha, rho,
                              reshape_to_square(w),
                              bLogSumExp=False)
    rho_next = update_rho(alpha, rho,
                          reshape_to_square(w),
                          bLogSumExp=False)
    alpha_next = reshape_to_flat(alpha_next)
    rho_next = reshape_to_flat(rho_next)
    alpha_rho_passed = np.concatenate((alpha_next, rho_next))
    return alpha_rho_passed


def loss(model, w):
    nData = np.size(w, axis=0)
    l1_errors = np.zeros(nData)
    for i in range(nData):
        alpha_rho = model(w[i])
        alpha, rho = decompose_dataset(alpha_rho, 'output')
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
            np.abs(alpha_rho - alpha_rho_next))
        
    return np.mean(l1_errors)


def remove_invalid_samples(arr, idx_invalid):
    return np.delete(arr, idx_invalid, axis=0)


def construct_nn_input(w):
    w = np.array([w])
    return w


def run_test_matching(w, alpha_rho_star, model):
    n_samples = np.size(w, axis=0)
    D_mp, D_mp_validity = get_D_mp(
        n_samples, alpha_rho_star)
    # idx_invalid_samples = np.where(D_mp_validity==False)[0]
    # D_mp = remove_invalid_samples(D_mp, idx_invalid_samples)
    print("MP performance:")
    print_assessment(D_mp_validity, n_samples)

    D_nn, D_nn_validity = get_D_nn(n_samples, w, model)
    print("NN performance:")

    print_assessment(D_nn_validity, n_samples)
    
    acc = np.array([])
    for d_nn, d_mp in zip(D_nn, D_mp):
        acc = np.append(acc, np.all(d_nn == d_mp))
    print(np.count_nonzero(acc)/np.size(acc)*100, "% acc")


    # idx_valid_samples = np.where(D_nn_validity==True)[0]
    # print("valid idices: ", idx_valid_samples)
    # tmp_idx = idx_valid_samples[20]
    # w_tmp = construct_nn_input(w[tmp_idx])
    # alpha_rho_tmp = model(w_tmp)
    # alpha_tmp, rho_tmp = decompose_dataset(alpha_rho_tmp[0], 'output')
    # print(f"alpha: \n{alpha_tmp}")
    # print(f"rho: \n{rho_tmp}")
    # print(f"alpha+rho: \n{alpha_tmp+rho_tmp}")
    # print(reshape_to_square(D_nn[tmp_idx]))
    # print(reshape_to_square(D_mp[tmp_idx]))



def get_D_mp(n_samples, alpha_rho_star):
    D_mp = np.zeros((n_samples, N_NODE**2), dtype=int)
    D_mp_validity = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        alpha_star, rho_star = decompose_dataset(alpha_rho_star[i], 'output')
        D_pred = get_pairing_matrix(alpha_star, rho_star)
        D_mp[i] = reshape_to_flat(D_pred)
        D_mp_validity[i] = check_validity(D_pred)
    return D_mp, D_mp_validity


def get_D_nn(n_samples, w, model):
    D_nn = np.zeros((n_samples, N_NODE**2), dtype=int)
    D_nn_validity = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        w_sample = construct_nn_input(w[i])
        alpha_rho = model(w_sample).numpy()[0]
        alpha_sample, rho_sample = decompose_dataset(
            alpha_rho, 'output')
        D_pred = get_pairing_matrix(alpha_sample, rho_sample)
        D_nn[i] = reshape_to_flat(D_pred)
        D_nn_validity[i] = check_validity(D_pred)
    return D_nn, D_nn_validity


def print_assessment(D_validity, n_samples):
    nValid = np.count_nonzero(D_validity)
    print(f"{nValid} out of {n_samples}",
          f"({nValid/n_samples*100} %)")


if __name__=="__main__":
    tic = time.time()
    main()
    toc = time.time()
    print(f"Runtime: {toc-tic}sec.")
