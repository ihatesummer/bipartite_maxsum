import os as os
import tensorflow as tf

N_NODE = 5

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2*(N_NODE**2))
    ])
    loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['mse'])
    return model


checkpoint_path = "NNtrained/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(os.listdir(checkpoint_dir))
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

model = create_model()
model.load_weights(latest)
