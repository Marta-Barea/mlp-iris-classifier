import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def build_model(input_dim: int, units: int = 8, learning_rate: float = 0.001):
    model = Sequential()
    model.add(Dense(units, input_dim=input_dim, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model
