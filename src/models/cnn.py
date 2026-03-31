"""Definizione del modello CNN per classificazione immagini.

Modello adattabile ai 5 dataset del paper (CIFAR-10, CIFAR-100,
MNIST, Fashion-MNIST, SVHN) con architettura comune e output
variabile in base al numero di classi.

Riferimento: Sezione 5.2 del paper, Listing 3.
"""

import tensorflow as tf
from tensorflow import keras


def create_model(input_shape, num_classes):
    """Crea un modello CNN per classificazione immagini.

    Args:
        input_shape: Tuple (height, width, channels) dell'input.
        num_classes: Numero di classi di output.

    Returns:
        Modello Keras compilato.
    """
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation="relu",
                            input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation="relu"),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
