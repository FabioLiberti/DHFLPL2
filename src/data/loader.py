"""Caricamento e preprocessing dei dataset.

Supporta i 5 dataset utilizzati nel paper:
CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, SVHN.

Riferimento: Sezione "Results" del paper.
"""

import numpy as np
import tensorflow as tf

from src.utils.config import SUPPORTED_DATASETS


def load_dataset(name):
    """Carica un dataset e restituisce dati normalizzati.

    Args:
        name: Nome del dataset (cifar10, cifar100, mnist,
              fashion_mnist, svhn).

    Returns:
        Tuple ((x_train, y_train), (x_test, y_test)) con valori
        normalizzati in [0, 1].

    Raises:
        ValueError: Se il dataset non e' supportato.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{name}' non supportato. "
            f"Usa uno tra: {SUPPORTED_DATASETS}"
        )

    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.cifar10.load_data()
        )
    elif name == "cifar100":
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.cifar100.load_data()
        )
    elif name == "mnist":
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.mnist.load_data()
        )
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
    elif name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
    elif name == "svhn":
        (x_train, y_train), (x_test, y_test) = _load_svhn()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return (x_train, y_train), (x_test, y_test)


def _load_svhn():
    """Carica il dataset SVHN da tensorflow_datasets o da URL.

    Returns:
        Tuple ((x_train, y_train), (x_test, y_test)).
    """
    try:
        import tensorflow_datasets as tfds
        train_ds = tfds.load("svhn_cropped", split="train", as_supervised=True)
        test_ds = tfds.load("svhn_cropped", split="test", as_supervised=True)

        x_train, y_train = [], []
        for img, label in train_ds:
            x_train.append(img.numpy())
            y_train.append(label.numpy())

        x_test, y_test = [], []
        for img, label in test_ds:
            x_test.append(img.numpy())
            y_test.append(label.numpy())

        return (
            (np.array(x_train), np.array(y_train)),
            (np.array(x_test), np.array(y_test)),
        )
    except ImportError:
        raise ImportError(
            "Per il dataset SVHN installa tensorflow-datasets: "
            "pip install tensorflow-datasets"
        )
