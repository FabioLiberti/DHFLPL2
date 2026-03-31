"""Configurazione centralizzata per il framework DHFLPL2."""

SUPPORTED_DATASETS = [
    "cifar10",
    "cifar100",
    "mnist",
    "fashion_mnist",
    "svhn",
]

DATASET_INFO = {
    "cifar10": {"num_classes": 10, "input_shape": (32, 32, 3)},
    "cifar100": {"num_classes": 100, "input_shape": (32, 32, 3)},
    "mnist": {"num_classes": 10, "input_shape": (28, 28, 1)},
    "fashion_mnist": {"num_classes": 10, "input_shape": (28, 28, 1)},
    "svhn": {"num_classes": 10, "input_shape": (32, 32, 3)},
}

DEFAULT_FL_CONFIG = {
    "num_rounds": 150,
    "local_epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.001,
    "min_precision_threshold": 0.50,
}
