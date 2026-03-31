"""Entry point per il client FL in ambiente containerizzato.

Avvia un Flower SuperNode che si auto-registra al SuperLink
e partecipa al training federato.

Uso (Docker/k3s):
    FL_SERVER_ADDRESS=server:8080 FL_DATASET=cifar10 python -m src.federation.client_app
"""

import os

import flwr as fl
import numpy as np

from src.models.cnn import create_model
from src.data.loader import load_dataset
from src.data.partitioner import split_data_for_federated_learning
from src.privacy.dp_mechanism import apply_dp_to_weights
from src.utils.config import DATASET_INFO


def get_env(key, default=None, cast=None):
    """Legge una variabile d'ambiente con cast opzionale."""
    value = os.environ.get(key, default)
    if value is not None and cast is not None:
        return cast(value)
    return value


class DHFLClient(fl.client.NumPyClient):
    """Client Flower per DHFLPL2.

    Implementa l'interfaccia NumPyClient di Flower per
    integrarsi con il framework SuperLink/SuperNode.
    """

    def __init__(self, model, x_train, y_train, x_test, y_test,
                 local_epochs, batch_size, dp_enabled, dp_epsilon,
                 dp_delta):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

        weights = self.model.get_weights()

        if self.dp_enabled:
            weights = apply_dp_to_weights(
                weights, self.dp_epsilon, self.dp_delta,
            )

        return weights, len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(
            self.x_test, self.y_test, verbose=0,
        )
        return loss, len(self.x_test), {"accuracy": accuracy}


def main():
    server_address = get_env("FL_SERVER_ADDRESS", "server:8080")
    client_id = get_env("FL_CLIENT_ID", "0")
    dataset = get_env("FL_DATASET", "cifar10")
    local_epochs = get_env("FL_LOCAL_EPOCHS", 1, int)
    batch_size = get_env("FL_BATCH_SIZE", 32, int)
    dp_enabled = get_env("FL_DP_ENABLED", "false").lower() == "true"
    dp_epsilon = get_env("FL_DP_EPSILON", 1.0, float)
    dp_delta = get_env("FL_DP_DELTA", 1e-5, float)

    info = DATASET_INFO[dataset]

    print(f"DHFLPL2 - FL Client (SuperNode)")
    print(f"  Client ID:   {client_id}")
    print(f"  Server:      {server_address}")
    print(f"  Dataset:     {dataset}")
    print(f"  DP enabled:  {dp_enabled}")

    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)

    # Simula la partizione: in produzione ogni nodo riceve
    # il proprio sottoinsieme di dati
    np.random.seed(hash(client_id) % 2**32)
    indices = np.random.permutation(len(x_train))
    partition_size = len(x_train) // 10
    x_train = x_train[indices[:partition_size]]
    y_train = y_train[indices[:partition_size]]

    model = create_model(info["input_shape"], info["num_classes"])

    client = DHFLClient(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        local_epochs=local_epochs,
        batch_size=batch_size,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_delta=dp_delta,
    )

    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
