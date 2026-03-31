"""Entry point per il server FL in ambiente containerizzato.

Avvia il server Flower SuperLink con la configurazione
ricevuta dalle variabili d'ambiente.

Uso (Docker/k3s):
    FL_DATASET=cifar10 FL_NUM_ROUNDS=150 python -m src.federation.server_app
"""

import os

import flwr as fl

from src.utils.config import DATASET_INFO


def get_env(key, default=None, cast=None):
    """Legge una variabile d'ambiente con cast opzionale."""
    value = os.environ.get(key, default)
    if value is not None and cast is not None:
        return cast(value)
    return value


def main():
    server_address = get_env("FL_SERVER_ADDRESS", "0.0.0.0:8080")
    num_rounds = get_env("FL_NUM_ROUNDS", 150, int)
    min_clients = get_env("FL_MIN_CLIENTS", 2, int)
    dataset = get_env("FL_DATASET", "cifar10")

    info = DATASET_INFO[dataset]

    print(f"DHFLPL2 - FL Server (SuperLink)")
    print(f"  Address:     {server_address}")
    print(f"  Dataset:     {dataset}")
    print(f"  Num rounds:  {num_rounds}")
    print(f"  Min clients: {min_clients}")
    print(f"  Input shape: {info['input_shape']}")
    print(f"  Num classes: {info['num_classes']}")

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
