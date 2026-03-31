"""Distribuzione non-IID dei dati tra client federati.

Implementa la logica di partizionamento descritta nel paper
per distribuire i dati in modo non uniforme tra i nodi.

Riferimento: Listing 1 del paper.
"""

import numpy as np


def split_data_for_federated_learning(data, num_clients):
    """Divide i dati tra i client in modo equo (non-IID tramite shuffle).

    I dati vengono prima mescolati casualmente per rompere
    l'ordinamento per classe, creando una distribuzione non-IID
    naturale quando suddivisi tra i client.

    Args:
        data: Tuple (x, y) con features e labels.
        num_clients: Numero di client federati.

    Returns:
        Lista di tuple [(x_i, y_i), ...] per ogni client.

    Riferimento: Listing 1 del paper.
    """
    x, y = data
    indices = np.random.permutation(len(x))
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    data_per_client = len(x) // num_clients

    datasets = [
        (
            x_shuffled[i * data_per_client:(i + 1) * data_per_client],
            y_shuffled[i * data_per_client:(i + 1) * data_per_client],
        )
        for i in range(num_clients)
    ]

    return datasets


def split_data_non_iid(data, num_clients, num_shards_per_client=2):
    """Distribuzione non-IID avanzata basata su shards per classe.

    Ordina i dati per label e li divide in shards, assegnando
    a ciascun client un sottoinsieme di shards. Questo genera
    una distribuzione fortemente non-IID dove ogni client vede
    solo un sottoinsieme delle classi.

    Args:
        data: Tuple (x, y) con features e labels.
        num_clients: Numero di client federati.
        num_shards_per_client: Shards assegnati a ogni client.

    Returns:
        Lista di tuple [(x_i, y_i), ...] per ogni client.
    """
    x, y = data
    num_shards = num_clients * num_shards_per_client
    shard_size = len(x) // num_shards

    sorted_indices = np.argsort(y)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    shards = [
        (
            x_sorted[i * shard_size:(i + 1) * shard_size],
            y_sorted[i * shard_size:(i + 1) * shard_size],
        )
        for i in range(num_shards)
    ]

    shard_indices = np.random.permutation(num_shards)

    datasets = []
    for i in range(num_clients):
        client_shards = shard_indices[
            i * num_shards_per_client:(i + 1) * num_shards_per_client
        ]
        client_x = np.concatenate([shards[s][0] for s in client_shards])
        client_y = np.concatenate([shards[s][1] for s in client_shards])
        datasets.append((client_x, client_y))

    return datasets
