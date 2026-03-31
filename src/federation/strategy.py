"""Strategie di aggregazione per Federated Learning.

Implementa FedAvg (Federated Averaging) come descritto nel paper:
w_{t+1} = sum_{k=1}^{K} (n_k / n) * w_{t+1}^k

Riferimento: Sezione 4.1, Algorithm 1 del paper.
"""

import numpy as np


def federated_averaging(client_weights, client_sizes):
    """Aggregazione FedAvg pesata sul numero di campioni per client.

    Args:
        client_weights: Lista di pesi dei modelli locali.
            Ogni elemento e' una lista di array numpy (layer weights).
        client_sizes: Lista con il numero di campioni per client.

    Returns:
        Lista di array numpy con i pesi aggregati del modello globale.
    """
    total_samples = sum(client_sizes)
    num_layers = len(client_weights[0])

    averaged_weights = []
    for layer_idx in range(num_layers):
        weighted_sum = np.zeros_like(client_weights[0][layer_idx])
        for client_idx, weights in enumerate(client_weights):
            weight_factor = client_sizes[client_idx] / total_samples
            weighted_sum += weights[layer_idx] * weight_factor
        averaged_weights.append(weighted_sum)

    return averaged_weights


def simple_averaging(client_weights):
    """Aggregazione con media semplice (non pesata).

    Variante semplificata usata quando tutti i client hanno
    lo stesso numero di campioni.

    Args:
        client_weights: Lista di pesi dei modelli locali.

    Returns:
        Lista di array numpy con i pesi mediati.
    """
    num_clients = len(client_weights)
    num_layers = len(client_weights[0])

    averaged_weights = []
    for layer_idx in range(num_layers):
        layer_sum = np.zeros_like(client_weights[0][layer_idx])
        for weights in client_weights:
            layer_sum += weights[layer_idx]
        averaged_weights.append(layer_sum / num_clients)

    return averaged_weights
