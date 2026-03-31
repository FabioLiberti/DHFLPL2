"""Demo: Confronto Federated Learning con e senza Differential Privacy.

Esegue lo stesso esperimento due volte:
1. FedAvg standard (senza DP)
2. FedAvg con Differential Privacy (epsilon=1.0, delta=1e-5)

Mostra il trade-off privacy/performance: la DP protegge i dati
ma riduce l'accuracy del modello.

Riferimento: Sezione 4.2 e 5.2 del paper, Equazione (4).

Uso:
    python -m demos.demo_dp_comparison
    python -m demos.demo_dp_comparison --dataset cifar10 --clients 5
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.cnn import create_model
from src.data.loader import load_dataset
from src.data.partitioner import split_data_for_federated_learning
from src.federation.client import FLClient
from src.federation.strategy import federated_averaging
from src.metrics.evaluation import evaluate_with_metrics
from src.privacy.dp_mechanism import apply_dp_to_weights


def run_fedavg(input_shape, num_classes, federated_datasets,
               test_data, num_rounds, dp_config=None):
    """Esegue FedAvg con o senza DP e restituisce lo storico."""
    x_test, y_test = test_data
    model = create_model(input_shape, num_classes)
    history = {"accuracy": [], "loss": [], "precision": [], "f1": []}

    for round_num in range(1, num_rounds + 1):
        global_weights = model.get_weights()
        client_weights = []
        client_sizes = []

        for i, (x_c, y_c) in enumerate(federated_datasets):
            client = FLClient(i, x_c, y_c, input_shape, num_classes)
            client.set_weights(global_weights)
            client.train(epochs=1, verbose=0)

            weights = client.get_weights()
            if dp_config is not None:
                weights = apply_dp_to_weights(
                    weights,
                    epsilon=dp_config["epsilon"],
                    delta=dp_config["delta"],
                    clip_norm=dp_config.get("clip_norm", 1.0),
                )
            client_weights.append(weights)
            client_sizes.append(len(x_c))

        new_weights = federated_averaging(client_weights, client_sizes)
        model.set_weights(new_weights)

        metrics = evaluate_with_metrics(model, x_test, y_test)
        history["accuracy"].append(metrics["accuracy"])
        history["loss"].append(metrics["loss"])
        history["precision"].append(metrics["precision"])
        history["f1"].append(metrics["f1"])

        label = "[DP]" if dp_config else "[Standard]"
        print(
            f"  Round {round_num:3d}/{num_rounds} {label} - "
            f"acc: {metrics['accuracy']:.4f} - "
            f"loss: {metrics['loss']:.4f} - "
            f"f1: {metrics['f1']:.4f}"
        )

    return history


def plot_comparison(history_std, history_dp, dataset, epsilon, output_path):
    """Genera il grafico di confronto standard vs DP."""
    rounds = range(1, len(history_std["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy
    axes[0].plot(rounds, history_std["accuracy"],
                 label="Standard FedAvg", linewidth=1.5)
    axes[0].plot(rounds, history_dp["accuracy"],
                 label=f"FedAvg + DP (eps={epsilon})", linewidth=1.5,
                 linestyle="--")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Accuracy - {dataset.upper()}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(rounds, history_std["loss"],
                 label="Standard FedAvg", linewidth=1.5)
    axes[1].plot(rounds, history_dp["loss"],
                 label=f"FedAvg + DP (eps={epsilon})", linewidth=1.5,
                 linestyle="--")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"Loss - {dataset.upper()}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(rounds, history_std["f1"],
                 label="Standard FedAvg", linewidth=1.5)
    axes[2].plot(rounds, history_dp["f1"],
                 label=f"FedAvg + DP (eps={epsilon})", linewidth=1.5,
                 linestyle="--")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title(f"F1 Score - {dataset.upper()}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        "Federated Learning: Standard vs Differential Privacy",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGrafico salvato in: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: confronto FL con/senza Differential Privacy",
    )
    parser.add_argument("--dataset", default="mnist",
                        help="Dataset (default: mnist)")
    parser.add_argument("--clients", type=int, default=2,
                        help="Numero di client (default: 2)")
    parser.add_argument("--rounds", type=int, default=30,
                        help="Numero di round (default: 30)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Budget di privacy epsilon (default: 1.0)")
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs",
    )
    os.makedirs(output_dir, exist_ok=True)

    from src.utils.config import DATASET_INFO
    info = DATASET_INFO[args.dataset]
    input_shape = info["input_shape"]
    num_classes = info["num_classes"]

    print(f"{'='*60}")
    print(f"DEMO: Confronto Standard vs Differential Privacy")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Client:   {args.clients}")
    print(f"  Round:    {args.rounds}")
    print(f"  Epsilon:  {args.epsilon}")
    print(f"{'='*60}")

    print("\nCaricamento dataset...")
    (x_train, y_train), (x_test, y_test) = load_dataset(args.dataset)

    federated_datasets = split_data_for_federated_learning(
        (x_train, y_train), args.clients,
    )

    print(f"\n--- FedAvg Standard (senza DP) ---")
    history_std = run_fedavg(
        input_shape, num_classes, federated_datasets,
        (x_test, y_test), args.rounds,
    )

    print(f"\n--- FedAvg con DP (epsilon={args.epsilon}) ---")
    history_dp = run_fedavg(
        input_shape, num_classes, federated_datasets,
        (x_test, y_test), args.rounds,
        dp_config={"epsilon": args.epsilon, "delta": 1e-5, "clip_norm": 1.0},
    )

    print(f"\n{'='*60}")
    print(f"RISULTATI FINALI")
    print(f"  Standard:  acc={history_std['accuracy'][-1]:.4f}  "
          f"f1={history_std['f1'][-1]:.4f}")
    print(f"  Con DP:    acc={history_dp['accuracy'][-1]:.4f}  "
          f"f1={history_dp['f1'][-1]:.4f}")
    diff = history_std["accuracy"][-1] - history_dp["accuracy"][-1]
    print(f"  Delta:     {diff:+.4f} accuracy")
    print(f"{'='*60}")

    plot_comparison(
        history_std, history_dp, args.dataset, args.epsilon,
        os.path.join(output_dir, f"dp_comparison_{args.dataset}.png"),
    )


if __name__ == "__main__":
    main()
