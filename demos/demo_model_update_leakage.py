"""Demo: Model Update Leakage Analysis.

Monitora come i pesi del modello cambiano nel tempo durante il
training federato, analizzando il rischio di information leakage.

Mostra:
1. Magnitudine degli aggiornamenti per round
2. Distribuzione dei delta nei pesi
3. Confronto del rischio con/senza DP

Riferimento: Sezione 4.2 del paper - Model Updates Leakage, Equazione (3).

Uso:
    python -m demos.demo_model_update_leakage
    python -m demos.demo_model_update_leakage --rounds 30 --epsilon 0.5
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
from src.privacy.dp_mechanism import apply_dp_to_weights
from src.privacy.threat_model import ModelUpdateLeakage


def run_and_track(input_shape, num_classes, federated_datasets,
                  num_rounds, dp_config=None):
    """Esegue FL tracciando ogni aggiornamento dei pesi."""
    model = create_model(input_shape, num_classes)
    tracker = ModelUpdateLeakage()
    magnitudes = []
    all_deltas = []

    for r in range(1, num_rounds + 1):
        weights_before = model.get_weights()
        global_weights = model.get_weights()
        client_weights = []
        client_sizes = []

        for i, (x_c, y_c) in enumerate(federated_datasets):
            client = FLClient(i, x_c, y_c, input_shape, num_classes)
            client.set_weights(global_weights)
            client.train(epochs=1, verbose=0)
            weights = client.get_weights()
            if dp_config:
                weights = apply_dp_to_weights(
                    weights, dp_config["epsilon"],
                    dp_config["delta"], dp_config.get("clip_norm", 1.0),
                )
            client_weights.append(weights)
            client_sizes.append(len(x_c))

        new_weights = federated_averaging(client_weights, client_sizes)
        model.set_weights(new_weights)

        tracker.record_update(weights_before, new_weights, r)

        deltas = [
            after - before
            for before, after in zip(weights_before, new_weights)
        ]
        magnitude = float(np.sqrt(sum(np.sum(d ** 2) for d in deltas)))
        magnitudes.append(magnitude)

        flat_deltas = np.concatenate([d.flatten() for d in deltas])
        all_deltas.append(flat_deltas)

        label = "[DP]" if dp_config else "[Std]"
        if r % 5 == 0 or r == 1:
            print(f"  Round {r:3d}/{num_rounds} {label} - "
                  f"update magnitude: {magnitude:.4f}")

    risk = tracker.analyze_leakage_risk()
    return magnitudes, all_deltas, risk


def plot_results(mag_std, mag_dp, deltas_std, deltas_dp,
                 risk_std, risk_dp, epsilon, output_path):
    """Genera i grafici di analisi del leakage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rounds = range(1, len(mag_std) + 1)

    # Magnitudine aggiornamenti
    axes[0, 0].plot(rounds, mag_std, label="Senza DP",
                    linewidth=1.5, color="#e74c3c")
    axes[0, 0].plot(rounds, mag_dp, label=f"Con DP (eps={epsilon})",
                    linewidth=1.5, color="#2ecc71", linestyle="--")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Magnitudine Update")
    axes[0, 0].set_title("Magnitudine aggiornamenti pesi per round")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Distribuzione delta (primo round)
    axes[0, 1].hist(deltas_std[0], bins=50, alpha=0.7,
                     label="Senza DP", color="#e74c3c", density=True)
    axes[0, 1].hist(deltas_dp[0], bins=50, alpha=0.7,
                     label=f"Con DP (eps={epsilon})", color="#2ecc71", density=True)
    axes[0, 1].set_xlabel("Delta pesi")
    axes[0, 1].set_ylabel("Densita'")
    axes[0, 1].set_title("Distribuzione delta pesi (Round 1)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Distribuzione delta (ultimo round)
    axes[1, 0].hist(deltas_std[-1], bins=50, alpha=0.7,
                     label="Senza DP", color="#e74c3c", density=True)
    axes[1, 0].hist(deltas_dp[-1], bins=50, alpha=0.7,
                     label=f"Con DP (eps={epsilon})", color="#2ecc71", density=True)
    axes[1, 0].set_xlabel("Delta pesi")
    axes[1, 0].set_ylabel("Densita'")
    axes[1, 0].set_title(f"Distribuzione delta pesi (Round {len(deltas_std)})")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Confronto rischio
    risk_labels = ["Senza DP", f"Con DP\n(eps={epsilon})"]
    risk_values = [risk_std["avg_magnitude"], risk_dp["avg_magnitude"]]
    risk_colors = [
        "#e74c3c" if risk_std["risk_level"] == "high"
        else "#f39c12" if risk_std["risk_level"] == "medium"
        else "#2ecc71",
        "#e74c3c" if risk_dp["risk_level"] == "high"
        else "#f39c12" if risk_dp["risk_level"] == "medium"
        else "#2ecc71",
    ]
    bars = axes[1, 1].bar(risk_labels, risk_values, color=risk_colors)
    axes[1, 1].set_ylabel("Magnitudine Media Update")
    axes[1, 1].set_title("Livello di Rischio Leakage")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    for bar, risk in zip(bars, [risk_std, risk_dp]):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            risk["risk_level"].upper(),
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    plt.suptitle(
        "Model Update Leakage Analysis",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGrafico salvato in: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Model Update Leakage Analysis",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"DEMO: Model Update Leakage Analysis")
    print(f"  Round: {args.rounds}  Client: {args.clients}")
    print(f"  Epsilon: {args.epsilon}")
    print(f"{'='*60}")

    print("\nCaricamento MNIST...")
    (x_train, y_train), _ = load_dataset("mnist")

    input_shape = (28, 28, 1)
    num_classes = 10
    federated_datasets = split_data_for_federated_learning(
        (x_train, y_train), args.clients,
    )

    print(f"\n--- Training SENZA DP ---")
    mag_std, deltas_std, risk_std = run_and_track(
        input_shape, num_classes, federated_datasets, args.rounds,
    )

    print(f"\n--- Training CON DP (epsilon={args.epsilon}) ---")
    mag_dp, deltas_dp, risk_dp = run_and_track(
        input_shape, num_classes, federated_datasets, args.rounds,
        dp_config={"epsilon": args.epsilon, "delta": 1e-5},
    )

    print(f"\n{'='*60}")
    print(f"RISULTATI")
    print(f"  Senza DP - Rischio: {risk_std['risk_level'].upper()} "
          f"(avg magnitude: {risk_std['avg_magnitude']:.4f})")
    print(f"  Con DP   - Rischio: {risk_dp['risk_level'].upper()} "
          f"(avg magnitude: {risk_dp['avg_magnitude']:.4f})")
    print(f"{'='*60}")

    plot_results(
        mag_std, mag_dp, deltas_std, deltas_dp,
        risk_std, risk_dp, args.epsilon,
        os.path.join(output_dir, "model_update_leakage.png"),
    )


if __name__ == "__main__":
    main()
