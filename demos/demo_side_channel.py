"""Demo: Side-Channel Attack Analysis.

Analizza le vulnerabilita' da side-channel nel processo federato:
- Varianza nei tempi di comunicazione tra client
- Varianza nella dimensione dei dati trasmessi
- Correlazione tra timing e dimensione del dataset locale

Mostra come un attaccante potrebbe inferire informazioni sulla
distribuzione dei dati osservando i pattern di comunicazione.

Riferimento: Sezione 4.2 del paper - Side-Channel Attacks.

Uso:
    python -m demos.demo_side_channel
    python -m demos.demo_side_channel --clients 10 --rounds 20
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.cnn import create_model
from src.data.loader import load_dataset
from src.data.partitioner import split_data_for_federated_learning
from src.federation.client import FLClient
from src.privacy.threat_model import SideChannelAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Side-Channel Attack Analysis",
    )
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"DEMO: Side-Channel Attack Analysis")
    print(f"  Client: {args.clients}  Round: {args.rounds}")
    print(f"{'='*60}")

    print("\nCaricamento MNIST...")
    (x_train, y_train), _ = load_dataset("mnist")

    input_shape = (28, 28, 1)
    num_classes = 10

    federated_datasets = split_data_for_federated_learning(
        (x_train, y_train), args.clients,
    )

    model = create_model(input_shape, num_classes)
    analyzer = SideChannelAnalyzer()

    # Raccogli metriche di comunicazione per round
    all_times = []     # [round][client]
    all_sizes = []

    for r in range(1, args.rounds + 1):
        global_weights = model.get_weights()
        round_times = []
        round_sizes = []

        for i, (x_c, y_c) in enumerate(federated_datasets):
            client = FLClient(i, x_c, y_c, input_shape, num_classes)
            client.set_weights(global_weights)

            start = time.perf_counter()
            client.train(epochs=1, verbose=0)
            weights = client.get_weights()
            elapsed = time.perf_counter() - start

            data_size = sum(w.nbytes for w in weights)

            analyzer.record_communication(i, weights, elapsed)
            round_times.append(elapsed)
            round_sizes.append(data_size)

        all_times.append(round_times)
        all_sizes.append(round_sizes)

        print(f"  Round {r:3d}/{args.rounds} - "
              f"avg time: {np.mean(round_times):.3f}s - "
              f"time variance: {np.var(round_times):.6f}")

    # Analisi
    analysis = analyzer.analyze()

    print(f"\n{'='*60}")
    print(f"ANALISI SIDE-CHANNEL")
    print(f"  Comunicazioni totali:  {analysis['num_communications']}")
    print(f"  Tempo medio:           {analysis['avg_time']:.4f}s")
    print(f"  Varianza tempo:        {analysis['time_variance']:.6f}")
    print(f"  Dimensione media:      {analysis['avg_size_bytes']/1024:.1f} KB")
    print(f"  Varianza dimensione:   {analysis['size_variance']:.0f}")
    print(f"  Livello di rischio:    {analysis['risk_level'].upper()}")
    print(f"{'='*60}")

    if analysis["risk_level"] == "low":
        print("\n  I pattern di comunicazione sono uniformi.")
        print("  Un attaccante non puo' inferire informazioni significative.")
    elif analysis["risk_level"] == "medium":
        print("\n  Esistono variazioni nei pattern di comunicazione.")
        print("  Un attaccante potrebbe inferire alcune informazioni.")
    else:
        print("\n  ATTENZIONE: forte variazione nei pattern di comunicazione.")
        print("  Un attaccante potrebbe inferire la distribuzione dei dati.")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Tempi per client
    times_per_client = np.array(all_times).T
    for i in range(args.clients):
        axes[0, 0].plot(range(1, args.rounds + 1), times_per_client[i],
                        label=f"Client {i}", linewidth=1.2)
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Tempo (s)")
    axes[0, 0].set_title("Tempo di training per client")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Distribuzione tempi
    all_flat_times = np.array(all_times).flatten()
    axes[0, 1].hist(all_flat_times, bins=20, color="#3498db", edgecolor="black")
    axes[0, 1].set_xlabel("Tempo (s)")
    axes[0, 1].set_ylabel("Frequenza")
    axes[0, 1].set_title("Distribuzione tempi di comunicazione")
    axes[0, 1].axvline(np.mean(all_flat_times), color="red",
                       linestyle="--", label=f"Media: {np.mean(all_flat_times):.3f}s")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Varianza per round
    variances = [np.var(t) for t in all_times]
    axes[1, 0].bar(range(1, args.rounds + 1), variances, color="#e74c3c")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Varianza tempo")
    axes[1, 0].set_title("Varianza temporale per round\n(alta varianza = piu' informazione per l'attaccante)")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Dimensione dati vs tempo (correlazione)
    sizes_flat = np.array(all_sizes).flatten()
    dataset_sizes = [len(ds[0]) for ds in federated_datasets] * args.rounds
    axes[1, 1].scatter(dataset_sizes, all_flat_times, alpha=0.5, s=20)
    axes[1, 1].set_xlabel("Dimensione dataset locale (campioni)")
    axes[1, 1].set_ylabel("Tempo di training (s)")
    axes[1, 1].set_title("Correlazione dataset size / timing\n(correlazione = side-channel exploitabile)")
    axes[1, 1].grid(True, alpha=0.3)

    # Calcola correlazione
    corr = np.corrcoef(dataset_sizes, all_flat_times)[0, 1]
    axes[1, 1].text(0.05, 0.95, f"Correlazione: {corr:.3f}",
                    transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle(
        f"Side-Channel Analysis ({args.clients} client, {args.rounds} round) "
        f"- Rischio: {analysis['risk_level'].upper()}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    output_path = os.path.join(output_dir, "side_channel_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGrafico salvato in: {output_path}")


if __name__ == "__main__":
    main()
