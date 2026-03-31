"""Demo: Membership Inference Attack.

Simula un attaccante che tenta di determinare se un dato campione
faceva parte del training set, analizzando i confidence score del modello.

Mostra:
1. Distribuzione confidence per campioni "member" (nel training set)
2. Distribuzione confidence per campioni "non-member" (fuori dal training set)
3. Come la DP riduce la separabilita' tra le due distribuzioni

Riferimento: Sezione 4.2 del paper - Membership Inference Attacks.

Uso:
    python -m demos.demo_membership_inference
    python -m demos.demo_membership_inference --rounds 20 --epsilon 0.5
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


def train_federated_model(input_shape, num_classes, train_data,
                          num_clients, num_rounds, dp_config=None):
    """Addestra un modello federato e lo restituisce."""
    federated_datasets = split_data_for_federated_learning(
        train_data, num_clients,
    )
    model = create_model(input_shape, num_classes)

    for r in range(1, num_rounds + 1):
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

        label = "[DP]" if dp_config else "[Std]"
        if r % 5 == 0 or r == 1:
            print(f"  Round {r:3d}/{num_rounds} {label}")

    return model


def get_confidence_scores(model, x_data, y_data):
    """Calcola il confidence score per la classe corretta."""
    predictions = model.predict(x_data, verbose=0)
    confidences = np.array([
        predictions[i, int(y_data[i])]
        for i in range(len(y_data))
    ])
    return confidences


def plot_results(conf_member_std, conf_nonmember_std,
                 conf_member_dp, conf_nonmember_dp,
                 epsilon, output_path):
    """Genera il pannello di confronto membership inference."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Senza DP
    axes[0, 0].hist(conf_member_std, bins=30, alpha=0.7,
                     label="Member (training)", color="#e74c3c")
    axes[0, 0].hist(conf_nonmember_std, bins=30, alpha=0.7,
                     label="Non-member (test)", color="#3498db")
    axes[0, 0].set_xlabel("Confidence Score")
    axes[0, 0].set_ylabel("Frequenza")
    axes[0, 0].set_title("SENZA Differential Privacy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Con DP
    axes[0, 1].hist(conf_member_dp, bins=30, alpha=0.7,
                     label="Member (training)", color="#e74c3c")
    axes[0, 1].hist(conf_nonmember_dp, bins=30, alpha=0.7,
                     label="Non-member (test)", color="#3498db")
    axes[0, 1].set_xlabel("Confidence Score")
    axes[0, 1].set_ylabel("Frequenza")
    axes[0, 1].set_title(f"CON Differential Privacy (eps={epsilon})")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Confronto medie
    means_std = [np.mean(conf_member_std), np.mean(conf_nonmember_std)]
    means_dp = [np.mean(conf_member_dp), np.mean(conf_nonmember_dp)]
    x_pos = np.arange(2)
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, means_std, width,
                    label="Senza DP", color=["#e74c3c", "#3498db"])
    axes[1, 0].bar(x_pos + width/2, means_dp, width,
                    label=f"Con DP (eps={epsilon})",
                    color=["#c0392b", "#2980b9"], alpha=0.7)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(["Member", "Non-member"])
    axes[1, 0].set_ylabel("Confidence Media")
    axes[1, 0].set_title("Confronto Confidence Media")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Attacco: accuracy dell'attaccante
    threshold = 0.5
    tp_std = np.sum(conf_member_std > threshold)
    tn_std = np.sum(conf_nonmember_std <= threshold)
    attack_acc_std = (tp_std + tn_std) / (len(conf_member_std) + len(conf_nonmember_std))

    tp_dp = np.sum(conf_member_dp > threshold)
    tn_dp = np.sum(conf_nonmember_dp <= threshold)
    attack_acc_dp = (tp_dp + tn_dp) / (len(conf_member_dp) + len(conf_nonmember_dp))

    labels = ["Senza DP", f"Con DP\n(eps={epsilon})", "Random\n(50%)"]
    accs = [attack_acc_std, attack_acc_dp, 0.5]
    colors = ["#e74c3c", "#2ecc71", "#95a5a6"]
    axes[1, 1].bar(labels, accs, color=colors)
    axes[1, 1].set_ylabel("Accuracy Attaccante")
    axes[1, 1].set_title("Successo Membership Inference Attack")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Membership Inference Attack: effetto della Differential Privacy",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGrafico salvato in: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Membership Inference Attack con/senza DP",
    )
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=500,
                        help="Campioni per valutazione membership")
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"DEMO: Membership Inference Attack")
    print(f"  Round: {args.rounds}  Client: {args.clients}")
    print(f"  Epsilon: {args.epsilon}  Campioni: {args.samples}")
    print(f"{'='*60}")

    print("\nCaricamento MNIST...")
    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")

    input_shape = (28, 28, 1)
    num_classes = 10

    # Campioni member (dal training) e non-member (dal test)
    member_idx = np.random.choice(len(x_train), args.samples, replace=False)
    nonmember_idx = np.random.choice(len(x_test), args.samples, replace=False)
    x_member = x_train[member_idx]
    y_member = y_train[member_idx]
    x_nonmember = x_test[nonmember_idx]
    y_nonmember = y_test[nonmember_idx]

    # Modello senza DP
    print(f"\n--- Training modello SENZA DP ({args.rounds} round) ---")
    model_std = train_federated_model(
        input_shape, num_classes, (x_train, y_train),
        args.clients, args.rounds,
    )

    # Modello con DP
    print(f"\n--- Training modello CON DP ({args.rounds} round) ---")
    model_dp = train_federated_model(
        input_shape, num_classes, (x_train, y_train),
        args.clients, args.rounds,
        dp_config={"epsilon": args.epsilon, "delta": 1e-5},
    )

    # Confidence scores
    print("\nCalcolo confidence scores...")
    conf_member_std = get_confidence_scores(model_std, x_member, y_member)
    conf_nonmember_std = get_confidence_scores(model_std, x_nonmember, y_nonmember)
    conf_member_dp = get_confidence_scores(model_dp, x_member, y_member)
    conf_nonmember_dp = get_confidence_scores(model_dp, x_nonmember, y_nonmember)

    # Risultati
    gap_std = np.mean(conf_member_std) - np.mean(conf_nonmember_std)
    gap_dp = np.mean(conf_member_dp) - np.mean(conf_nonmember_dp)

    print(f"\n{'='*60}")
    print(f"RISULTATI")
    print(f"  Senza DP - Gap confidence member/non-member: {gap_std:.4f}")
    print(f"  Con DP   - Gap confidence member/non-member: {gap_dp:.4f}")
    print(f"  La DP riduce il gap del {(1-abs(gap_dp)/(abs(gap_std)+1e-8))*100:.1f}%")
    print(f"{'='*60}")

    plot_results(
        conf_member_std, conf_nonmember_std,
        conf_member_dp, conf_nonmember_dp,
        args.epsilon,
        os.path.join(output_dir, "membership_inference_attack.png"),
    )


if __name__ == "__main__":
    main()
