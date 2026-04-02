"""Entry point per eseguire gli esperimenti del paper.

Riproduce i risultati delle Figure 2 e 3:
- 5 dataset (CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, SVHN)
- Da 2 a 50 client federati
- 150 round di FedAvg per configurazione

Due modalita' di esecuzione:
- simulation (default): esegue tutto in-process usando FLServer
- flower: avvia server e client Flower per deployment distribuito

Uso:
    python -m experiments.run_experiment --config experiments/configs/cifar10.yml
    python -m experiments.run_experiment --config experiments/configs/mnist.yml --clients 2
    python -m experiments.run_experiment --config experiments/configs/cifar10_dp.yml
    python -m experiments.run_experiment --all
    python -m experiments.run_experiment --config experiments/configs/cifar10.yml --mode flower
"""

import argparse
import json
import os
import time

import yaml

from src.data.loader import load_dataset
from src.data.partitioner import split_data_for_federated_learning
from src.federation.server import FLServer
from src.privacy.dp_mechanism import apply_dp_to_weights


def load_config(config_path):
    """Carica la configurazione da file YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_single_experiment(config, num_clients, output_dir):
    """Esegue un singolo esperimento con una configurazione specifica.

    Args:
        config: Dict con la configurazione.
        num_clients: Numero di client federati.
        output_dir: Directory per salvare i risultati.

    Returns:
        Dict con i risultati dell'esperimento.
    """
    dataset_name = config["dataset"]
    input_shape = tuple(config["input_shape"])
    num_classes = config["num_classes"]
    fl_config = config["federation"]
    dp_config = config.get("privacy", {})

    print(f"\n{'='*60}")
    print(f"Esperimento: {dataset_name} | Client: {num_clients}")
    print(f"Round: {fl_config['num_rounds']} | Epoche locali: {fl_config['local_epochs']}")
    print(f"{'='*60}")

    print(f"Caricamento dataset {dataset_name}...")
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)

    print(f"Partizionamento dati per {num_clients} client (non-IID)...")
    federated_datasets = split_data_for_federated_learning(
        (x_train, y_train), num_clients,
    )

    server = FLServer(input_shape, num_classes)
    server.register_clients(federated_datasets)

    start_time = time.time()

    if dp_config.get("enabled", False):
        history = _run_with_dp(
            server, (x_test, y_test), fl_config, dp_config,
        )
    else:
        history = server.run(
            num_rounds=fl_config["num_rounds"],
            test_data=(x_test, y_test),
            local_epochs=fl_config["local_epochs"],
            batch_size=fl_config["batch_size"],
            verbose=1,
        )

    elapsed = time.time() - start_time

    result = {
        "dataset": dataset_name,
        "num_clients": num_clients,
        "num_rounds": fl_config["num_rounds"],
        "local_epochs": fl_config["local_epochs"],
        "dp_enabled": dp_config.get("enabled", False),
        "elapsed_seconds": round(elapsed, 2),
        "final_accuracy": history["accuracy"][-1],
        "final_loss": history["loss"][-1],
        "final_precision": history["precision"][-1],
        "final_recall": history["recall"][-1],
        "final_f1": history["f1"][-1],
        "history": history,
    }

    dp_suffix = "_dp" if dp_config.get("enabled", False) else ""
    result_path = os.path.join(
        output_dir, f"{dataset_name}{dp_suffix}_{num_clients}clients.json",
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nRisultato finale: acc={result['final_accuracy']:.4f} "
          f"loss={result['final_loss']:.4f} "
          f"f1={result['final_f1']:.4f}")
    print(f"Tempo: {elapsed:.1f}s | Salvato in: {result_path}")

    return result


def _run_with_dp(server, test_data, fl_config, dp_config):
    """Esegue il training federato con Differential Privacy.

    Applica DP ai pesi dei client prima dell'aggregazione.
    """
    x_test, y_test = test_data
    epsilon = dp_config["epsilon"]
    delta = dp_config["delta"]
    clip_norm = dp_config.get("clip_norm", 1.0)

    from tqdm import tqdm
    from src.federation.strategy import federated_averaging
    from src.metrics.evaluation import evaluate_with_metrics

    num_rounds = fl_config["num_rounds"]
    pbar = tqdm(
        range(1, num_rounds + 1),
        desc="FedAvg+DP",
        unit="round",
        bar_format="{l_bar}{bar:30}{r_bar}",
    )

    for round_num in pbar:
        global_weights = server.global_model.get_weights()

        client_weights = []
        client_sizes = []

        for client in server.clients:
            client.set_weights(global_weights)
            client.train(
                epochs=fl_config["local_epochs"],
                batch_size=fl_config["batch_size"],
                verbose=0,
            )
            raw_weights = client.get_weights()
            dp_weights = apply_dp_to_weights(
                raw_weights, epsilon, delta, clip_norm,
            )
            client_weights.append(dp_weights)
            client_sizes.append(client.num_samples)

        new_weights = federated_averaging(client_weights, client_sizes)
        server.global_model.set_weights(new_weights)

        loss, acc = server.global_model.evaluate(x_test, y_test, verbose=0)
        metrics = evaluate_with_metrics(server.global_model, x_test, y_test)

        server.history["round"].append(round_num)
        server.history["loss"].append(loss)
        server.history["accuracy"].append(acc)
        server.history["precision"].append(metrics["precision"])
        server.history["recall"].append(metrics["recall"])
        server.history["f1"].append(metrics["f1"])

        pbar.set_postfix(
            loss=f"{loss:.4f}",
            acc=f"{acc:.4f}",
            f1=f"{metrics['f1']:.4f}",
        )

    return server.history


def run_from_config(config_path, clients_override=None, output_dir=None):
    """Esegue tutti gli esperimenti definiti in un file config.

    Args:
        config_path: Path al file YAML.
        clients_override: Se specificato, esegue solo con questo numero di client.
        output_dir: Directory output (default: experiments/results/).
    """
    config = load_config(config_path)

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results",
        )
    os.makedirs(output_dir, exist_ok=True)

    if clients_override is not None:
        client_counts = [clients_override]
    else:
        client_counts = config.get("clients", [2, 5, 10, 20, 50])

    results = []
    for num_clients in client_counts:
        result = run_single_experiment(config, num_clients, output_dir)
        results.append(result)

    return results


def run_all(output_dir=None):
    """Esegue tutti gli esperimenti per tutti i dataset."""
    configs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs",
    )
    config_files = sorted([
        f for f in os.listdir(configs_dir) if f.endswith(".yml")
    ])

    all_results = []
    for config_file in config_files:
        config_path = os.path.join(configs_dir, config_file)
        results = run_from_config(config_path, output_dir=output_dir)
        all_results.extend(results)

    print(f"\n{'='*60}")
    print(f"Completati {len(all_results)} esperimenti")
    print(f"{'='*60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Esegui esperimenti di Federated Learning (paper DHFLPL2)",
    )
    parser.add_argument(
        "--config", type=str,
        help="Path al file di configurazione YAML",
    )
    parser.add_argument(
        "--clients", type=int, default=None,
        help="Numero di client (override config)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory di output per i risultati",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Esegui tutti gli esperimenti per tutti i dataset",
    )
    parser.add_argument(
        "--mode", type=str, choices=["simulation", "flower"],
        default="simulation",
        help="Modalita': simulation (in-process) o flower (distribuito)",
    )

    args = parser.parse_args()

    if args.mode == "flower":
        _print_flower_instructions(args)
    elif args.all:
        run_all(output_dir=args.output)
    elif args.config:
        run_from_config(
            args.config,
            clients_override=args.clients,
            output_dir=args.output,
        )
    else:
        parser.print_help()


def _print_flower_instructions(args):
    """Istruzioni per l'esecuzione in modalita' Flower distribuita."""
    print("=" * 60)
    print("MODALITA' FLOWER (deployment distribuito)")
    print("=" * 60)
    print()
    print("Per eseguire con Flower, avvia i componenti separatamente:")
    print()
    print("1. Avvia il server (SuperLink):")
    print("   FL_DATASET=cifar10 FL_NUM_ROUNDS=150 \\")
    print("     python -m src.federation.server_app")
    print()
    print("2. Avvia i client (SuperNode) in terminali separati:")
    print("   FL_SERVER_ADDRESS=localhost:8080 FL_DATASET=cifar10 \\")
    print("     FL_CLIENT_ID=1 python -m src.federation.client_app")
    print()
    print("   FL_SERVER_ADDRESS=localhost:8080 FL_DATASET=cifar10 \\")
    print("     FL_CLIENT_ID=2 python -m src.federation.client_app")
    print()
    print("Oppure usa Docker Compose:")
    print("   cd deploy/docker && docker-compose up --build")
    print()
    print("Oppure deploya su k3s:")
    print("   ./scripts/deploy.sh --clients 10 --dataset cifar10")
    print("=" * 60)


if __name__ == "__main__":
    main()
