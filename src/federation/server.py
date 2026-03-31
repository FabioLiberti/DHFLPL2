"""Server per Federated Learning con aggregazione FedAvg.

Il server coordina il processo federato: distribuisce il modello
globale ai client, raccoglie i pesi aggiornati e li aggrega.

Riferimento: Sezione 4.1, Algorithm 1 del paper.
"""

import numpy as np

from src.models.cnn import create_model
from src.federation.client import FLClient
from src.federation.strategy import federated_averaging
from src.metrics.evaluation import evaluate_model


class FLServer:
    """Server federato che orchestra il processo FedAvg.

    Attributes:
        global_model: Modello globale Keras.
        clients: Lista di FLClient.
        history: Storico delle metriche per round.
    """

    def __init__(self, input_shape, num_classes):
        """Inizializza il server con il modello globale.

        Args:
            input_shape: Shape dell'input del modello.
            num_classes: Numero di classi di output.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.global_model = create_model(input_shape, num_classes)
        self.clients = []
        self.history = {
            "round": [],
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

    def register_clients(self, federated_datasets):
        """Registra i client con i rispettivi dataset.

        Args:
            federated_datasets: Lista di tuple (x_train, y_train)
                per ogni client.
        """
        self.clients = []
        for i, (x_train, y_train) in enumerate(federated_datasets):
            client = FLClient(
                client_id=i,
                x_train=x_train,
                y_train=y_train,
                input_shape=self.input_shape,
                num_classes=self.num_classes,
            )
            self.clients.append(client)

    def run(self, num_rounds, test_data, local_epochs=1,
            batch_size=32, verbose=1):
        """Esegue il processo di Federated Learning.

        Per ogni round:
        1. Distribuisce i pesi globali a tutti i client
        2. Ogni client esegue training locale
        3. Raccoglie i pesi aggiornati
        4. Aggrega con FedAvg
        5. Valuta il modello globale sui dati di test

        Args:
            num_rounds: Numero di round federati (150 nel paper).
            test_data: Tuple (x_test, y_test) per la valutazione.
            local_epochs: Epoche di training locale per round.
            batch_size: Dimensione del batch.
            verbose: 0 = silenzioso, 1 = progresso, 2 = dettagliato.

        Returns:
            Dict con lo storico delle metriche.

        Riferimento: Algorithm 1 del paper.
        """
        x_test, y_test = test_data

        for round_num in range(1, num_rounds + 1):
            global_weights = self.global_model.get_weights()

            client_weights = []
            client_sizes = []

            for client in self.clients:
                client.set_weights(global_weights)
                client.train(
                    epochs=local_epochs,
                    batch_size=batch_size,
                    verbose=0,
                )
                client_weights.append(client.get_weights())
                client_sizes.append(client.num_samples)

            new_weights = federated_averaging(client_weights, client_sizes)
            self.global_model.set_weights(new_weights)

            loss, acc = self.global_model.evaluate(
                x_test, y_test, verbose=0,
            )
            metrics = evaluate_model(
                self.global_model, x_test, y_test,
            )

            self.history["round"].append(round_num)
            self.history["loss"].append(loss)
            self.history["accuracy"].append(acc)
            self.history["precision"].append(metrics["precision"])
            self.history["recall"].append(metrics["recall"])
            self.history["f1"].append(metrics["f1"])

            if verbose >= 1:
                print(
                    f"Round {round_num}/{num_rounds} - "
                    f"loss: {loss:.4f} - acc: {acc:.4f} - "
                    f"precision: {metrics['precision']:.4f} - "
                    f"recall: {metrics['recall']:.4f} - "
                    f"f1: {metrics['f1']:.4f}"
                )

        return self.history

    def get_global_weights(self):
        """Restituisce i pesi del modello globale."""
        return self.global_model.get_weights()

    def get_current_precision(self):
        """Restituisce l'ultima precision registrata.

        Utilizzata per l'autoscaling: se precision < 50%
        si aggiungono nuovi nodi worker.

        Returns:
            Ultimo valore di precision, o 0.0 se nessun round eseguito.
        """
        if self.history["precision"]:
            return self.history["precision"][-1]
        return 0.0
