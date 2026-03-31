"""Client per Federated Learning.

Ogni client riceve il modello globale, esegue il training locale
sui propri dati e restituisce i pesi aggiornati al server.

Riferimento: Listing 2 del paper.
"""

from src.models.cnn import create_model


class FLClient:
    """Client federato che esegue training locale.

    Attributes:
        client_id: Identificativo del client.
        model: Modello Keras locale.
        x_train: Dati di training locali.
        y_train: Labels di training locali.
    """

    def __init__(self, client_id, x_train, y_train, input_shape, num_classes):
        """Inizializza il client con i dati locali.

        Args:
            client_id: Identificativo univoco del client.
            x_train: Features di training locali.
            y_train: Labels di training locali.
            input_shape: Shape dell'input per il modello.
            num_classes: Numero di classi.
        """
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.model = create_model(input_shape, num_classes)

    def set_weights(self, weights):
        """Aggiorna i pesi del modello locale con quelli globali.

        Args:
            weights: Lista di array numpy con i pesi del modello globale.
        """
        self.model.set_weights(weights)

    def get_weights(self):
        """Restituisce i pesi del modello locale.

        Returns:
            Lista di array numpy.
        """
        return self.model.get_weights()

    def train(self, epochs=1, batch_size=32, verbose=0):
        """Esegue il training locale sul dataset del client.

        Args:
            epochs: Numero di epoche locali (default 1 come nel paper).
            batch_size: Dimensione del batch.
            verbose: Livello di verbosita' (0 = silenzioso).

        Returns:
            History object di Keras con le metriche di training.
        """
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        return history

    @property
    def num_samples(self):
        """Numero di campioni nel dataset locale."""
        return len(self.x_train)
