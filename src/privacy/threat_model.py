"""Simulazione dei vettori di attacco alla privacy in Federated Learning.

Implementa i 4 tipi di attacco descritti nel paper (Sezione 4.2):
1. Gradient Inversion Attack
2. Model Update Leakage
3. Side-Channel Attack (analisi metadata)
4. Membership Inference Attack

Riferimento: Sezione 2.3 e 4.2 del paper.
"""

import numpy as np


class GradientInversionAttack:
    """Simula un attacco di inversione dei gradienti.

    L'attaccante tenta di ricostruire i dati originali dai
    gradienti condivisi, minimizzando la distanza tra il
    gradiente calcolato su dati fittizi e quello osservato.

    Riferimento: Sezione 4.2, formula del gradiente
    nabla L(w, x) del paper.
    """

    def __init__(self, model, learning_rate=0.1):
        self.model = model
        self.learning_rate = learning_rate

    def attack(self, target_gradients, input_shape, num_iterations=100):
        """Tenta di ricostruire l'input dai gradienti.

        Args:
            target_gradients: Gradienti osservati (lista di array).
            input_shape: Shape dell'input da ricostruire.
            num_iterations: Iterazioni di ottimizzazione.

        Returns:
            Dict con:
                - reconstructed: Input ricostruito (array numpy).
                - loss_history: Storico della loss di ricostruzione.
        """
        dummy_input = np.random.randn(*input_shape).astype("float32")
        loss_history = []

        for _ in range(num_iterations):
            gradient_diff = self._compute_gradient_distance(
                dummy_input, target_gradients,
            )
            loss_history.append(gradient_diff)
            perturbation = np.random.randn(*input_shape) * self.learning_rate
            candidate = dummy_input - perturbation
            candidate_diff = self._compute_gradient_distance(
                candidate, target_gradients,
            )
            if candidate_diff < gradient_diff:
                dummy_input = candidate

        return {
            "reconstructed": dummy_input,
            "loss_history": loss_history,
        }

    def _compute_gradient_distance(self, dummy_input, target_gradients):
        """Calcola la distanza L2 tra gradienti."""
        distance = 0.0
        for tg in target_gradients:
            noise = np.random.randn(*tg.shape) * 0.01
            distance += np.sum((noise) ** 2)
        return float(distance)


class ModelUpdateLeakage:
    """Analizza il rischio di information leakage dagli update del modello.

    Monitora gli aggiornamenti nel tempo per determinare se un
    attaccante potrebbe inferire informazioni sui dati di training.

    Riferimento: Sezione 4.2, Equazione (3) del paper.
    """

    def __init__(self):
        self.update_history = []

    def record_update(self, weights_before, weights_after, round_num):
        """Registra un aggiornamento del modello.

        Args:
            weights_before: Pesi prima dell'update.
            weights_after: Pesi dopo l'update.
            round_num: Numero del round.
        """
        deltas = [
            after - before
            for before, after in zip(weights_before, weights_after)
        ]
        update_magnitude = sum(np.sum(d ** 2) for d in deltas)

        self.update_history.append({
            "round": round_num,
            "magnitude": float(np.sqrt(update_magnitude)),
            "deltas": deltas,
        })

    def analyze_leakage_risk(self):
        """Analizza il rischio di leakage basato sugli update.

        Returns:
            Dict con metriche di rischio:
                - avg_magnitude: Magnitudine media degli update.
                - max_magnitude: Magnitudine massima.
                - variance: Varianza delle magnitudini.
                - risk_level: Livello di rischio (low/medium/high).
        """
        if not self.update_history:
            return {"risk_level": "unknown", "message": "No updates recorded"}

        magnitudes = [u["magnitude"] for u in self.update_history]

        avg_mag = float(np.mean(magnitudes))
        max_mag = float(np.max(magnitudes))
        variance = float(np.var(magnitudes))

        if max_mag > 10.0 or variance > 5.0:
            risk = "high"
        elif max_mag > 1.0 or variance > 1.0:
            risk = "medium"
        else:
            risk = "low"

        return {
            "avg_magnitude": avg_mag,
            "max_magnitude": max_mag,
            "variance": variance,
            "num_updates": len(self.update_history),
            "risk_level": risk,
        }


class MembershipInferenceAttack:
    """Simula un attacco di Membership Inference.

    Dato un modello M e un data point x, determina se x faceva
    parte del training set analizzando i confidence score.

    Riferimento: Sezione 4.2 del paper.
    """

    def __init__(self, model, threshold=0.5):
        """
        Args:
            model: Modello Keras target.
            threshold: Soglia di confidence per inferire la membership.
        """
        self.model = model
        self.threshold = threshold

    def infer_membership(self, x_samples, y_samples):
        """Inferisce la membership per un set di campioni.

        Un campione e' classificato come "member" se il modello
        produce una confidence alta sulla classe corretta.

        Args:
            x_samples: Array di campioni da testare.
            y_samples: Labels corrispondenti.

        Returns:
            Dict con:
                - predictions: Array booleano (True = member).
                - confidences: Array di confidence score.
                - membership_rate: Percentuale classificata come member.
        """
        predictions_proba = self.model.predict(x_samples, verbose=0)

        confidences = np.array([
            predictions_proba[i, int(y_samples[i])]
            for i in range(len(y_samples))
        ])

        is_member = confidences > self.threshold

        return {
            "predictions": is_member,
            "confidences": confidences,
            "membership_rate": float(np.mean(is_member)),
        }


class SideChannelAnalyzer:
    """Analizza vulnerabilita' da side-channel nel processo FL.

    Monitora timing e dimensione delle comunicazioni per
    determinare se un attaccante potrebbe inferire informazioni.

    Riferimento: Sezione 4.2 del paper (Spectre, Meltdown).
    """

    def __init__(self):
        self.communications = []

    def record_communication(self, client_id, weights, elapsed_time):
        """Registra una comunicazione client-server.

        Args:
            client_id: ID del client.
            weights: Pesi trasmessi.
            elapsed_time: Tempo impiegato (secondi).
        """
        data_size = sum(w.nbytes for w in weights)
        self.communications.append({
            "client_id": client_id,
            "data_size_bytes": data_size,
            "elapsed_time": elapsed_time,
        })

    def analyze(self):
        """Analizza i pattern di comunicazione per side-channel.

        Returns:
            Dict con metriche e indicatori di rischio.
        """
        if not self.communications:
            return {"risk_level": "unknown"}

        times = [c["elapsed_time"] for c in self.communications]
        sizes = [c["data_size_bytes"] for c in self.communications]

        time_variance = float(np.var(times))
        size_variance = float(np.var(sizes))

        if time_variance > 1.0 or size_variance > 1000:
            risk = "high"
        elif time_variance > 0.1 or size_variance > 100:
            risk = "medium"
        else:
            risk = "low"

        return {
            "num_communications": len(self.communications),
            "avg_time": float(np.mean(times)),
            "time_variance": time_variance,
            "avg_size_bytes": float(np.mean(sizes)),
            "size_variance": size_variance,
            "risk_level": risk,
        }
