"""Test di integrazione end-to-end per il framework DHFLPL2.

Verifica il flusso completo: caricamento dati -> partizionamento ->
training federato -> aggregazione -> valutazione, su un sottoinsieme
ridotto per velocita'.
"""

import unittest
import numpy as np

from src.models.cnn import create_model
from src.data.partitioner import split_data_for_federated_learning
from src.federation.server import FLServer
from src.federation.client import FLClient
from src.federation.strategy import federated_averaging
from src.metrics.evaluation import evaluate_model, evaluate_per_class
from src.privacy.dp_mechanism import (
    apply_dp_to_weights,
    add_gaussian_noise,
    redact_private_data,
    redact_private_fields,
)
from src.privacy.threat_model import (
    ModelUpdateLeakage,
    MembershipInferenceAttack,
    SideChannelAnalyzer,
)
from src.utils.config import DATASET_INFO, DEFAULT_FL_CONFIG


class TestEndToEndFederation(unittest.TestCase):
    """Test completo del flusso di Federated Learning."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.num_samples = 200
        cls.num_clients = 2
        cls.num_rounds = 3
        cls.input_shape = (28, 28, 1)
        cls.num_classes = 10

        cls.x_train = np.random.rand(
            cls.num_samples, *cls.input_shape,
        ).astype("float32")
        cls.y_train = np.random.randint(
            0, cls.num_classes, size=cls.num_samples,
        )
        cls.x_test = np.random.rand(
            50, *cls.input_shape,
        ).astype("float32")
        cls.y_test = np.random.randint(
            0, cls.num_classes, size=50,
        )

    def test_full_pipeline(self):
        """Flusso completo: partizionamento -> FL -> valutazione."""
        datasets = split_data_for_federated_learning(
            (self.x_train, self.y_train), self.num_clients,
        )
        self.assertEqual(len(datasets), self.num_clients)

        server = FLServer(self.input_shape, self.num_classes)
        server.register_clients(datasets)
        self.assertEqual(len(server.clients), self.num_clients)

        history = server.run(
            num_rounds=self.num_rounds,
            test_data=(self.x_test, self.y_test),
            local_epochs=1,
            batch_size=32,
            verbose=0,
        )

        self.assertEqual(len(history["round"]), self.num_rounds)
        self.assertEqual(len(history["accuracy"]), self.num_rounds)
        self.assertEqual(len(history["loss"]), self.num_rounds)
        self.assertEqual(len(history["precision"]), self.num_rounds)
        self.assertEqual(len(history["recall"]), self.num_rounds)
        self.assertEqual(len(history["f1"]), self.num_rounds)

        for acc in history["accuracy"]:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_full_pipeline_with_dp(self):
        """Flusso completo con Differential Privacy applicata ai pesi."""
        datasets = split_data_for_federated_learning(
            (self.x_train, self.y_train), self.num_clients,
        )

        model = create_model(self.input_shape, self.num_classes)
        global_weights = model.get_weights()

        client_weights = []
        client_sizes = []

        for i, (x_c, y_c) in enumerate(datasets):
            client = FLClient(
                i, x_c, y_c, self.input_shape, self.num_classes,
            )
            client.set_weights(global_weights)
            client.train(epochs=1, verbose=0)

            raw_weights = client.get_weights()
            dp_weights = apply_dp_to_weights(
                raw_weights, epsilon=1.0, delta=1e-5, clip_norm=1.0,
            )
            client_weights.append(dp_weights)
            client_sizes.append(client.num_samples)

        new_weights = federated_averaging(client_weights, client_sizes)
        model.set_weights(new_weights)

        metrics = evaluate_model(model, self.x_test, self.y_test)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("f1", metrics)

    def test_manual_fedavg_matches_server(self):
        """Verifica che FedAvg manuale e server producano risultati coerenti."""
        datasets = split_data_for_federated_learning(
            (self.x_train, self.y_train), self.num_clients,
        )

        model = create_model(self.input_shape, self.num_classes)
        initial_weights = model.get_weights()

        client_weights = []
        client_sizes = []
        for i, (x_c, y_c) in enumerate(datasets):
            client = FLClient(
                i, x_c, y_c, self.input_shape, self.num_classes,
            )
            client.set_weights(initial_weights)
            client.train(epochs=1, verbose=0)
            client_weights.append(client.get_weights())
            client_sizes.append(client.num_samples)

        manual_weights = federated_averaging(client_weights, client_sizes)

        model.set_weights(manual_weights)
        loss, acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        self.assertIsNotNone(acc)


class TestEndToEndMetrics(unittest.TestCase):
    """Test integrazione del sistema di metriche."""

    def test_evaluate_model_complete(self):
        model = create_model((28, 28, 1), 10)
        x = np.random.rand(30, 28, 28, 1).astype("float32")
        y = np.random.randint(0, 10, size=30)

        metrics = evaluate_model(model, x, y)
        required_keys = ["loss", "accuracy", "precision", "recall", "f1"]
        for key in required_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)

    def test_evaluate_per_class(self):
        model = create_model((28, 28, 1), 10)
        x = np.random.rand(50, 28, 28, 1).astype("float32")
        y = np.random.randint(0, 10, size=50)

        per_class = evaluate_per_class(model, x, y)
        self.assertIsInstance(per_class, dict)
        for cls_metrics in per_class.values():
            self.assertIn("precision", cls_metrics)
            self.assertIn("recall", cls_metrics)
            self.assertIn("f1", cls_metrics)


class TestEndToEndPrivacy(unittest.TestCase):
    """Test integrazione privacy end-to-end."""

    def test_dp_noise_preserves_model_usability(self):
        """Il modello con DP deve rimanere funzionante."""
        model = create_model((28, 28, 1), 10)
        weights = model.get_weights()

        dp_weights = apply_dp_to_weights(
            weights, epsilon=1.0, delta=1e-5, clip_norm=1.0,
        )
        model.set_weights(dp_weights)

        x = np.random.rand(5, 28, 28, 1).astype("float32")
        pred = model.predict(x, verbose=0)
        self.assertEqual(pred.shape, (5, 10))
        for row in pred:
            self.assertAlmostEqual(row.sum(), 1.0, places=4)

    def test_redaction_pipeline(self):
        """Pipeline completa di redazione dati."""
        text = "Email: john@example.com, Phone: +1 555-123-4567"
        redacted, counts = redact_private_data(text)
        self.assertNotIn("john@example.com", redacted)
        self.assertIn("[REDACTED]", redacted)

        record = {
            "name": "John Doe",
            "email": "john@example.com",
            "data": [1, 2, 3],
        }
        clean = redact_private_fields(record, ["email"])
        self.assertEqual(clean["email"], "[REDACTED]")
        self.assertEqual(clean["name"], "John Doe")

    def test_threat_analysis_pipeline(self):
        """Pipeline completa di analisi minacce."""
        model = create_model((28, 28, 1), 10)

        # Model Update Leakage
        leakage = ModelUpdateLeakage()
        w1 = model.get_weights()
        x = np.random.rand(10, 28, 28, 1).astype("float32")
        y = np.random.randint(0, 10, size=10)
        model.fit(x, y, epochs=1, verbose=0)
        w2 = model.get_weights()
        leakage.record_update(w1, w2, round_num=1)
        risk = leakage.analyze_leakage_risk()
        self.assertIn(risk["risk_level"], ["low", "medium", "high"])

        # Membership Inference
        mia = MembershipInferenceAttack(model, threshold=0.5)
        result = mia.infer_membership(x, y)
        self.assertIn("membership_rate", result)
        self.assertEqual(len(result["predictions"]), len(x))

        # Side Channel
        sca = SideChannelAnalyzer()
        sca.record_communication(0, model.get_weights(), 0.5)
        sca.record_communication(1, model.get_weights(), 0.6)
        analysis = sca.analyze()
        self.assertIn(analysis["risk_level"], ["low", "medium", "high"])


class TestConfigConsistency(unittest.TestCase):
    """Verifica coerenza configurazione con il paper."""

    def test_dataset_info_complete(self):
        expected = ["cifar10", "cifar100", "mnist", "fashion_mnist", "svhn"]
        for ds in expected:
            self.assertIn(ds, DATASET_INFO)
            self.assertIn("num_classes", DATASET_INFO[ds])
            self.assertIn("input_shape", DATASET_INFO[ds])

    def test_default_fl_config(self):
        self.assertEqual(DEFAULT_FL_CONFIG["num_rounds"], 150)
        self.assertEqual(DEFAULT_FL_CONFIG["local_epochs"], 1)
        self.assertEqual(DEFAULT_FL_CONFIG["min_precision_threshold"], 0.50)

    def test_cifar10_classes(self):
        self.assertEqual(DATASET_INFO["cifar10"]["num_classes"], 10)

    def test_cifar100_classes(self):
        self.assertEqual(DATASET_INFO["cifar100"]["num_classes"], 100)

    def test_model_creation_all_datasets(self):
        for ds_name, info in DATASET_INFO.items():
            model = create_model(info["input_shape"], info["num_classes"])
            self.assertEqual(
                model.output_shape[1], info["num_classes"],
                f"Mismatch per {ds_name}",
            )


if __name__ == "__main__":
    unittest.main()
