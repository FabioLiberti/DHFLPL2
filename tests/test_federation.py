"""Test per il modulo federation."""

import unittest
import numpy as np
from src.federation.strategy import federated_averaging, simple_averaging
from src.federation.client import FLClient


class TestFedAvg(unittest.TestCase):

    def test_weighted_averaging(self):
        w1 = [np.array([1.0, 2.0]), np.array([3.0])]
        w2 = [np.array([3.0, 4.0]), np.array([5.0])]
        # Client 1: 75 samples, Client 2: 25 samples
        result = federated_averaging([w1, w2], [75, 25])
        np.testing.assert_allclose(result[0], [1.5, 2.5])
        np.testing.assert_allclose(result[1], [3.5])

    def test_simple_averaging(self):
        w1 = [np.array([1.0, 2.0])]
        w2 = [np.array([3.0, 4.0])]
        result = simple_averaging([w1, w2])
        np.testing.assert_allclose(result[0], [2.0, 3.0])

    def test_equal_weights_same_as_simple(self):
        w1 = [np.array([2.0, 4.0])]
        w2 = [np.array([6.0, 8.0])]
        weighted = federated_averaging([w1, w2], [50, 50])
        simple = simple_averaging([w1, w2])
        np.testing.assert_allclose(weighted[0], simple[0])


class TestFLClient(unittest.TestCase):

    def test_client_creation(self):
        x = np.random.rand(50, 28, 28, 1).astype("float32")
        y = np.random.randint(0, 10, size=50)
        client = FLClient(0, x, y, (28, 28, 1), 10)
        self.assertEqual(client.client_id, 0)
        self.assertEqual(client.num_samples, 50)

    def test_client_get_set_weights(self):
        x = np.random.rand(20, 28, 28, 1).astype("float32")
        y = np.random.randint(0, 10, size=20)
        client = FLClient(0, x, y, (28, 28, 1), 10)
        weights = client.get_weights()
        client.set_weights(weights)
        new_weights = client.get_weights()
        for w1, w2 in zip(weights, new_weights):
            np.testing.assert_array_equal(w1, w2)

    def test_client_train(self):
        x = np.random.rand(20, 28, 28, 1).astype("float32")
        y = np.random.randint(0, 10, size=20)
        client = FLClient(0, x, y, (28, 28, 1), 10)
        history = client.train(epochs=1, batch_size=10)
        self.assertIn("loss", history.history)


if __name__ == "__main__":
    unittest.main()
