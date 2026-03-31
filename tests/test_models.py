"""Test per il modulo models."""

import unittest
import numpy as np
from src.models.cnn import create_model
from src.utils.config import DATASET_INFO


class TestCNNModel(unittest.TestCase):

    def test_create_model_cifar10(self):
        info = DATASET_INFO["cifar10"]
        model = create_model(info["input_shape"], info["num_classes"])
        self.assertEqual(model.output_shape, (None, 10))

    def test_create_model_cifar100(self):
        info = DATASET_INFO["cifar100"]
        model = create_model(info["input_shape"], info["num_classes"])
        self.assertEqual(model.output_shape, (None, 100))

    def test_create_model_mnist(self):
        info = DATASET_INFO["mnist"]
        model = create_model(info["input_shape"], info["num_classes"])
        self.assertEqual(model.output_shape, (None, 10))

    def test_model_predict(self):
        info = DATASET_INFO["cifar10"]
        model = create_model(info["input_shape"], info["num_classes"])
        x = np.random.rand(4, 32, 32, 3).astype("float32")
        pred = model.predict(x, verbose=0)
        self.assertEqual(pred.shape, (4, 10))
        np.testing.assert_allclose(pred.sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
