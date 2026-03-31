"""Test per il modulo data."""

import unittest
import numpy as np
from src.data.partitioner import (
    split_data_for_federated_learning,
    split_data_non_iid,
)


class TestPartitioner(unittest.TestCase):

    def setUp(self):
        self.x = np.random.rand(100, 8).astype("float32")
        self.y = np.random.randint(0, 10, size=100)

    def test_split_equal(self):
        datasets = split_data_for_federated_learning(
            (self.x, self.y), num_clients=5,
        )
        self.assertEqual(len(datasets), 5)
        for x_c, y_c in datasets:
            self.assertEqual(len(x_c), 20)
            self.assertEqual(len(y_c), 20)

    def test_split_preserves_total(self):
        datasets = split_data_for_federated_learning(
            (self.x, self.y), num_clients=4,
        )
        total = sum(len(d[0]) for d in datasets)
        self.assertEqual(total, 100)

    def test_split_non_iid(self):
        datasets = split_data_non_iid(
            (self.x, self.y), num_clients=5, num_shards_per_client=2,
        )
        self.assertEqual(len(datasets), 5)
        total = sum(len(d[0]) for d in datasets)
        self.assertEqual(total, 100)


if __name__ == "__main__":
    unittest.main()
