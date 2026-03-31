"""Test per il modulo privacy."""

import unittest
import numpy as np
from src.privacy.dp_mechanism import (
    redact_private_data,
    redact_private_fields,
    add_gaussian_noise,
    add_laplace_noise,
    clip_gradients,
    apply_dp_to_weights,
)
from src.privacy.threat_model import (
    ModelUpdateLeakage,
    MembershipInferenceAttack,
    SideChannelAnalyzer,
)


class TestRedaction(unittest.TestCase):

    def test_redact_email(self):
        text = "Contact me at user@example.com for info"
        redacted, counts = redact_private_data(text)
        self.assertNotIn("user@example.com", redacted)
        self.assertIn("[REDACTED]", redacted)
        self.assertGreaterEqual(counts["email"], 1)

    def test_redact_phone(self):
        text = "Call me at +39 333 1234567"
        redacted, counts = redact_private_data(text)
        self.assertIn("[REDACTED]", redacted)

    def test_redact_fields(self):
        record = {"name": "Mario Rossi", "email": "m@r.it", "age": 30}
        redacted = redact_private_fields(record, ["email"])
        self.assertEqual(redacted["email"], "[REDACTED]")
        self.assertEqual(redacted["name"], "Mario Rossi")
        self.assertEqual(redacted["age"], 30)


class TestNoiseMechanisms(unittest.TestCase):

    def test_gaussian_noise_shape(self):
        data = np.zeros((10, 5))
        noisy = add_gaussian_noise(data, epsilon=1.0, delta=1e-5)
        self.assertEqual(noisy.shape, data.shape)

    def test_gaussian_noise_not_zero(self):
        data = np.zeros((100,))
        noisy = add_gaussian_noise(data, epsilon=1.0, delta=1e-5)
        self.assertFalse(np.allclose(noisy, data))

    def test_laplace_noise_shape(self):
        data = np.ones((10,))
        noisy = add_laplace_noise(data, epsilon=1.0)
        self.assertEqual(noisy.shape, data.shape)

    def test_lower_epsilon_more_noise(self):
        data = np.zeros((1000,))
        noisy_high_eps = add_gaussian_noise(data, epsilon=10.0, delta=1e-5)
        noisy_low_eps = add_gaussian_noise(data, epsilon=0.1, delta=1e-5)
        self.assertGreater(
            np.std(noisy_low_eps), np.std(noisy_high_eps),
        )


class TestGradientClipping(unittest.TestCase):

    def test_clip_reduces_norm(self):
        grads = [np.ones((10,)) * 10]
        clipped = clip_gradients(grads, clip_norm=1.0)
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in clipped))
        self.assertLessEqual(total_norm, 1.0 + 1e-6)

    def test_clip_no_change_if_small(self):
        grads = [np.ones((3,)) * 0.1]
        clipped = clip_gradients(grads, clip_norm=10.0)
        np.testing.assert_allclose(clipped[0], grads[0])


class TestDPWeights(unittest.TestCase):

    def test_apply_dp_changes_weights(self):
        weights = [np.ones((5, 5)), np.ones((5,))]
        dp_weights = apply_dp_to_weights(
            weights, epsilon=1.0, delta=1e-5, clip_norm=1.0,
        )
        self.assertEqual(len(dp_weights), len(weights))
        self.assertFalse(np.allclose(dp_weights[0], weights[0]))


class TestModelUpdateLeakage(unittest.TestCase):

    def test_analyze_empty(self):
        analyzer = ModelUpdateLeakage()
        result = analyzer.analyze_leakage_risk()
        self.assertEqual(result["risk_level"], "unknown")

    def test_analyze_low_risk(self):
        analyzer = ModelUpdateLeakage()
        w1 = [np.zeros((5,))]
        w2 = [np.ones((5,)) * 0.01]
        for i in range(5):
            analyzer.record_update(w1, w2, round_num=i)
        result = analyzer.analyze_leakage_risk()
        self.assertEqual(result["risk_level"], "low")


class TestSideChannelAnalyzer(unittest.TestCase):

    def test_analyze_empty(self):
        analyzer = SideChannelAnalyzer()
        result = analyzer.analyze()
        self.assertEqual(result["risk_level"], "unknown")

    def test_analyze_uniform(self):
        analyzer = SideChannelAnalyzer()
        weights = [np.ones((10,))]
        for i in range(5):
            analyzer.record_communication(i, weights, elapsed_time=0.5)
        result = analyzer.analyze()
        self.assertEqual(result["risk_level"], "low")


if __name__ == "__main__":
    unittest.main()
