import unittest

import numpy as np

from geomag_dataset.features import build_time_domain_input, raw_delta, raw_diff


class FeatureTests(unittest.TestCase):
    def test_delta_uses_leading_median(self) -> None:
        raw = np.array([[10, 12, 14, 20], [1, 3, 5, 7]], dtype=np.float32)
        result = raw_delta(raw, baseline_minutes=3)
        np.testing.assert_allclose(
            result,
            np.array([[-2, 0, 2, 8], [-2, 0, 2, 4]], dtype=np.float32),
        )

    def test_diff_preserves_shape(self) -> None:
        raw = np.array([[1, 3, 6], [4, 4, 1]], dtype=np.float32)
        result = raw_diff(raw)
        np.testing.assert_allclose(
            result,
            np.array([[0, 2, 3], [0, 0, -3]], dtype=np.float32),
        )

    def test_combined_input_has_six_channels(self) -> None:
        raw = np.arange(18, dtype=np.float32).reshape(3, 6)
        result = build_time_domain_input(raw, baseline_minutes=3)
        self.assertEqual(result.shape, (6, 6))


if __name__ == "__main__":
    unittest.main()
