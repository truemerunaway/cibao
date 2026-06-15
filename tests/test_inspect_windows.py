import unittest

import pandas as pd

from inspect_windows import select_diverse_samples


class InspectionSelectionTests(unittest.TestCase):
    def test_selection_uses_distinct_time_groups(self) -> None:
        frame = pd.DataFrame(
            {
                "split": ["train"] * 6,
                "label": [1] * 6,
                "time_group_id": ["a", "a", "b", "b", "c", "c"],
                "window_start": pd.date_range("2020-01-01", periods=6, freq="h"),
            }
        )
        selected = select_diverse_samples(frame, "train", 1, 10, seed=42)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected["time_group_id"].nunique(), 3)


if __name__ == "__main__":
    unittest.main()
