import unittest

import numpy as np
import pandas as pd

from geomag_dataset.windows import _event_masks, _window_sums


class WindowTests(unittest.TestCase):
    def test_window_sums(self) -> None:
        mask = np.array([False, True, False, True, True], dtype=bool)
        starts = np.array([0, 1, 2])
        np.testing.assert_array_equal(_window_sums(mask, starts, 3), [1, 2, 2])

    def test_previous_year_event_buffer_blocks_new_year(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_start": pd.Timestamp("2020-12-31 22:00"),
                    "event_end": pd.Timestamp("2020-12-31 23:00"),
                }
            ]
        )
        _, blocked = _event_masks(events, 2021, 525600, buffer_minutes=24 * 60)
        self.assertTrue(blocked[0])
        self.assertTrue(blocked[23 * 60 - 1])
        self.assertFalse(blocked[23 * 60])


if __name__ == "__main__":
    unittest.main()
