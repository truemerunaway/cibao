import tempfile
import unittest
from pathlib import Path

import numpy as np

from geomag_dataset.symh import build_storm_events
from geomag_dataset.utils import atomic_save_npy, minutes_in_year


class SymhEventTests(unittest.TestCase):
    def test_nearby_core_segments_merge_into_one_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            symh_dir = root / "symh"
            length = minutes_in_year(2021)
            values = np.zeros(length, dtype=np.float32)
            valid = np.ones(length, dtype=bool)

            values[100:140] = -60
            values[240:280] = -70
            atomic_save_npy(symh_dir / "2021_symh.npy", values)
            atomic_save_npy(symh_dir / "2021_valid.npy", valid)

            events = build_storm_events(
                root,
                [2021],
                storm_threshold_nt=-50,
                minimum_core_minutes=30,
                merge_gap_hours=12,
                split_config={
                    "train_years": [2021],
                    "val_years": [],
                    "test_years": [],
                },
            )
            self.assertEqual(len(events), 1)
            self.assertEqual(int(events.iloc[0]["core_minutes"]), 80)
            self.assertEqual(events.iloc[0]["split"], "train")

    def test_short_core_segments_do_not_form_event_after_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            symh_dir = root / "symh"
            length = minutes_in_year(2021)
            values = np.zeros(length, dtype=np.float32)
            valid = np.ones(length, dtype=bool)
            values[100:120] = -60
            values[240:260] = -70
            atomic_save_npy(symh_dir / "2021_symh.npy", values)
            atomic_save_npy(symh_dir / "2021_valid.npy", valid)

            events = build_storm_events(
                root,
                [2021],
                storm_threshold_nt=-50,
                minimum_core_minutes=30,
                merge_gap_hours=12,
                split_config={
                    "train_years": [2021],
                    "val_years": [],
                    "test_years": [],
                },
            )
            self.assertEqual(len(events), 0)

    def test_invalid_gap_prevents_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            symh_dir = root / "symh"
            length = minutes_in_year(2021)
            values = np.zeros(length, dtype=np.float32)
            valid = np.ones(length, dtype=bool)
            values[100:140] = -60
            values[240:280] = -70
            valid[180] = False
            atomic_save_npy(symh_dir / "2021_symh.npy", values)
            atomic_save_npy(symh_dir / "2021_valid.npy", valid)

            events = build_storm_events(
                root,
                [2021],
                storm_threshold_nt=-50,
                minimum_core_minutes=30,
                merge_gap_hours=12,
                split_config={
                    "train_years": [2021],
                    "val_years": [],
                    "test_years": [],
                },
            )
            self.assertEqual(len(events), 2)


if __name__ == "__main__":
    unittest.main()
