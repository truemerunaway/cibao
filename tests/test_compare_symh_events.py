import tempfile
import unittest
from pathlib import Path

import pandas as pd

from compare_symh_events import build_reference_events, load_symh_series


class CompareSymhEventsTests(unittest.TestCase):
    def test_builds_reference_events_from_five_column_symh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "symh_2024.txt"
            rows = []
            for offset in range(180):
                value = -60 if 10 <= offset < 80 else -10
                rows.append(f"2024 1 {offset // 60} {offset % 60} {value}")
            path.write_text("\n".join(rows), encoding="utf-8")

            symh = load_symh_series(path, 2024)
            events = build_reference_events(
                symh,
                year=2024,
                threshold=-50,
                minimum_core_minutes=60,
                merge_gap_hours=12,
            )

            self.assertEqual(len(events), 1)
            self.assertEqual(
                events.iloc[0]["event_start"],
                pd.Timestamp("2024-01-01 00:10"),
            )
            self.assertEqual(events.iloc[0]["core_minutes"], 70)


if __name__ == "__main__":
    unittest.main()
