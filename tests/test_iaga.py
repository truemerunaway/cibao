import tempfile
import unittest
from pathlib import Path

import numpy as np

from geomag_dataset.iaga import (
    convert_iaga_year,
    parse_iaga_header,
    select_best_source,
)


def write_iaga(path: Path, data_type: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        " Format                 IAGA-2002                                    |",
        " Station Name           Test Station                                 |",
        " IAGA Code              TST                                          |",
        " Reported               XYZG                                         |",
        " Data Interval Type     1-minute                                     |",
        f" Data Type              {data_type:<45}|",
        " Publication Date       2025-01-01                                   |",
        "DATE       TIME         DOY     TSTX      TSTY      TSTZ      TSTG   |",
        *rows,
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


class IAGATests(unittest.TestCase):
    def test_selects_highest_accepted_quality_and_rejects_variation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "TST" / "2021"
            write_iaga(directory / "provisional.txt", "Provisional", [])
            write_iaga(directory / "quasi.txt", "Quasi-definitive", [])
            write_iaga(directory / "variation.txt", "Variation", [])

            selected, candidates = select_best_source(
                root,
                "TST",
                2021,
                ["Definitive", "Quasi-definitive", "Provisional"],
                ["Variation"],
            )
            self.assertIsNotNone(selected)
            self.assertEqual(selected.data_type, "Quasi-definitive")
            self.assertEqual(len(candidates), 3)

    def test_partial_file_converts_and_marks_missing_minutes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "TST" / "2021" / "source.txt"
            write_iaga(
                source,
                "Provisional",
                [
                    "2021-01-01 00:00:00.000 001 10.0 20.0 30.0 40.0",
                    "2021-01-01 00:01:00.000 001 11.0 21.0 31.0 41.0",
                    "2021-01-01 00:02:00.000 001 99999 22.0 32.0 42.0",
                ],
            )
            header = parse_iaga_header(source)
            metadata = convert_iaga_year(
                header,
                "TST",
                2021,
                root / "output",
                ["X", "Y", "Z"],
                [99999],
                chunk_rows=2,
                overwrite=True,
            )
            data = np.load(metadata["data_path"])
            valid = np.load(metadata["valid_path"])
            self.assertEqual(data.shape, (525600, 3))
            self.assertTrue(valid[0])
            self.assertTrue(valid[1])
            self.assertFalse(valid[2])
            self.assertEqual(metadata["data_type"], "Provisional")


if __name__ == "__main__":
    unittest.main()
