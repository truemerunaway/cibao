from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .features import build_representations


class IndexedWindowDataset:
    """
    Lightweight indexed dataset.

    It loads only the Parquet index into memory. Annual arrays remain memory-mapped
    and each window is sliced on demand.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str | None = None,
        labels: Iterable[int] = (0, 1),
        feature_config: dict[str, Any] | None = None,
        baseline_minutes: int = 60,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        index_root = self.dataset_root / "windows"
        if not index_root.exists():
            raise FileNotFoundError(f"Window index does not exist: {index_root}")

        filters: list[tuple[str, str, Any]] = []
        if split is not None:
            filters.append(("split", "==", split))
        label_values = list(labels)
        if label_values:
            filters.append(("label", "in", label_values))

        self.index = pd.read_parquet(index_root, filters=filters or None)
        self.index = self.index.sort_values(
            ["center_time", "station"], kind="stable"
        ).reset_index(drop=True)
        self.feature_config = feature_config or {
            "raw_delta": True,
            "raw_diff": True,
            "psd": {"enabled": False},
            "cwt": {"enabled": False},
        }
        self.baseline_minutes = int(baseline_minutes)
        self._array_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _load_array(self, relative_path: str) -> np.ndarray:
        if relative_path not in self._array_cache:
            path = self.dataset_root / relative_path
            self._array_cache[relative_path] = np.load(path, mmap_mode="r")
        return self._array_cache[relative_path]

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.index.iloc[item]
        annual = self._load_array(str(row["source_data"]))
        start = int(row["start_offset"])
        length = int(row["length"])
        raw = np.asarray(annual[start : start + length], dtype=np.float32).T
        if raw.shape != (3, length) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid indexed window: {row['sample_id']}")

        representations = build_representations(
            raw,
            self.feature_config,
            baseline_minutes=self.baseline_minutes,
        )
        return {
            **representations,
            "label": int(row["label"]),
            "sample_id": str(row["sample_id"]),
            "station": str(row["station"]),
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "event_id": str(row["event_id"]),
        }
