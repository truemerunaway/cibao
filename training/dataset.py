from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geomag_dataset.features import raw_delta


class WindowArrayReader:
    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)
        self._cache: dict[str, np.ndarray] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def load_raw(self, row: pd.Series) -> np.ndarray:
        relative_path = str(row["source_data"])
        if relative_path not in self._cache:
            self._cache[relative_path] = np.load(
                self.dataset_root / relative_path,
                mmap_mode="r",
            )
        annual = self._cache[relative_path]
        start = int(row["start_offset"])
        length = int(row["length"])
        raw = np.asarray(annual[start : start + length], dtype=np.float32).T
        if raw.shape != (3, length) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid indexed window: {row['sample_id']}")
        return raw


class RawDeltaDataset:
    def __init__(
        self,
        dataset_root: str | Path,
        index: pd.DataFrame,
        normalization: Any,
        baseline_minutes: int = 60,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("PyTorch is required for model training.") from exc
        self.index = index.reset_index(drop=True).copy()
        self.reader = WindowArrayReader(dataset_root)
        self.normalization = normalization
        self.baseline_minutes = int(baseline_minutes)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        import torch

        row = self.index.iloc[int(item)]
        features = raw_delta(
            self.reader.load_raw(row),
            baseline_minutes=self.baseline_minutes,
        )
        normalized = self.normalization.transform(features, str(row["station"]))
        return {
            "input": torch.from_numpy(np.ascontiguousarray(normalized)),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "position": int(item),
        }
