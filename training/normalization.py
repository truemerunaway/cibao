from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geomag_dataset.features import raw_delta

from .dataset import WindowArrayReader

LOGGER = logging.getLogger(__name__)


def _stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _channel_stats(values: np.ndarray) -> dict[str, list[float]]:
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], axis=1)
    return {
        "median": median.astype(float).tolist(),
        "q1": q1.astype(float).tolist(),
        "q3": q3.astype(float).tolist(),
        "iqr": (q3 - q1).astype(float).tolist(),
    }


class StationRobustNormalizer:
    def __init__(
        self,
        station_stats: dict[str, dict[str, Any]],
        global_stats: dict[str, Any],
        epsilon: float = 1.0e-6,
        clip: float = 10.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.station_stats = station_stats
        self.global_stats = global_stats
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.metadata = metadata or {}
        self._warned_stations: set[str] = set()

    @classmethod
    def fit(
        cls,
        dataset_root: str | Path,
        training_index: pd.DataFrame,
        training_years: list[int],
        baseline_minutes: int,
        max_windows_per_station: int,
        global_max_points_per_station_channel: int,
        seed: int,
        epsilon: float,
        clip: float,
    ) -> "StationRobustNormalizer":
        if training_index.empty:
            raise ValueError("Cannot fit normalization on an empty training index.")
        if "group_year" in training_index:
            observed_years = set(training_index["group_year"].astype(int))
        else:
            observed_years = set(pd.to_datetime(training_index["center_time"]).dt.year)
        unexpected = observed_years - set(training_years)
        if unexpected:
            raise ValueError(
                f"Normalization index contains non-training years: {sorted(unexpected)}"
            )

        reader = WindowArrayReader(dataset_root)
        station_stats: dict[str, dict[str, Any]] = {}
        global_samples: list[list[np.ndarray]] = [[], [], []]
        total_sampled_windows = 0

        for station, station_frame in training_index.groupby("station", sort=True):
            sample_count = min(int(max_windows_per_station), len(station_frame))
            rng = np.random.default_rng(_stable_seed(seed, str(station)))
            selected = np.sort(
                rng.choice(len(station_frame), size=sample_count, replace=False)
            )
            windows = np.empty((sample_count, 3, int(station_frame.iloc[0]["length"])))
            windows = windows.astype(np.float32, copy=False)
            for output_index, row_index in enumerate(selected):
                row = station_frame.iloc[int(row_index)]
                windows[output_index] = raw_delta(
                    reader.load_raw(row),
                    baseline_minutes=baseline_minutes,
                )
            channel_values = windows.transpose(1, 0, 2).reshape(3, -1)
            stats = _channel_stats(channel_values)
            stats.update(
                {
                    "sampled_windows": sample_count,
                    "sampled_points_per_channel": int(channel_values.shape[1]),
                    "available_training_windows": int(len(station_frame)),
                }
            )
            station_stats[str(station)] = stats
            total_sampled_windows += sample_count

            global_limit = min(
                int(global_max_points_per_station_channel),
                channel_values.shape[1],
            )
            for channel in range(3):
                if global_limit < channel_values.shape[1]:
                    indices = rng.choice(
                        channel_values.shape[1],
                        size=global_limit,
                        replace=False,
                    )
                    global_samples[channel].append(channel_values[channel, indices])
                else:
                    global_samples[channel].append(channel_values[channel])

        global_values = np.stack(
            [np.concatenate(parts) for parts in global_samples],
            axis=0,
        )
        global_stats = _channel_stats(global_values)
        global_stats["sampled_points_per_channel"] = int(global_values.shape[1])
        metadata = {
            "method": "station_robust",
            "training_years": [int(year) for year in training_years],
            "seed": int(seed),
            "baseline_minutes": int(baseline_minutes),
            "max_windows_per_station": int(max_windows_per_station),
            "total_sampled_windows": int(total_sampled_windows),
            "fit_window_count": int(len(training_index)),
        }
        return cls(
            station_stats=station_stats,
            global_stats=global_stats,
            epsilon=epsilon,
            clip=clip,
            metadata=metadata,
        )

    def _stats_for(self, station: str) -> dict[str, Any]:
        station = station.upper()
        if station in self.station_stats:
            return self.station_stats[station]
        if station not in self._warned_stations:
            LOGGER.warning(
                "No normalization statistics for station %s; using global XYZ values.",
                station,
            )
            self._warned_stations.add(station)
        return self.global_stats

    def transform(self, values: np.ndarray, station: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != 3:
            raise ValueError("Expected raw_delta with shape [3, time].")
        stats = self._stats_for(station)
        center = np.asarray(stats["median"], dtype=np.float32)[:, None]
        scale = np.asarray(stats["iqr"], dtype=np.float32)[:, None]
        normalized = (values - center) / np.maximum(scale, self.epsilon)
        normalized = np.clip(normalized, -self.clip, self.clip)
        if not np.isfinite(normalized).all():
            raise ValueError("Normalization produced non-finite values.")
        return normalized.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_stats": self.station_stats,
            "global_stats": self.global_stats,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(output)

    @classmethod
    def load(cls, path: str | Path) -> "StationRobustNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
