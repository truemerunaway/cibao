from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterator

import numpy as np
import pandas as pd


class BalancedGroupBatchSampler:
    """Balanced batches with event-first positives and date-first negatives."""

    def __init__(
        self,
        index: pd.DataFrame,
        batch_size: int,
        samples_per_epoch: int,
        seed: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0 or self.batch_size % 2:
            raise ValueError("batch_size must be positive and even.")
        if self.samples_per_epoch <= 0 or self.samples_per_epoch % 2:
            raise ValueError("samples_per_epoch must be positive and even.")

        storm: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        nonstorm: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        centers = pd.to_datetime(index["center_time"])
        for position, row in index.reset_index(drop=True).iterrows():
            label = int(row["label"])
            station = str(row["station"])
            if label == 1:
                event_id = str(row["event_id"])
                if not event_id:
                    raise ValueError("Storm sample is missing event_id.")
                storm[event_id][station].append(position)
            elif label == 0:
                group_id = f"nonstorm_{centers.iloc[position]:%Y%m%d}"
                nonstorm[group_id][station].append(position)

        if not storm or not nonstorm:
            raise ValueError("Sampler requires both Storm and NonStorm groups.")
        self.storm = {
            group: {station: np.asarray(items) for station, items in stations.items()}
            for group, stations in storm.items()
        }
        self.nonstorm = {
            group: {station: np.asarray(items) for station, items in stations.items()}
            for group, stations in nonstorm.items()
        }
        self.storm_groups = sorted(self.storm)
        self.nonstorm_groups = sorted(self.nonstorm)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _sample_group(
        rng: np.random.Generator,
        groups: list[str],
        mapping: dict[str, dict[str, np.ndarray]],
    ) -> int:
        group = groups[int(rng.integers(len(groups)))]
        stations = sorted(mapping[group])
        station = stations[int(rng.integers(len(stations)))]
        positions = mapping[group][station]
        return int(positions[int(rng.integers(len(positions)))])

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        remaining = self.samples_per_epoch
        while remaining:
            current_size = min(self.batch_size, remaining)
            if current_size % 2:
                raise RuntimeError("Every balanced batch must contain an even count.")
            half = current_size // 2
            batch = [
                self._sample_group(rng, self.storm_groups, self.storm)
                for _ in range(half)
            ]
            batch.extend(
                self._sample_group(rng, self.nonstorm_groups, self.nonstorm)
                for _ in range(half)
            )
            rng.shuffle(batch)
            yield batch
            remaining -= current_size

    def __len__(self) -> int:
        return math.ceil(self.samples_per_epoch / self.batch_size)
