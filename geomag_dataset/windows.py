from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import minutes_in_year, split_for_year

LOGGER = logging.getLogger(__name__)


def _window_sums(mask: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    prefix = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    return prefix[starts + length] - prefix[starts]


def _event_masks(
    events: pd.DataFrame,
    year: int,
    year_minutes: int,
    buffer_minutes: int,
) -> tuple[np.ndarray, np.ndarray]:
    event_index = np.full(year_minutes, -1, dtype=np.int32)
    blocked = np.zeros(year_minutes, dtype=bool)
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year + 1, month=1, day=1)

    for index, event in events.iterrows():
        event_start = pd.Timestamp(event["event_start"])
        event_end = pd.Timestamp(event["event_end"])
        buffered_start = event_start - pd.Timedelta(minutes=buffer_minutes)
        buffered_end = event_end + pd.Timedelta(minutes=buffer_minutes)
        if buffered_end <= year_start or buffered_start >= year_end:
            continue

        start_offset = int(np.floor((event_start - year_start).total_seconds() / 60))
        end_offset = int(np.ceil((event_end - year_start).total_seconds() / 60))
        start = max(0, start_offset)
        end = min(year_minutes, end_offset)
        if start < end:
            event_index[start:end] = int(index)

        blocked_start = max(
            0, int(np.floor((buffered_start - year_start).total_seconds() / 60))
        )
        blocked_end = min(
            year_minutes, int(np.ceil((buffered_end - year_start).total_seconds() / 60))
        )
        if blocked_start < blocked_end:
            blocked[blocked_start:blocked_end] = True

    return event_index, blocked


def _nearest_event_distance_minutes(
    centers: np.ndarray,
    events: pd.DataFrame,
    year: int,
) -> np.ndarray:
    distances = np.full(centers.shape, np.inf, dtype=np.float64)
    if events.empty:
        return distances
    year_start = pd.Timestamp(year=year, month=1, day=1)

    for _, event in events.iterrows():
        start = (pd.Timestamp(event["event_start"]) - year_start).total_seconds() / 60
        end = (pd.Timestamp(event["event_end"]) - year_start).total_seconds() / 60
        distance = np.where(
            centers < start,
            start - centers,
            np.where(centers >= end, centers - end, 0.0),
        )
        distances = np.minimum(distances, distance)
    return distances


def _window_symh_minimum(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    result = np.full(starts.shape, np.nan, dtype=np.float32)
    for output_index, start in enumerate(starts):
        window = values[start : start + length]
        finite = np.isfinite(window)
        if finite.any():
            result[output_index] = np.min(window[finite])
    return result


def build_window_partition(
    output_root: Path,
    station: str,
    year: int,
    events: pd.DataFrame,
    window_config: dict[str, Any],
    symh_config: dict[str, Any],
    split_config: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    station = station.upper()
    continuous_dir = output_root / "continuous" / station
    data_path = continuous_dir / f"{year}_data.npy"
    valid_path = continuous_dir / f"{year}_valid.npy"
    metadata_path = continuous_dir / f"{year}_metadata.json"
    symh_path = output_root / "symh" / f"{year}_symh.npy"
    symh_valid_path = output_root / "symh" / f"{year}_valid.npy"
    partition_dir = output_root / "windows" / f"year={year}"
    partition_path = partition_dir / f"{station}.parquet"

    if not overwrite and partition_path.exists():
        frame = pd.read_parquet(partition_path, columns=["label"])
        return {
            "station": station,
            "year": year,
            "status": "EXISTING",
            "windows": len(frame),
            "storm_windows": int((frame["label"] == 1).sum()),
            "nonstorm_windows": int((frame["label"] == 0).sum()),
            "uncertain_windows": int((frame["label"] == -1).sum()),
        }

    required = (data_path, valid_path, metadata_path, symh_path, symh_valid_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {
            "station": station,
            "year": year,
            "status": "MISSING_INPUT",
            "missing": missing,
            "windows": 0,
        }

    with metadata_path.open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)

    data_valid = np.load(valid_path, mmap_mode="r").astype(bool, copy=False)
    symh_values = np.load(symh_path, mmap_mode="r")
    symh_valid = np.load(symh_valid_path, mmap_mode="r").astype(bool, copy=False)
    expected = minutes_in_year(year)
    if len(data_valid) != expected or len(symh_values) != expected:
        raise ValueError(f"Unexpected annual array length for {station} {year}")

    window_length = int(round(float(window_config["length_hours"]) * 60))
    stride = int(round(float(window_config["stride_hours"]) * 60))
    if window_length <= 0 or stride <= 0:
        raise ValueError("Window length and stride must be positive.")
    starts = np.arange(0, expected - window_length + 1, stride, dtype=np.int64)
    centers = starts + window_length // 2

    invalid_data_count = _window_sums(~data_valid, starts, window_length)
    complete = invalid_data_count == 0
    starts = starts[complete]
    centers = centers[complete]
    if starts.size == 0:
        return {
            "station": station,
            "year": year,
            "status": "NO_COMPLETE_WINDOWS",
            "windows": 0,
        }

    event_index, blocked = _event_masks(
        events,
        year,
        expected,
        buffer_minutes=int(round(float(symh_config["nonstorm_buffer_hours"]) * 60)),
    )
    storm = event_index[centers] >= 0

    symh_bad = (~symh_valid) | (
        symh_values <= float(symh_config["nonstorm_threshold_nt"])
    )
    symh_bad_count = _window_sums(symh_bad, starts, window_length)
    blocked_count = _window_sums(blocked, starts, window_length)
    nonstorm = (~storm) & (symh_bad_count == 0) & (blocked_count == 0)
    labels = np.full(starts.shape, -1, dtype=np.int8)
    labels[nonstorm] = 0
    labels[storm] = 1

    year_start = pd.Timestamp(year=year, month=1, day=1)
    window_starts = year_start + pd.to_timedelta(starts, unit="m")
    window_ends = window_starts + pd.Timedelta(minutes=window_length)
    center_times = year_start + pd.to_timedelta(centers, unit="m")
    symh_min = _window_symh_minimum(symh_values, starts, window_length)
    nearest_distance = _nearest_event_distance_minutes(centers, events, year)

    event_ids: list[str] = []
    time_groups: list[str] = []
    splits: list[str] = []
    label_names = np.where(labels == 1, "Storm", np.where(labels == 0, "NonStorm", "Uncertain"))
    year_split = split_for_year(year, split_config)

    for label, center, event_row_index in zip(labels, center_times, event_index[centers]):
        if label == 1 and event_row_index >= 0:
            event = events.loc[int(event_row_index)]
            event_id = str(event["event_id"])
            event_ids.append(event_id)
            time_groups.append(event_id)
            splits.append(str(event["split"]))
        elif label == 0:
            group_id = f"nonstorm_{center:%Y%m%d}"
            event_ids.append("")
            time_groups.append(group_id)
            splits.append(year_split)
        else:
            group_id = f"uncertain_{center:%Y%m%d}"
            event_ids.append("")
            time_groups.append(group_id)
            splits.append(year_split)

    source_relative = data_path.relative_to(output_root).as_posix()
    sample_ids = [
        f"{station}_{timestamp:%Y%m%dT%H%M}_{int(window_config['length_hours'])}h"
        for timestamp in window_starts
    ]
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "station": station,
            "window_start": window_starts,
            "window_end": window_ends,
            "center_time": center_times,
            "source_data": source_relative,
            "start_offset": starts,
            "length": window_length,
            "label": labels,
            "label_name": label_names,
            "event_id": event_ids,
            "time_group_id": time_groups,
            "data_type": source_metadata["data_type"],
            "symh_min": symh_min,
            "distance_to_nearest_storm_minutes": nearest_distance,
            "quality_status": "good",
            "split": splits,
        }
    )

    partition_dir.mkdir(parents=True, exist_ok=True)
    temporary = partition_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, partition_path)
    return {
        "station": station,
        "year": year,
        "status": "CREATED",
        "windows": len(frame),
        "storm_windows": int((labels == 1).sum()),
        "nonstorm_windows": int((labels == 0).sum()),
        "uncertain_windows": int((labels == -1).sum()),
        "skipped_missing_windows": int((~complete).sum()),
        "partition_path": str(partition_path),
    }
