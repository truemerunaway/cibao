from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import atomic_save_npy, atomic_write_json, minutes_in_year, split_for_year

LOGGER = logging.getLogger(__name__)


@dataclass
class CoreRun:
    start: int
    end: int
    core_minutes: int


def parse_symh_year(
    path: Path,
    year: int,
    output_root: Path,
    missing_markers: Iterable[int],
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = output_root / "symh"
    values_path = output_dir / f"{year}_symh.npy"
    valid_path = output_dir / f"{year}_valid.npy"
    metadata_path = output_dir / f"{year}_metadata.json"

    if not overwrite and values_path.exists() and valid_path.exists() and metadata_path.exists():
        LOGGER.info("Skip existing SYM-H conversion: %s", year)
        import json

        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if not path.exists():
        raise FileNotFoundError(f"SYM-H file does not exist: {path}")

    rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if rows.shape[1] < 5:
        raise ValueError(f"Expected five SYM-H columns in {path}, found {rows.shape[1]}")

    expected = minutes_in_year(year)
    values = np.full(expected, np.nan, dtype=np.float32)
    counts = np.zeros(expected, dtype=np.uint8)

    row_year = rows[:, 0]
    day = rows[:, 1]
    hour = rows[:, 2]
    minute = rows[:, 3]
    symh = rows[:, 4].astype(np.float64)
    offsets = (day - 1) * 1440 + hour * 60 + minute

    valid_time = (
        (row_year == year)
        & (day >= 1)
        & (day <= (366 if expected == 527040 else 365))
        & (hour >= 0)
        & (hour < 24)
        & (minute >= 0)
        & (minute < 60)
        & (offsets >= 0)
        & (offsets < expected)
    )
    marker_array = np.asarray(list(missing_markers), dtype=np.float64)
    valid_value = np.isfinite(symh)
    if marker_array.size:
        valid_value &= ~np.isin(symh, marker_array)

    accepted = valid_time & valid_value
    accepted_offsets = offsets[accepted].astype(np.int64)
    values[accepted_offsets] = symh[accepted].astype(np.float32)
    np.add.at(counts, accepted_offsets, 1)
    valid = (counts == 1) & np.isfinite(values)

    atomic_save_npy(values_path, values)
    atomic_save_npy(valid_path, valid)
    metadata: dict[str, Any] = {
        "year": year,
        "source_file": str(path),
        "expected_minutes": expected,
        "actual_rows": int(rows.shape[0]),
        "valid_minutes": int(valid.sum()),
        "missing_minutes": int((counts == 0).sum()),
        "duplicate_minutes": int((counts > 1).sum()),
        "invalid_rows": int((~accepted).sum()),
        "values_path": str(values_path),
        "valid_path": str(valid_path),
        "status": "CONVERTED",
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def _find_core_runs(core: np.ndarray) -> list[CoreRun]:
    indices = np.flatnonzero(core)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks] + 1, [indices[-1] + 1]))
    return [
        CoreRun(int(start), int(end), int(end - start))
        for start, end in zip(starts, ends)
    ]


def _merge_core_runs(
    runs: list[CoreRun],
    valid: np.ndarray,
    merge_gap_minutes: int,
) -> list[CoreRun]:
    if not runs:
        return []
    invalid_prefix = np.concatenate(([0], np.cumsum(~valid, dtype=np.int64)))
    merged: list[CoreRun] = [runs[0]]

    for run in runs[1:]:
        current = merged[-1]
        gap = run.start - current.end
        gap_has_invalid = invalid_prefix[run.start] - invalid_prefix[current.end] > 0
        if gap <= merge_gap_minutes and not gap_has_invalid:
            current.end = run.end
            current.core_minutes += run.core_minutes
        else:
            merged.append(run)
    return merged


def _event_split(
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    split_config: dict[str, Any],
) -> str:
    start_split = split_for_year(event_start.year, split_config)
    end_inclusive = event_end - pd.Timedelta(minutes=1)
    end_split = split_for_year(end_inclusive.year, split_config)
    return start_split if start_split == end_split else "excluded"


def build_storm_events(
    output_root: Path,
    years: list[int],
    storm_threshold_nt: float,
    minimum_core_minutes: int,
    merge_gap_hours: float,
    split_config: dict[str, Any],
) -> pd.DataFrame:
    values_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    for year in years:
        values_parts.append(np.load(output_root / "symh" / f"{year}_symh.npy"))
        valid_parts.append(np.load(output_root / "symh" / f"{year}_valid.npy"))

    values = np.concatenate(values_parts)
    valid = np.concatenate(valid_parts).astype(bool, copy=False)
    core = valid & (values <= storm_threshold_nt)
    raw_runs = _find_core_runs(core)
    merged_runs = _merge_core_runs(
        raw_runs,
        valid,
        merge_gap_minutes=int(round(merge_gap_hours * 60)),
    )
    accepted_runs = [run for run in merged_runs if run.core_minutes >= minimum_core_minutes]
    origin = pd.Timestamp(year=years[0], month=1, day=1)

    event_columns = [
        "event_id",
        "event_start",
        "event_end",
        "symh_min",
        "core_minutes",
        "duration_minutes",
        "storm_threshold_nt",
        "merge_gap_hours",
        "split",
    ]
    records: list[dict[str, Any]] = []
    for run in accepted_runs:
        event_start = origin + pd.Timedelta(minutes=run.start)
        event_end = origin + pd.Timedelta(minutes=run.end)
        event_values = values[run.start : run.end]
        event_valid = valid[run.start : run.end]
        event_minimum = float(np.min(event_values[event_valid]))
        event_id = f"storm_{event_start:%Y%m%dT%H%M}"
        records.append(
            {
                "event_id": event_id,
                "event_start": event_start,
                "event_end": event_end,
                "symh_min": event_minimum,
                "core_minutes": run.core_minutes,
                "duration_minutes": run.end - run.start,
                "storm_threshold_nt": storm_threshold_nt,
                "merge_gap_hours": merge_gap_hours,
                "split": _event_split(event_start, event_end, split_config),
            }
        )

    events = pd.DataFrame.from_records(records, columns=event_columns)
    labels_dir = output_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(labels_dir / "storm_events.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    return events
