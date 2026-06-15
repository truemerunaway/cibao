from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


INDEX_COLUMNS = [
    "sample_id",
    "station",
    "window_start",
    "window_end",
    "center_time",
    "source_data",
    "start_offset",
    "length",
    "label",
    "event_id",
    "symh_min",
]


@dataclass
class FoldAssignment:
    name: str
    train: pd.DataFrame
    validation: pd.DataFrame
    audit: dict[str, Any]


def load_events(dataset_root: str | Path) -> pd.DataFrame:
    path = Path(dataset_root) / "labels" / "storm_events.csv"
    if not path.exists():
        raise FileNotFoundError(f"Storm event table not found: {path}")
    events = pd.read_csv(path, parse_dates=["event_start", "event_end"])
    required = {"event_id", "event_start", "event_end"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Storm event table is missing columns: {sorted(missing)}")
    return events


def load_window_index(
    dataset_root: str | Path,
    years: Iterable[int],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    root = Path(dataset_root) / "windows"
    selected_columns = columns or INDEX_COLUMNS
    frames: list[pd.DataFrame] = []
    for year in sorted({int(value) for value in years}):
        partition = root / f"year={year}"
        if not partition.exists():
            continue
        for path in sorted(partition.glob("*.parquet")):
            frame = pd.read_parquet(path, columns=selected_columns)
            frame["partition_year"] = year
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No Parquet window partitions found below {root}")
    result = pd.concat(frames, ignore_index=True)
    for column in ("window_start", "window_end", "center_time"):
        result[column] = pd.to_datetime(result[column])
    result["station"] = result["station"].astype(str)
    result["event_id"] = result["event_id"].fillna("").astype(str)
    return result


def boundary_event_ids(
    events: pd.DataFrame,
    development_years: Iterable[int],
    final_test_years: Iterable[int],
) -> set[str]:
    development_end = max(int(year) for year in development_years)
    test_start = min(int(year) for year in final_test_years)
    if development_end + 1 != test_start:
        raise ValueError("Development and final-test years must meet at one boundary.")
    boundary = pd.Timestamp(year=test_start, month=1, day=1)
    mask = (events["event_start"] < boundary) & (events["event_end"] > boundary)
    return set(events.loc[mask, "event_id"].astype(str))


def annotate_groups(index: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    frame = index.copy()
    event_years = events.set_index("event_id")["event_start"].dt.year.to_dict()
    storm = frame["label"].eq(1)
    missing_events = sorted(
        set(frame.loc[storm, "event_id"]) - set(event_years)
    )
    if missing_events:
        raise ValueError(
            f"Storm windows reference unknown event_id values: {missing_events[:5]}"
        )

    center_dates = frame["center_time"].dt.strftime("%Y%m%d")
    frame["group_id"] = np.where(
        storm,
        frame["event_id"],
        np.where(
            frame["label"].eq(0),
            "nonstorm_" + center_dates,
            "uncertain_" + center_dates,
        ),
    )
    frame["group_year"] = frame["center_time"].dt.year.astype(int)
    frame.loc[storm, "group_year"] = (
        frame.loc[storm, "event_id"].map(event_years).astype(int)
    )
    return frame


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "Storm": int(frame["label"].eq(1).sum()),
        "NonStorm": int(frame["label"].eq(0).sum()),
        "Uncertain": int(frame["label"].eq(-1).sum()),
    }


def _station_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(station): int(count)
        for station, count in frame.groupby("station", sort=True).size().items()
    }


def make_fold_assignment(
    annotated_index: pd.DataFrame,
    events: pd.DataFrame,
    development_years: Iterable[int],
    final_test_years: Iterable[int],
    validation_years: Iterable[int],
    name: str,
) -> FoldAssignment:
    development_years = sorted({int(year) for year in development_years})
    validation_years = sorted({int(year) for year in validation_years})
    train_years = sorted(set(development_years) - set(validation_years))
    excluded_events = boundary_event_ids(events, development_years, final_test_years)

    development = annotated_index["group_year"].isin(development_years)
    boundary_excluded = annotated_index["event_id"].isin(excluded_events)
    eligible = annotated_index.loc[development & ~boundary_excluded].copy()
    train_all = eligible.loc[eligible["group_year"].isin(train_years)].copy()
    val_all = eligible.loc[eligible["group_year"].isin(validation_years)].copy()

    train_event_ids = set(train_all.loc[train_all["label"].eq(1), "event_id"])
    val_event_ids = set(val_all.loc[val_all["label"].eq(1), "event_id"])
    train_nonstorm_ids = set(train_all.loc[train_all["label"].eq(0), "group_id"])
    val_nonstorm_ids = set(val_all.loc[val_all["label"].eq(0), "group_id"])
    train_uncertain_ids = set(train_all.loc[train_all["label"].eq(-1), "group_id"])
    val_uncertain_ids = set(val_all.loc[val_all["label"].eq(-1), "group_id"])
    event_leaks = sorted(train_event_ids & val_event_ids)
    nonstorm_leaks = sorted(train_nonstorm_ids & val_nonstorm_ids)
    uncertain_leaks = sorted(train_uncertain_ids & val_uncertain_ids)
    if event_leaks or nonstorm_leaks or uncertain_leaks:
        raise AssertionError(
            f"Leakage detected in {name}: events={len(event_leaks)}, "
            f"nonstorm_dates={len(nonstorm_leaks)}, "
            f"uncertain_dates={len(uncertain_leaks)}"
        )

    boundary_rows = annotated_index.loc[boundary_excluded]
    audit = {
        "fold": name,
        "training_years": train_years,
        "validation_years": validation_years,
        "train_label_counts": _label_counts(train_all),
        "validation_label_counts": _label_counts(val_all),
        "train_event_count": len(train_event_ids),
        "validation_event_count": len(val_event_ids),
        "train_station_window_counts": _station_counts(train_all),
        "validation_station_window_counts": _station_counts(val_all),
        "event_id_leakage_count": len(event_leaks),
        "time_group_id_leakage_count": len(nonstorm_leaks),
        "uncertain_time_group_id_leakage_count": len(uncertain_leaks),
        "excluded_boundary_event_ids": sorted(excluded_events),
        "excluded_boundary_window_count": int(len(boundary_rows)),
    }
    labels = {0, 1}
    return FoldAssignment(
        name=name,
        train=train_all.loc[train_all["label"].isin(labels)].reset_index(drop=True),
        validation=val_all.loc[val_all["label"].isin(labels)].reset_index(drop=True),
        audit=audit,
    )


def make_final_partitions(
    annotated_index: pd.DataFrame,
    events: pd.DataFrame,
    development_years: Iterable[int],
    final_test_years: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    development_years = sorted({int(year) for year in development_years})
    final_test_years = sorted({int(year) for year in final_test_years})
    excluded_events = boundary_event_ids(events, development_years, final_test_years)
    boundary_excluded = annotated_index["event_id"].isin(excluded_events)
    labels = annotated_index["label"].isin([0, 1])
    final_train = annotated_index.loc[
        annotated_index["group_year"].isin(development_years)
        & ~boundary_excluded
        & labels
    ].copy()
    final_test = annotated_index.loc[
        annotated_index["group_year"].isin(final_test_years)
        & ~boundary_excluded
        & labels
    ].copy()
    audit = {
        "development_years": development_years,
        "final_test_years": final_test_years,
        "excluded_boundary_event_ids": sorted(excluded_events),
        "excluded_boundary_window_count": int(boundary_excluded.sum()),
        "final_train_label_counts": _label_counts(final_train),
        "final_test_label_counts": _label_counts(final_test),
        "final_train_station_window_counts": _station_counts(final_train),
        "final_test_station_window_counts": _station_counts(final_test),
    }
    return (
        final_train.reset_index(drop=True),
        final_test.reset_index(drop=True),
        audit,
    )
