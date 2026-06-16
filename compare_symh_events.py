from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from geomag_dataset.utils import minutes_in_year
from training.event_postprocess import event_metrics, match_events


MISSING_MARKERS = {99999, 999999, -99999, -999999}


@dataclass
class Run:
    start: int
    end: int
    core_minutes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build reference storm events from a 1-minute SYM-H/SYM index file "
            "and compare them with predicted event CSV output."
        )
    )
    parser.add_argument("--symh", required=True, help="SYM-H file for the target year.")
    parser.add_argument(
        "--predicted-events",
        required=True,
        help="Model event CSV, for example inference/infer_csv/events.csv.",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--output-dir",
        default="inference/infer_csv/symh_compare",
        help="Directory for comparison CSV/JSON outputs.",
    )
    parser.add_argument("--threshold", type=float, default=-50.0)
    parser.add_argument("--minimum-core-minutes", type=int, default=60)
    parser.add_argument("--merge-gap-hours", type=float, default=12.0)
    parser.add_argument("--match-tolerance-hours", type=float, default=0.0)
    return parser.parse_args()


def _numeric_rows(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip().replace(",", " ")
        if not text or text.startswith("#"):
            continue
        values: list[float] = []
        ok = True
        for token in text.split():
            try:
                values.append(float(token))
            except ValueError:
                ok = False
                break
        if ok and len(values) >= 5:
            rows.append(values)
    if not rows:
        raise ValueError(
            "No numeric rows found. Expected columns: year doy hour minute symh_nt."
        )
    width = max(len(row) for row in rows)
    padded = [row + [np.nan] * (width - len(row)) for row in rows]
    return np.asarray(padded, dtype=float)


def load_symh_series(
    path: str | Path,
    year: int,
    missing_markers: Iterable[int] = MISSING_MARKERS,
) -> pd.DataFrame:
    path = Path(path)
    rows = _numeric_rows(path)
    first = rows[:, 0].astype(int)
    if not np.all((1900 <= first) & (first <= 2100)):
        raise ValueError(
            "Unsupported SYM-H format. Convert it to whitespace columns: "
            "year day_of_year hour minute symh_nt."
        )
    row_year = rows[:, 0].astype(int)
    doy = rows[:, 1].astype(int)
    hour = rows[:, 2].astype(int)
    minute = rows[:, 3].astype(int)
    symh = rows[:, 4].astype(float)

    expected = minutes_in_year(year)
    offsets = (doy - 1) * 1440 + hour * 60 + minute
    valid_time = (
        (row_year == year)
        & (doy >= 1)
        & (doy <= (366 if expected == 527040 else 365))
        & (hour >= 0)
        & (hour < 24)
        & (minute >= 0)
        & (minute < 60)
        & (offsets >= 0)
        & (offsets < expected)
    )
    marker_array = np.asarray(list(missing_markers), dtype=float)
    valid_value = np.isfinite(symh)
    if marker_array.size:
        valid_value &= ~np.isin(symh, marker_array)

    values = np.full(expected, np.nan, dtype=np.float32)
    counts = np.zeros(expected, dtype=np.uint8)
    accepted = valid_time & valid_value
    accepted_offsets = offsets[accepted].astype(np.int64)
    values[accepted_offsets] = symh[accepted].astype(np.float32)
    np.add.at(counts, accepted_offsets, 1)
    valid = (counts == 1) & np.isfinite(values)
    times = pd.date_range(
        pd.Timestamp(year=year, month=1, day=1),
        periods=expected,
        freq="1min",
    )
    return pd.DataFrame(
        {
            "timestamp": times,
            "symh_nt": values,
            "valid": valid,
            "below_threshold": valid & (values <= -50.0),
        }
    )


def _find_runs(mask: np.ndarray) -> list[Run]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks] + 1, [indices[-1] + 1]))
    return [
        Run(int(start), int(end), int(end - start))
        for start, end in zip(starts, ends)
    ]


def _merge_runs(runs: list[Run], valid: np.ndarray, merge_gap_minutes: int) -> list[Run]:
    if not runs:
        return []
    invalid_prefix = np.concatenate(([0], np.cumsum(~valid, dtype=np.int64)))
    merged = [runs[0]]
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


def build_reference_events(
    symh: pd.DataFrame,
    year: int,
    threshold: float,
    minimum_core_minutes: int,
    merge_gap_hours: float,
) -> pd.DataFrame:
    values = symh["symh_nt"].to_numpy(dtype=float)
    valid = symh["valid"].to_numpy(dtype=bool)
    core = valid & (values <= float(threshold))
    raw_runs = _find_runs(core)
    qualified = [run for run in raw_runs if run.core_minutes >= minimum_core_minutes]
    merged = _merge_runs(
        qualified,
        valid,
        merge_gap_minutes=int(round(float(merge_gap_hours) * 60)),
    )
    origin = pd.Timestamp(year=year, month=1, day=1)
    records: list[dict[str, object]] = []
    for run in merged:
        event_values = values[run.start : run.end]
        event_valid = valid[run.start : run.end]
        start = origin + pd.Timedelta(minutes=run.start)
        end = origin + pd.Timedelta(minutes=run.end)
        records.append(
            {
                "event_id": f"symh_{start:%Y%m%dT%H%M}",
                "event_start": start,
                "event_end": end,
                "symh_min": float(np.nanmin(event_values[event_valid])),
                "core_minutes": int(run.core_minutes),
                "duration_minutes": int(run.end - run.start),
                "threshold_nt": float(threshold),
                "merge_gap_hours": float(merge_gap_hours),
            }
        )
    return pd.DataFrame.from_records(records)


def _read_predicted_events(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "event_start",
                "event_end",
                "maximum_probability",
                "supporting_stations",
            ]
        )
    frame["event_start"] = pd.to_datetime(frame["event_start"])
    frame["event_end"] = pd.to_datetime(frame["event_end"])
    return frame


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symh = load_symh_series(args.symh, args.year)
    references = build_reference_events(
        symh,
        year=args.year,
        threshold=args.threshold,
        minimum_core_minutes=args.minimum_core_minutes,
        merge_gap_hours=args.merge_gap_hours,
    )
    predicted = _read_predicted_events(args.predicted_events)

    metrics, matches = event_metrics(
        predicted,
        references,
        match_tolerance_hours=args.match_tolerance_hours,
    )
    matches, matched_predictions, matched_references = match_events(
        predicted,
        references,
        match_tolerance_hours=args.match_tolerance_hours,
    )
    missed = references.loc[
        ~references.index.isin(matched_references)
    ].reset_index(drop=True)
    false_alarms = predicted.loc[
        ~predicted.index.isin(matched_predictions)
    ].reset_index(drop=True)

    symh.to_csv(output_dir / "symh_timeseries.csv", index=False)
    references.to_csv(output_dir / "symh_reference_events.csv", index=False)
    predicted.to_csv(output_dir / "model_predicted_events.csv", index=False)
    matches.to_csv(output_dir / "matched_events.csv", index=False)
    missed.to_csv(output_dir / "missed_symh_events.csv", index=False)
    false_alarms.to_csv(output_dir / "false_alarm_model_events.csv", index=False)
    _write_json(output_dir / "summary.json", metrics)

    print(f"Reference events: {metrics['reference_event_count']}")
    print(f"Detected events: {metrics['detected_event_count']}")
    print(f"Missed events: {metrics['missed_event_count']}")
    print(f"Predicted events: {metrics['predicted_event_count']}")
    print(f"False alarms: {metrics['false_alarm_event_count']}")
    print(f"Detection rate: {metrics['event_detection_rate']}")
    print(f"Event precision: {metrics['event_precision']}")
    print(f"Event F1: {metrics['event_f1']}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
