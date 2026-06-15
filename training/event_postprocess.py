from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def smooth_station_scores(
    predictions: pd.DataFrame,
    smoothing_points: int,
    stride_hours: float = 1.0,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["center_time"] = pd.to_datetime(frame["center_time"])
    frame = frame.sort_values(["station", "center_time"], kind="stable")
    pieces: list[pd.DataFrame] = []
    max_gap = pd.Timedelta(hours=float(stride_hours) * 1.5)
    for _, station_frame in frame.groupby("station", sort=False):
        station_frame = station_frame.copy()
        segment = station_frame["center_time"].diff().gt(max_gap).cumsum()
        station_frame["smoothed_score"] = (
            station_frame.groupby(segment, sort=False)["score"]
            .transform(
                lambda values: values.rolling(
                    window=int(smoothing_points),
                    center=True,
                    min_periods=1,
                ).mean()
            )
            .astype(float)
        )
        pieces.append(station_frame)
    if not pieces:
        return frame.assign(smoothed_score=pd.Series(dtype=float))
    return pd.concat(pieces, ignore_index=True)


def fuse_station_scores(
    smoothed: pd.DataFrame,
    threshold: float,
    min_stations: int = 3,
    method: str = "median",
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for center_time, frame in smoothed.groupby("center_time", sort=True):
        if len(frame) < int(min_stations):
            continue
        scores = frame["smoothed_score"].to_numpy(dtype=float)
        if method == "median":
            fused = float(np.median(scores))
        elif method == "mean":
            fused = float(np.mean(scores))
        elif method == "top3_mean":
            fused = float(np.mean(np.sort(scores)[-min(3, len(scores)) :]))
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        supporting = sorted(
            frame.loc[frame["smoothed_score"].ge(threshold), "station"].astype(str)
        )
        records.append(
            {
                "center_time": center_time,
                "score": fused,
                "station_count": int(len(frame)),
                "supporting_stations": supporting,
            }
        )
    return pd.DataFrame.from_records(records)


def _positive_runs(
    timeline: pd.DataFrame,
    stride_hours: float,
) -> list[pd.DataFrame]:
    positive = timeline.loc[timeline["positive"]].sort_values("center_time").copy()
    if positive.empty:
        return []
    max_gap = pd.Timedelta(hours=float(stride_hours) * 1.5)
    segment = positive["center_time"].diff().gt(max_gap).cumsum()
    return [frame.copy() for _, frame in positive.groupby(segment, sort=False)]


def timeline_to_events(
    timeline: pd.DataFrame,
    threshold: float,
    merge_gap_hours: float,
    stride_hours: float = 1.0,
    boundary_half_width_hours: float = 0.5,
) -> pd.DataFrame:
    if timeline.empty:
        return pd.DataFrame(
            columns=[
                "event_start",
                "event_end",
                "maximum_probability",
                "supporting_stations",
            ]
        )
    frame = timeline.copy()
    frame["center_time"] = pd.to_datetime(frame["center_time"])
    frame["positive"] = frame["score"].ge(float(threshold))
    runs = _positive_runs(frame, stride_hours)
    half_width = pd.Timedelta(hours=float(boundary_half_width_hours))
    candidates: list[dict[str, Any]] = []
    for run in runs:
        supports: set[str] = set()
        if "supporting_stations" in run:
            for values in run["supporting_stations"]:
                supports.update(str(value) for value in values)
        candidates.append(
            {
                "event_start": run["center_time"].iloc[0] - half_width,
                "event_end": run["center_time"].iloc[-1] + half_width,
                "maximum_probability": float(run["score"].max()),
                "supporting_stations": supports,
            }
        )

    merged: list[dict[str, Any]] = []
    allowed_gap = pd.Timedelta(hours=float(merge_gap_hours))
    for candidate in candidates:
        if merged and candidate["event_start"] - merged[-1]["event_end"] <= allowed_gap:
            merged[-1]["event_end"] = max(
                merged[-1]["event_end"], candidate["event_end"]
            )
            merged[-1]["maximum_probability"] = max(
                merged[-1]["maximum_probability"],
                candidate["maximum_probability"],
            )
            merged[-1]["supporting_stations"].update(
                candidate["supporting_stations"]
            )
        else:
            merged.append(candidate)
    for event in merged:
        event["supporting_stations"] = sorted(event["supporting_stations"])
    return pd.DataFrame.from_records(merged)


def predictions_to_events(
    predictions: pd.DataFrame,
    threshold: float,
    smoothing_points: int,
    merge_gap_hours: float,
    fusion: str = "median",
    min_stations: int = 3,
    stride_hours: float = 1.0,
    boundary_half_width_hours: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_stations = int(predictions["station"].nunique()) if not predictions.empty else 0
    effective_min_stations = min(int(min_stations), max(available_stations, 1))
    smoothed = smooth_station_scores(
        predictions,
        smoothing_points=smoothing_points,
        stride_hours=stride_hours,
    )
    fused = fuse_station_scores(
        smoothed,
        threshold=threshold,
        min_stations=effective_min_stations,
        method=fusion,
    )
    events = timeline_to_events(
        fused,
        threshold=threshold,
        merge_gap_hours=merge_gap_hours,
        stride_hours=stride_hours,
        boundary_half_width_hours=boundary_half_width_hours,
    )
    return events, fused, smoothed


def filter_reference_events(
    events: pd.DataFrame,
    years: Iterable[int],
    excluded_event_ids: Iterable[str] = (),
) -> pd.DataFrame:
    years = {int(year) for year in years}
    excluded = set(str(value) for value in excluded_event_ids)
    frame = events.copy()
    frame["event_start"] = pd.to_datetime(frame["event_start"])
    frame["event_end"] = pd.to_datetime(frame["event_end"])
    return frame.loc[
        frame["event_start"].dt.year.isin(years)
        & ~frame["event_id"].astype(str).isin(excluded)
    ].reset_index(drop=True)


def _overlap_seconds(
    predicted: pd.Series,
    reference: pd.Series,
    tolerance: pd.Timedelta,
) -> float:
    start = max(
        predicted["event_start"] - tolerance,
        reference["event_start"],
    )
    end = min(
        predicted["event_end"] + tolerance,
        reference["event_end"],
    )
    return max(0.0, (end - start).total_seconds())


def match_events(
    predicted_events: pd.DataFrame,
    reference_events: pd.DataFrame,
    match_tolerance_hours: float = 0,
) -> tuple[pd.DataFrame, set[int], set[int]]:
    predicted = predicted_events.copy()
    reference = reference_events.copy()
    for frame in (predicted, reference):
        frame["event_start"] = pd.to_datetime(frame["event_start"])
        frame["event_end"] = pd.to_datetime(frame["event_end"])
    tolerance = pd.Timedelta(hours=float(match_tolerance_hours))
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in predicted.iterrows():
        for ref_index, ref in reference.iterrows():
            overlap = _overlap_seconds(pred, ref, tolerance)
            if overlap > 0:
                candidates.append((overlap, int(pred_index), int(ref_index)))
    candidates.sort(reverse=True)
    matched_predictions: set[int] = set()
    matched_references: set[int] = set()
    records: list[dict[str, Any]] = []
    for overlap, pred_index, ref_index in candidates:
        if pred_index in matched_predictions or ref_index in matched_references:
            continue
        matched_predictions.add(pred_index)
        matched_references.add(ref_index)
        pred = predicted.loc[pred_index]
        ref = reference.loc[ref_index]
        record = {
            "predicted_index": pred_index,
            "reference_index": ref_index,
            "event_id": str(ref.get("event_id", "")),
            "overlap_hours": overlap / 3600.0,
            "start_error_hours": (
                pred["event_start"] - ref["event_start"]
            ).total_seconds()
            / 3600.0,
            "end_error_hours": (
                pred["event_end"] - ref["event_end"]
            ).total_seconds()
            / 3600.0,
            "duration_error_hours": (
                (pred["event_end"] - pred["event_start"])
                - (ref["event_end"] - ref["event_start"])
            ).total_seconds()
            / 3600.0,
        }
        for column in ("symh_min", "duration_minutes"):
            if column in ref:
                record[column] = float(ref[column])
        records.append(record)
    return pd.DataFrame.from_records(records), matched_predictions, matched_references


def _error_summary(matches: pd.DataFrame, column: str) -> dict[str, float | None]:
    if matches.empty:
        return {"mean": None, "median": None, "mean_absolute": None}
    values = matches[column].to_numpy(dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_absolute": float(np.abs(values).mean()),
    }


def event_metrics(
    predicted_events: pd.DataFrame,
    reference_events: pd.DataFrame,
    match_tolerance_hours: float = 0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    matches, matched_predictions, matched_references = match_events(
        predicted_events,
        reference_events,
        match_tolerance_hours=match_tolerance_hours,
    )
    reference_count = len(reference_events)
    predicted_count = len(predicted_events)
    detected = len(matched_references)
    false_alarms = predicted_count - len(matched_predictions)
    detection_rate = detected / reference_count if reference_count else np.nan
    event_precision = (
        len(matched_predictions) / predicted_count if predicted_count else np.nan
    )
    if np.isfinite(detection_rate) and np.isfinite(event_precision):
        denominator = detection_rate + event_precision
        event_f1 = (
            2 * detection_rate * event_precision / denominator
            if denominator
            else 0.0
        )
    else:
        event_f1 = np.nan

    unmatched = predicted_events.loc[
        ~predicted_events.index.isin(matched_predictions)
    ].copy()
    false_alarms_by_year: dict[str, int] = {}
    if not unmatched.empty:
        years = pd.to_datetime(unmatched["event_start"]).dt.year
        false_alarms_by_year = {
            str(year): int(count)
            for year, count in years.value_counts().sort_index().items()
        }
    reference_years = pd.to_datetime(reference_events["event_start"]).dt.year
    detected_by_reference = set(matched_references)
    detection_by_year: dict[str, dict[str, float | int]] = {}
    for year in sorted(reference_years.unique()):
        year_indices = set(reference_events.index[reference_years.eq(year)])
        year_detected = len(year_indices & detected_by_reference)
        year_total = len(year_indices)
        detection_by_year[str(int(year))] = {
            "reference_event_count": int(year_total),
            "detected_event_count": int(year_detected),
            "missed_event_count": int(year_total - year_detected),
            "detection_rate": float(year_detected / year_total)
            if year_total
            else 0.0,
        }
    metrics = {
        "reference_event_count": int(reference_count),
        "detected_event_count": int(detected),
        "missed_event_count": int(reference_count - detected),
        "event_detection_rate": float(detection_rate)
        if np.isfinite(detection_rate)
        else None,
        "predicted_event_count": int(predicted_count),
        "false_alarm_event_count": int(false_alarms),
        "event_precision": float(event_precision)
        if np.isfinite(event_precision)
        else None,
        "event_f1": float(event_f1) if np.isfinite(event_f1) else None,
        "detection_by_year": detection_by_year,
        "false_alarms_by_year": false_alarms_by_year,
        "start_error_hours": _error_summary(matches, "start_error_hours"),
        "end_error_hours": _error_summary(matches, "end_error_hours"),
        "duration_error_hours": _error_summary(matches, "duration_error_hours"),
    }
    return metrics, matches


def event_stratified_report(matches: pd.DataFrame, references: pd.DataFrame) -> dict[str, Any]:
    if references.empty:
        return {}
    frame = references.reset_index(drop=True).copy()
    frame["detected"] = frame.index.isin(set(matches.get("reference_index", [])))
    result: dict[str, Any] = {}
    if "symh_min" in frame:
        frame["strength_bin"] = pd.cut(
            frame["symh_min"],
            bins=[-np.inf, -150, -100, -50],
            labels=["<=-150", "(-150,-100]", "(-100,-50]"],
        )
        result["by_symh_min"] = {
            str(group): {
                "reference_event_count": int(len(values)),
                "detected_event_count": int(values["detected"].sum()),
                "detection_rate": float(values["detected"].mean()),
            }
            for group, values in frame.groupby("strength_bin", observed=True)
        }
    if "duration_minutes" in frame:
        hours = frame["duration_minutes"].astype(float) / 60.0
        frame["duration_bin"] = pd.cut(
            hours,
            bins=[0, 6, 12, 24, np.inf],
            labels=["<=6h", "(6,12]h", "(12,24]h", ">24h"],
        )
        result["by_duration"] = {
            str(group): {
                "reference_event_count": int(len(values)),
                "detected_event_count": int(values["detected"].sum()),
                "detection_rate": float(values["detected"].mean()),
            }
            for group, values in frame.groupby("duration_bin", observed=True)
        }
    return result
