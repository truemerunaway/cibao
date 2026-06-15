from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from training.config import load_training_config
from training.event_postprocess import predictions_to_events
from training.inference import (
    load_iaga_series,
    load_inference_artifacts,
    load_npy_series,
    score_continuous_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect storm events in one or more long XYZ time series."
    )
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Input as STATION=path. IAGA TXT and NumPy arrays are supported.",
    )
    parser.add_argument(
        "--start-time",
        help="Required for NumPy inputs; shared UTC start time at one-minute cadence.",
    )
    parser.add_argument("--output", required=True, help="Output event CSV path.")
    parser.add_argument(
        "--window-output",
        help="Optional CSV path for hourly station scores.",
    )
    return parser.parse_args()


def _parse_series(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--series must use STATION=path.")
    station, path = value.split("=", maxsplit=1)
    return station.strip().upper(), Path(path).expanduser()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    output_root = Path(config["data"]["output_root"])
    final_dir = output_root / "final"
    parameters = json.loads(
        (output_root / "selected_postprocess.json").read_text(encoding="utf-8")
    )
    model, normalizer, device, _ = load_inference_artifacts(
        final_dir / "final_model.pt",
        final_dir / "normalization.json",
        config["runtime"]["device"],
    )
    score_frames: list[pd.DataFrame] = []
    for station_hint, path in map(_parse_series, args.series):
        if path.suffix.lower() == ".npy":
            if not args.start_time:
                raise ValueError("--start-time is required for NumPy inputs.")
            station, times, values = load_npy_series(
                path,
                station_hint,
                args.start_time,
            )
        else:
            station, times, values = load_iaga_series(path)
            if station_hint and station_hint != station:
                raise ValueError(
                    f"Station hint {station_hint} does not match IAGA code {station}."
                )
        score_frames.append(
            score_continuous_series(
                model,
                normalizer,
                station,
                times,
                values,
                device,
                baseline_minutes=int(config["features"]["baseline_minutes"]),
                window_minutes=360,
                stride_minutes=60,
                batch_size=int(config["evaluation"]["batch_size"]),
            )
        )
    scores = pd.concat(score_frames, ignore_index=True)
    events, _, _ = predictions_to_events(
        scores,
        threshold=float(parameters["threshold"]),
        smoothing_points=int(parameters["smoothing_points"]),
        merge_gap_hours=float(parameters["merge_gap_hours"]),
        fusion=str(parameters["fusion"]),
        min_stations=int(parameters["min_stations"]),
        stride_hours=float(parameters["stride_hours"]),
        boundary_half_width_hours=float(parameters["boundary_half_width_hours"]),
    )
    if "supporting_stations" in events:
        events["supporting_stations"] = events["supporting_stations"].map(
            lambda values: ";".join(values)
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output, index=False)
    if args.window_output:
        window_output = Path(args.window_output)
        window_output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(window_output, index=False)


if __name__ == "__main__":
    main()
