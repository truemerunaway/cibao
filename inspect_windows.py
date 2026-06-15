from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geomag_dataset.features import raw_delta, raw_diff


LABEL_NAMES = {-1: "Uncertain", 0: "NonStorm", 1: "Storm"}
COLORS = {"X": "#1f77b4", "Y": "#d62728", "Z": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw quality-control figures from the generated window index."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=10,
        help="Maximum figures for each split and label.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to inspect.",
    )
    parser.add_argument(
        "--labels",
        default="-1,0,1",
        help="Comma-separated labels to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to <output_root>/figures_check.",
    )
    return parser.parse_args()


def load_index(index_root: Path) -> pd.DataFrame:
    columns = [
        "sample_id",
        "station",
        "window_start",
        "window_end",
        "source_data",
        "start_offset",
        "length",
        "label",
        "label_name",
        "event_id",
        "time_group_id",
        "data_type",
        "symh_min",
        "split",
    ]
    frame = pd.read_parquet(index_root, columns=columns)
    frame["window_start"] = pd.to_datetime(frame["window_start"])
    frame["window_end"] = pd.to_datetime(frame["window_end"])
    return frame


def select_diverse_samples(
    frame: pd.DataFrame,
    split: str,
    label: int,
    count: int,
    seed: int,
) -> pd.DataFrame:
    subset = frame[(frame["split"] == split) & (frame["label"] == label)].copy()
    if subset.empty:
        return subset

    # One random station/window per event or date group before final sampling.
    diverse = (
        subset.sample(frac=1, random_state=seed)
        .drop_duplicates("time_group_id")
        .reset_index(drop=True)
    )
    return diverse.sample(
        n=min(count, len(diverse)),
        random_state=seed + label + 2,
    ).sort_values("window_start")


def load_window(
    dataset_root: Path,
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    annual = np.load(dataset_root / str(row["source_data"]), mmap_mode="r")
    start = int(row["start_offset"])
    length = int(row["length"])
    raw = np.asarray(annual[start : start + length], dtype=np.float32).T

    year = pd.Timestamp(row["window_start"]).year
    symh_annual = np.load(
        dataset_root / "symh" / f"{year}_symh.npy",
        mmap_mode="r",
    )
    symh = np.asarray(symh_annual[start : start + length], dtype=np.float32)
    return raw, symh


def draw_sample(
    dataset_root: Path,
    output_path: Path,
    row: pd.Series,
    baseline_minutes: int,
    storm_threshold: float,
    nonstorm_threshold: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for inspection figures. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    raw, symh = load_window(dataset_root, row)
    delta = raw_delta(raw, baseline_minutes)
    difference = raw_diff(raw)
    hours = np.arange(raw.shape[1], dtype=np.float32) / 60.0

    figure = plt.figure(figsize=(14, 11), constrained_layout=True)
    grid = figure.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.2])
    components = ("X", "Y", "Z")

    for channel, component in enumerate(components):
        delta_axis = figure.add_subplot(grid[channel, 0])
        diff_axis = figure.add_subplot(grid[channel, 1], sharex=delta_axis)
        color = COLORS[component]

        delta_axis.plot(hours, delta[channel], color=color, linewidth=0.9)
        delta_axis.axhline(0, color="#777777", linewidth=0.6)
        delta_axis.set_ylabel(f"delta {component} (nT)")
        delta_axis.grid(alpha=0.2)

        diff_axis.plot(hours, difference[channel], color=color, linewidth=0.8)
        diff_axis.axhline(0, color="#777777", linewidth=0.6)
        diff_axis.set_ylabel(f"diff {component} (nT/min)")
        diff_axis.grid(alpha=0.2)

    symh_axis = figure.add_subplot(grid[3, :])
    symh_axis.plot(hours, symh, color="#111111", linewidth=1.0, label="SYM-H")
    symh_axis.axhline(
        storm_threshold,
        color="#b2182b",
        linestyle="--",
        linewidth=1.0,
        label=f"storm threshold ({storm_threshold:g} nT)",
    )
    symh_axis.axhline(
        nonstorm_threshold,
        color="#ef8a62",
        linestyle=":",
        linewidth=1.0,
        label=f"nonstorm threshold ({nonstorm_threshold:g} nT)",
    )
    symh_axis.set_xlabel("Hours from window start")
    symh_axis.set_ylabel("SYM-H (nT)")
    symh_axis.grid(alpha=0.2)
    symh_axis.legend(loc="best")

    event_id = str(row["event_id"]) if pd.notna(row["event_id"]) else ""
    title = (
        f"{row['sample_id']} | {LABEL_NAMES[int(row['label'])]} | "
        f"split={row['split']} | type={row['data_type']}\n"
        f"{row['window_start']} to {row['window_end']} | "
        f"event={event_id or '-'} | SYM-H min={float(row['symh_min']):.1f} nT"
    )
    figure.suptitle(title, fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    from geomag_dataset.config import load_config

    args = parse_args()
    config = load_config(Path(args.config))
    dataset_root = Path(config["paths"]["output_root"])
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else dataset_root / "figures_check"
    )
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    labels = [int(item.strip()) for item in args.labels.split(",") if item.strip()]
    frame = load_index(dataset_root / "windows")
    manifest: list[dict[str, Any]] = []

    for split in splits:
        for label in labels:
            selected = select_diverse_samples(
                frame,
                split,
                label,
                args.samples_per_group,
                args.seed,
            )
            label_name = LABEL_NAMES[label]
            for _, row in selected.iterrows():
                output_path = (
                    output_root
                    / split
                    / label_name
                    / f"{row['sample_id']}.png"
                )
                draw_sample(
                    dataset_root,
                    output_path,
                    row,
                    int(config["window"]["baseline_minutes"]),
                    float(config["symh"]["storm_threshold_nt"]),
                    float(config["symh"]["nonstorm_threshold_nt"]),
                )
                manifest.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": split,
                        "label": label,
                        "label_name": label_name,
                        "station": row["station"],
                        "window_start": row["window_start"],
                        "event_id": row["event_id"],
                        "figure_path": str(output_path),
                    }
                )

    manifest_frame = pd.DataFrame.from_records(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_frame.to_csv(output_root / "manifest.csv", index=False)
    summary = {
        "figures": len(manifest_frame),
        "splits": splits,
        "labels": labels,
        "samples_per_group": args.samples_per_group,
        "output_root": str(output_root),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
