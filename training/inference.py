from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from geomag_dataset.features import raw_delta
from geomag_dataset.iaga import component_columns, parse_iaga_header

from .normalization import StationRobustNormalizer
from .trainer import require_torch, resolve_device

LOGGER = logging.getLogger(__name__)


DEFAULT_MISSING_MARKERS = [
    99999,
    99999.0,
    999999,
    999999.0,
    -99999,
    -99999.0,
    -999999,
    -999999.0,
]


def load_model_checkpoint(path: str | Path, device):
    from .models import CNN1D

    torch = require_torch()
    try:
        checkpoint = torch.load(
            Path(path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(Path(path), map_location=device)
    model_config = checkpoint.get(
        "model_config",
        {"input_channels": 3, "dropout": 0.3},
    )
    model = CNN1D(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, model_config


def load_iaga_series(
    path: str | Path,
    missing_markers: Iterable[float] = DEFAULT_MISSING_MARKERS,
) -> tuple[str, pd.DatetimeIndex, np.ndarray]:
    source = Path(path)
    header = parse_iaga_header(source)
    mapping = component_columns(header, ["X", "Y", "Z"])
    selected = ["DATE", "TIME", *mapping.values()]
    frame = pd.read_csv(
        source,
        sep=r"\s+",
        skiprows=header.data_start_line,
        names=header.columns,
        usecols=selected,
        dtype=str,
        engine="c",
    )
    timestamps = pd.to_datetime(
        frame["DATE"].astype(str) + " " + frame["TIME"].astype(str),
        errors="coerce",
    )
    values = np.column_stack(
        [
            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
            for column in mapping.values()
        ]
    )
    marker_array = np.asarray(list(missing_markers), dtype=np.float32)
    if marker_array.size:
        values[np.isin(values, marker_array)] = np.nan
    valid_time = timestamps.notna().to_numpy()
    timestamps = pd.DatetimeIndex(timestamps[valid_time])
    values = values[valid_time]
    order = np.argsort(timestamps.values)
    timestamps = timestamps[order]
    values = values[order]
    if timestamps.has_duplicates:
        keep = ~timestamps.duplicated(keep=False)
        timestamps = timestamps[keep]
        values = values[keep]
    if timestamps.empty:
        raise ValueError(f"No valid timestamps in {source}")
    full_index = pd.date_range(timestamps.min(), timestamps.max(), freq="1min")
    reindexed = pd.DataFrame(values, index=timestamps).reindex(full_index)
    station = header.iaga_code or source.stem[:3].upper()
    return station, full_index, reindexed.to_numpy(dtype=np.float32)


def load_npy_series(
    path: str | Path,
    station: str,
    start_time: str | pd.Timestamp,
) -> tuple[str, pd.DatetimeIndex, np.ndarray]:
    values = np.load(Path(path), mmap_mode="r")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("NumPy input must have shape [time, 3] or [3, time].")
    if values.shape[1] == 3:
        time_major = values
    elif values.shape[0] == 3:
        time_major = values.T
    else:
        raise ValueError("NumPy input must contain exactly three XYZ channels.")
    times = pd.date_range(pd.Timestamp(start_time), periods=len(time_major), freq="1min")
    return station.upper(), times, time_major


def score_continuous_series(
    model,
    normalizer: StationRobustNormalizer,
    station: str,
    times: pd.DatetimeIndex,
    values: np.ndarray,
    device,
    baseline_minutes: int = 60,
    window_minutes: int = 360,
    stride_minutes: int = 60,
    batch_size: int = 512,
) -> pd.DataFrame:
    torch = require_torch()
    if len(times) != len(values):
        raise ValueError("times and values must have the same length.")
    if len(values) < window_minutes:
        return pd.DataFrame(columns=["station", "center_time", "score"])
    valid = np.isfinite(values).all(axis=1)
    prefix = np.concatenate(([0], np.cumsum((~valid).astype(np.int64))))
    starts = np.arange(
        0,
        len(values) - int(window_minutes) + 1,
        int(stride_minutes),
        dtype=np.int64,
    )
    invalid_counts = prefix[starts + int(window_minutes)] - prefix[starts]
    starts = starts[invalid_counts == 0]
    records: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), int(batch_size)):
            batch_starts = starts[offset : offset + int(batch_size)]
            batch = np.empty(
                (len(batch_starts), 3, int(window_minutes)),
                dtype=np.float32,
            )
            for index, start in enumerate(batch_starts):
                delta = raw_delta(
                    np.asarray(values[start : start + window_minutes]).T,
                    baseline_minutes=baseline_minutes,
                )
                batch[index] = normalizer.transform(delta, station)
            logits = model(torch.from_numpy(batch).to(device))
            scores = torch.sigmoid(logits).cpu().numpy()
            for start, score in zip(batch_starts, scores):
                center = times[int(start) + int(window_minutes) // 2]
                records.append(
                    {
                        "station": station,
                        "center_time": center,
                        "score": float(score),
                    }
                )
    return pd.DataFrame.from_records(records)


def load_inference_artifacts(
    model_path: str | Path,
    normalization_path: str | Path,
    requested_device: str,
):
    device = resolve_device(requested_device)
    model, model_config = load_model_checkpoint(model_path, device)
    normalization = StationRobustNormalizer.load(normalization_path)
    return model, normalization, device, model_config
