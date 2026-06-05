from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "station_data": {
        "components": ["X", "Y", "Z"],
        "accepted_data_types": ["Definitive", "Quasi-definitive", "Provisional"],
        "rejected_data_types": ["Variation"],
        "missing_value_markers": [
            99999,
            99999.0,
            999999,
            999999.0,
            -99999,
            -99999.0,
            -999999,
            -999999.0,
        ],
        "read_chunk_rows": 100000,
    },
    "symh": {
        "file_pattern": "{year}.txt",
        "missing_value_markers": [99999, 999999, -99999, -999999],
        "storm_threshold_nt": -50,
        "nonstorm_threshold_nt": -30,
        "minimum_core_minutes": 30,
        "merge_gap_hours": 12,
        "nonstorm_buffer_hours": 24,
    },
    "window": {
        "length_hours": 6,
        "stride_hours": 1,
        "baseline_minutes": 60,
    },
    "features": {
        "raw_delta": True,
        "raw_diff": True,
        "psd": {"enabled": False, "nperseg": 128, "noverlap": 64},
        "cwt": {
            "enabled": False,
            "wavelet": "morl",
            "num_scales": 32,
            "min_period_minutes": 4,
            "max_period_minutes": 180,
        },
    },
    "runtime": {"overwrite": False, "log_level": "INFO"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_path(value: str, config_dir: Path) -> str:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = (config_dir / expanded).resolve()
    return str(expanded)


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULTS, loaded)

    required_sections = ("paths", "stations", "years", "split")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    for key in ("station_data_root", "symh_data_root", "output_root"):
        if key not in config["paths"]:
            raise ValueError(f"Missing paths.{key}")
        config["paths"][key] = _expand_path(config["paths"][key], path.parent)

    start, end = int(config["years"]["start"]), int(config["years"]["end"])
    if end < start:
        raise ValueError("years.end must be greater than or equal to years.start")
    config["years"]["start"] = start
    config["years"]["end"] = end

    accepted = {str(item).strip().lower() for item in config["station_data"]["accepted_data_types"]}
    rejected = {str(item).strip().lower() for item in config["station_data"]["rejected_data_types"]}
    if accepted & rejected:
        raise ValueError("Accepted and rejected station data types overlap.")

    train = set(config["split"]["train_years"])
    val = set(config["split"]["val_years"])
    test = set(config["split"]["test_years"])
    if train & val or train & test or val & test:
        raise ValueError("train_years, val_years and test_years must not overlap.")

    return config
