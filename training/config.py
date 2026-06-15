from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "data": {
        "development_years": list(range(2008, 2020)),
        "final_test_years": [2020, 2021, 2022],
        "labels": [0, 1],
    },
    "folds": {
        "fold_1_val_years": [2008, 2011, 2014, 2017],
        "fold_2_val_years": [2009, 2012, 2015, 2018],
        "fold_3_val_years": [2010, 2013, 2016, 2019],
    },
    "features": {
        "raw_delta": True,
        "raw_diff": False,
        "psd": False,
        "cwt": False,
        "baseline_minutes": 60,
    },
    "normalization": {
        "method": "station_robust",
        "max_windows_per_station": 5000,
        "global_max_points_per_station_channel": 500000,
        "clip": 10.0,
        "epsilon": 1.0e-6,
    },
    "model": {
        "input_channels": 3,
        "dropout": 0.3,
    },
    "training": {
        "batch_size": 256,
        "samples_per_epoch": 60000,
        "max_epochs": 30,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "patience": 6,
        "seed": 42,
        "amp": True,
        "num_workers": 4,
        "pin_memory": True,
    },
    "evaluation": {
        "batch_size": 512,
        "num_workers": 4,
        "threshold_objective": "event_f1",
    },
    "postprocess": {
        "threshold_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "smoothing_grid": [1, 3, 5],
        "merge_gap_hours_grid": [1, 3, 6],
        "fusion": "median",
        "min_stations": 3,
        "stride_hours": 1,
        "boundary_half_width_hours": 0.5,
        "match_tolerance_hours": 0,
    },
    "runtime": {
        "device": "cuda",
        "log_level": "INFO",
    },
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


def load_training_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULTS, loaded)

    for key in ("dataset_root", "output_root"):
        value = config["data"].get(key)
        if not value:
            raise ValueError(f"Missing data.{key}")
        config["data"][key] = _expand_path(str(value), config_path.parent)

    development = [int(year) for year in config["data"]["development_years"]]
    test = [int(year) for year in config["data"]["final_test_years"]]
    if development != list(range(2008, 2020)):
        raise ValueError("Experiment 1 development years must be exactly 2008-2019.")
    if test != [2020, 2021, 2022]:
        raise ValueError("Experiment 1 final-test years must be exactly 2020-2022.")
    if set(development) & set(test):
        raise ValueError("Development and final-test years must not overlap.")
    config["data"]["development_years"] = development
    config["data"]["final_test_years"] = test

    fold_years: list[int] = []
    for fold_name in ("fold_1", "fold_2", "fold_3"):
        key = f"{fold_name}_val_years"
        years = [int(year) for year in config["folds"][key]]
        config["folds"][key] = years
        fold_years.extend(years)
    if sorted(fold_years) != sorted(development):
        raise ValueError(
            "The three validation-year lists must partition development_years exactly."
        )

    features = config["features"]
    if not features.get("raw_delta", False):
        raise ValueError("Experiment 1 requires raw_delta.")
    forbidden = ("raw_diff", "psd", "cwt")
    if any(bool(features.get(name, False)) for name in forbidden):
        raise ValueError("Experiment 1 permits raw_delta only.")
    if int(config["model"]["input_channels"]) != 3:
        raise ValueError("Experiment 1 requires exactly three XYZ raw_delta channels.")

    batch_size = int(config["training"]["batch_size"])
    samples_per_epoch = int(config["training"]["samples_per_epoch"])
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("training.batch_size must be a positive even integer.")
    if samples_per_epoch <= 0 or samples_per_epoch % 2:
        raise ValueError("training.samples_per_epoch must be a positive even integer.")
    if int(config["training"]["max_epochs"]) <= 0:
        raise ValueError("training.max_epochs must be positive.")
    if int(config["training"]["patience"]) <= 0:
        raise ValueError("training.patience must be positive.")
    if str(config["normalization"]["method"]) != "station_robust":
        raise ValueError("Experiment 1 requires station_robust normalization.")

    config["_config_path"] = str(config_path)
    return config
