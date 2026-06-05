from __future__ import annotations

import json
import logging
import os
from calendar import isleap
from pathlib import Path
from typing import Any

import numpy as np


def minutes_in_year(year: int) -> int:
    return (366 if isleap(year) else 365) * 1440


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=_json_default)
    os.replace(temporary, path)


def atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=False, sort_keys=False)
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def configure_logging(output_root: Path, level: str) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "build_dataset.log", encoding="utf-8"),
        ],
        force=True,
    )


def split_for_year(year: int, split_config: dict[str, Any]) -> str:
    if year in split_config["train_years"]:
        return "train"
    if year in split_config["val_years"]:
        return "val"
    if year in split_config["test_years"]:
        return "test"
    return "excluded"
