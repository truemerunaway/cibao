from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import atomic_save_npy, atomic_write_json, minutes_in_year

LOGGER = logging.getLogger(__name__)

DATA_TYPE_NAMES = {
    "d": "Definitive",
    "definitive": "Definitive",
    "q": "Quasi-definitive",
    "quasi-definitive": "Quasi-definitive",
    "quasi definitive": "Quasi-definitive",
    "p": "Provisional",
    "provisional": "Provisional",
    "v": "Variation",
    "variation": "Variation",
}

DATA_TYPE_RANK = {
    "Definitive": 3,
    "Quasi-definitive": 2,
    "Provisional": 1,
    "Variation": 0,
}


@dataclass(frozen=True)
class IAGAHeader:
    path: str
    iaga_code: str
    station_name: str
    reported: str
    data_type: str
    publication_date: str
    data_interval_type: str
    columns: list[str]
    data_start_line: int


def normalize_data_type(value: str) -> str:
    normalized = " ".join(value.strip().lower().replace("_", " ").split())
    normalized = normalized.replace(" - ", "-")
    return DATA_TYPE_NAMES.get(normalized, value.strip())


def _header_value(line: str, label: str) -> str | None:
    text = line.rstrip().rstrip("|").strip()
    if not text.lower().startswith(label.lower()):
        return None
    return text[len(label) :].strip()


def parse_iaga_header(path: Path) -> IAGAHeader:
    values: dict[str, str] = {}
    columns: list[str] | None = None
    data_start_line: int | None = None

    labels = {
        "iaga_code": "IAGA Code",
        "station_name": "Station Name",
        "reported": "Reported",
        "data_type": "Data Type",
        "publication_date": "Publication Date",
        "data_interval_type": "Data Interval Type",
    }

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip()
            if stripped.upper().startswith("DATE") and "TIME" in stripped.upper():
                columns = [token for token in stripped.rstrip("|").split() if token != "|"]
                data_start_line = line_number + 1
                break
            for key, label in labels.items():
                value = _header_value(line, label)
                if value is not None:
                    values[key] = value

    if columns is None or data_start_line is None:
        raise ValueError(f"IAGA data header line was not found in {path}")
    if "data_type" not in values:
        raise ValueError(f"Data Type header is missing in {path}")

    return IAGAHeader(
        path=str(path),
        iaga_code=values.get("iaga_code", "").upper(),
        station_name=values.get("station_name", ""),
        reported=values.get("reported", "").upper(),
        data_type=normalize_data_type(values["data_type"]),
        publication_date=values.get("publication_date", ""),
        data_interval_type=values.get("data_interval_type", ""),
        columns=columns,
        data_start_line=data_start_line,
    )


def _publication_sort_value(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.min


def discover_candidates(root: Path, station: str, year: int) -> list[Path]:
    year_dir = root / station / str(year)
    if not year_dir.exists():
        return []
    return sorted(path for path in year_dir.rglob("*.txt") if path.is_file())


def select_best_source(
    root: Path,
    station: str,
    year: int,
    accepted_data_types: Iterable[str],
    rejected_data_types: Iterable[str],
) -> tuple[IAGAHeader | None, list[dict[str, str]]]:
    accepted = {normalize_data_type(value).lower() for value in accepted_data_types}
    rejected = {normalize_data_type(value).lower() for value in rejected_data_types}
    usable: list[IAGAHeader] = []
    candidate_status: list[dict[str, str]] = []

    for path in discover_candidates(root, station, year):
        try:
            header = parse_iaga_header(path)
        except Exception as exc:  # Keep the batch running and report the bad candidate.
            candidate_status.append(
                {"path": str(path), "status": "INVALID_HEADER", "reason": str(exc)}
            )
            continue

        data_type_key = header.data_type.lower()
        if data_type_key in rejected:
            status = "REJECTED_DATA_TYPE"
        elif data_type_key not in accepted:
            status = "UNSUPPORTED_DATA_TYPE"
        elif "1-minute" not in header.data_interval_type.lower().replace(" ", "-"):
            status = "UNSUPPORTED_INTERVAL"
        elif header.iaga_code and header.iaga_code != station.upper():
            status = "IAGA_CODE_MISMATCH"
        else:
            status = "ACCEPTED_CANDIDATE"
            usable.append(header)

        candidate_status.append(
            {
                "path": str(path),
                "status": status,
                "data_type": header.data_type,
                "iaga_code": header.iaga_code,
            }
        )

    if not usable:
        return None, candidate_status

    usable.sort(
        key=lambda item: (
            DATA_TYPE_RANK.get(item.data_type, -1),
            _publication_sort_value(item.publication_date),
            item.path,
        ),
        reverse=True,
    )
    return usable[0], candidate_status


def component_columns(header: IAGAHeader, components: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    upper_columns = [column.upper() for column in header.columns]
    reserved = {"DATE", "TIME", "DOY"}
    for component in components:
        component = component.upper()
        preferred_names = {component}
        if header.iaga_code:
            preferred_names.add(f"{header.iaga_code}{component}")
        matches = [
            original
            for original, upper in zip(header.columns, upper_columns)
            if upper in preferred_names
        ]
        if not matches:
            matches = [
                original
                for original, upper in zip(header.columns, upper_columns)
                if upper not in reserved and upper.endswith(component)
            ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one column for component {component} in "
                f"{header.path}; found {matches}"
            )
        mapping[component] = matches[0]
    return mapping


def convert_iaga_year(
    header: IAGAHeader,
    station: str,
    year: int,
    output_root: Path,
    components: list[str],
    missing_markers: Iterable[float],
    chunk_rows: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    station = station.upper()
    output_dir = output_root / "continuous" / station
    data_path = output_dir / f"{year}_data.npy"
    valid_path = output_dir / f"{year}_valid.npy"
    metadata_path = output_dir / f"{year}_metadata.json"

    if not overwrite and data_path.exists() and valid_path.exists() and metadata_path.exists():
        LOGGER.info("Skip existing converted data: %s %s", station, year)
        import json

        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    expected = minutes_in_year(year)
    data = np.full((expected, len(components)), np.nan, dtype=np.float32)
    observation_count = np.zeros(expected, dtype=np.uint8)
    mapping = component_columns(header, components)
    selected_columns = ["DATE", "TIME", *mapping.values()]
    source = Path(header.path)
    actual_rows = 0
    invalid_timestamp_rows = 0
    out_of_year_rows = 0

    reader = pd.read_csv(
        source,
        sep=r"\s+",
        skiprows=header.data_start_line,
        names=header.columns,
        usecols=selected_columns,
        dtype=str,
        chunksize=chunk_rows,
        engine="c",
    )

    year_start = pd.Timestamp(year=year, month=1, day=1)
    marker_array = np.asarray(list(missing_markers), dtype=np.float64)

    for chunk in reader:
        actual_rows += len(chunk)
        timestamps = pd.to_datetime(
            chunk["DATE"].astype(str) + " " + chunk["TIME"].astype(str),
            errors="coerce",
        )
        seconds = (timestamps - year_start).dt.total_seconds().to_numpy(dtype=np.float64)
        minute_values = seconds / 60.0
        valid_time = np.isfinite(minute_values) & np.isclose(
            minute_values, np.rint(minute_values), atol=1e-6
        )
        offsets = np.zeros(len(chunk), dtype=np.int64)
        offsets[valid_time] = np.rint(minute_values[valid_time]).astype(np.int64)
        in_year = valid_time & (offsets >= 0) & (offsets < expected)
        invalid_timestamp_rows += int((~valid_time).sum())
        out_of_year_rows += int((valid_time & ~in_year).sum())

        if not in_year.any():
            continue

        selected_offsets = offsets[in_year]
        values = np.column_stack(
            [
                pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=np.float64)
                for column in mapping.values()
            ]
        )[in_year]
        if marker_array.size:
            values[np.isin(values, marker_array)] = np.nan

        data[selected_offsets] = values.astype(np.float32)
        np.add.at(observation_count, selected_offsets, 1)

    finite_rows = np.isfinite(data).all(axis=1)
    valid = (observation_count == 1) & finite_rows
    duplicate_minutes = int((observation_count > 1).sum())
    missing_minutes = int((observation_count == 0).sum())
    invalid_value_minutes = int(((observation_count > 0) & ~finite_rows).sum())

    atomic_save_npy(data_path, data)
    atomic_save_npy(valid_path, valid)

    metadata: dict[str, Any] = {
        "station": station,
        "year": year,
        "source_file": str(source),
        "iaga_code": header.iaga_code,
        "station_name": header.station_name,
        "reported": header.reported,
        "data_type": header.data_type,
        "data_type_rank": DATA_TYPE_RANK.get(header.data_type, -1),
        "publication_date": header.publication_date,
        "data_interval_type": header.data_interval_type,
        "components": [item.upper() for item in components],
        "expected_minutes": expected,
        "actual_rows": actual_rows,
        "valid_minutes": int(valid.sum()),
        "missing_minutes": missing_minutes,
        "invalid_value_minutes": invalid_value_minutes,
        "duplicate_minutes": duplicate_minutes,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "out_of_year_rows": out_of_year_rows,
        "data_path": str(data_path),
        "valid_path": str(valid_path),
        "status": "CONVERTED",
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def header_audit_record(
    station: str,
    year: int,
    selected: IAGAHeader | None,
    candidates: list[dict[str, str]],
) -> dict[str, Any]:
    if selected is None:
        return {
            "station": station,
            "year": year,
            "status": "NO_ACCEPTED_SOURCE",
            "source_file": "",
            "data_type": "",
            "reported": "",
            "candidate_count": len(candidates),
            "candidate_details": candidates,
        }
    record = asdict(selected)
    record.update(
        {
            "station": station,
            "year": year,
            "status": "SOURCE_SELECTED",
            "source_file": selected.path,
            "candidate_count": len(candidates),
            "candidate_details": candidates,
        }
    )
    record.pop("path", None)
    return record
