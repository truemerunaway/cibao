from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .iaga import (
    convert_iaga_year,
    header_audit_record,
    select_best_source,
)
from .symh import build_storm_events, parse_symh_year
from .utils import atomic_write_json, atomic_write_yaml
from .windows import build_window_partition

LOGGER = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(
        self,
        config: dict[str, Any],
        stations: list[str] | None = None,
        years: list[int] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.config = config
        self.stations = stations or [item.upper() for item in config["stations"]]
        self.years = years or list(
            range(config["years"]["start"], config["years"]["end"] + 1)
        )
        self.years = sorted(set(self.years))
        self.overwrite = overwrite
        self.station_root = Path(config["paths"]["station_data_root"])
        self.symh_root = Path(config["paths"]["symh_data_root"])
        self.output_root = Path(config["paths"]["output_root"])
        self.output_root.mkdir(parents=True, exist_ok=True)
        atomic_write_yaml(self.output_root / "config_used.yaml", config)

    def run(self, stage: str) -> None:
        if stage == "audit":
            self.run_audit()
        elif stage == "convert":
            self.run_convert()
        elif stage == "symh":
            self.run_symh()
        elif stage == "windows":
            self.run_windows()
        elif stage == "all":
            self.run_audit()
            self.run_convert()
            self.run_symh()
            self.run_windows()
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _select_source(self, station: str, year: int):
        station_config = self.config["station_data"]
        return select_best_source(
            self.station_root,
            station,
            year,
            station_config["accepted_data_types"],
            station_config["rejected_data_types"],
        )

    def run_audit(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        total = len(self.stations) * len(self.years)
        for station in self.stations:
            for year in tqdm(self.years, desc=f"Audit {station}", leave=False):
                selected, candidates = self._select_source(station, year)
                records.append(header_audit_record(station, year, selected, candidates))

        frame = pd.DataFrame.from_records(records)
        if "candidate_details" in frame:
            frame["candidate_details"] = frame["candidate_details"].map(
                lambda value: json.dumps(value, ensure_ascii=True)
            )
        audit_dir = self.output_root / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(audit_dir / "station_year_sources.csv", index=False)
        LOGGER.info(
            "Audited %d station-year inputs; %d accepted sources.",
            total,
            int((frame["status"] == "SOURCE_SELECTED").sum()),
        )
        return frame

    def run_convert(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        station_config = self.config["station_data"]

        for station in self.stations:
            for year in tqdm(self.years, desc=f"Convert {station}", leave=False):
                selected, candidates = self._select_source(station, year)
                if selected is None:
                    records.append(
                        {
                            "station": station,
                            "year": year,
                            "status": "NO_ACCEPTED_SOURCE",
                            "candidate_count": len(candidates),
                        }
                    )
                    continue
                try:
                    record = convert_iaga_year(
                        selected,
                        station,
                        year,
                        self.output_root,
                        [item.upper() for item in station_config["components"]],
                        station_config["missing_value_markers"],
                        int(station_config["read_chunk_rows"]),
                        overwrite=self.overwrite,
                    )
                except Exception as exc:
                    LOGGER.exception("Failed to convert %s %s", station, year)
                    record = {
                        "station": station,
                        "year": year,
                        "status": "CONVERSION_FAILED",
                        "source_file": selected.path,
                        "error": str(exc),
                    }
                records.append(record)

        frame = pd.DataFrame.from_records(records)
        audit_dir = self.output_root / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(audit_dir / "station_year_conversion.csv", index=False)
        LOGGER.info(
            "Converted or reused %d station-year arrays.",
            int(frame["status"].isin(["CONVERTED", "EXISTING"]).sum())
            if "status" in frame
            else 0,
        )
        return frame

    def run_symh(self) -> pd.DataFrame:
        if self.years != list(range(self.years[0], self.years[-1] + 1)):
            raise ValueError("SYM-H event generation requires a consecutive year range.")

        symh_config = self.config["symh"]
        records: list[dict[str, Any]] = []
        for year in tqdm(self.years, desc="Convert SYM-H"):
            path = self.symh_root / symh_config["file_pattern"].format(year=year)
            record = parse_symh_year(
                path,
                year,
                self.output_root,
                symh_config["missing_value_markers"],
                overwrite=self.overwrite,
            )
            records.append(record)

        audit_dir = self.output_root / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(records).to_csv(
            audit_dir / "symh_year_conversion.csv", index=False
        )
        events = build_storm_events(
            self.output_root,
            self.years,
            float(symh_config["storm_threshold_nt"]),
            int(symh_config["minimum_core_minutes"]),
            float(symh_config["merge_gap_hours"]),
            self.config["split"],
        )
        LOGGER.info("Generated %d merged storm events.", len(events))
        return events

    def run_windows(self) -> pd.DataFrame:
        events_path = self.output_root / "labels" / "storm_events.csv"
        if not events_path.exists():
            raise FileNotFoundError(
                f"Storm event table not found: {events_path}. Run --stage symh first."
            )
        events = pd.read_csv(
            events_path,
            parse_dates=["event_start", "event_end"],
        )
        records: list[dict[str, Any]] = []

        for station in self.stations:
            for year in tqdm(self.years, desc=f"Windows {station}", leave=False):
                try:
                    record = build_window_partition(
                        self.output_root,
                        station,
                        year,
                        events,
                        self.config["window"],
                        self.config["symh"],
                        self.config["split"],
                        overwrite=self.overwrite,
                    )
                except Exception as exc:
                    LOGGER.exception("Failed to build windows for %s %s", station, year)
                    record = {
                        "station": station,
                        "year": year,
                        "status": "WINDOW_BUILD_FAILED",
                        "windows": 0,
                        "error": str(exc),
                    }
                records.append(record)

        summary = pd.DataFrame.from_records(records)
        reports_dir = self.output_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(reports_dir / "window_partitions.csv", index=False)
        numeric_columns = [
            column
            for column in (
                "windows",
                "storm_windows",
                "nonstorm_windows",
                "uncertain_windows",
                "skipped_missing_windows",
            )
            if column in summary
        ]
        totals = {column: int(summary[column].fillna(0).sum()) for column in numeric_columns}
        totals["stations"] = self.stations
        totals["years"] = self.years
        atomic_write_json(reports_dir / "dataset_summary.json", totals)
        LOGGER.info("Window index summary: %s", totals)
        return summary
