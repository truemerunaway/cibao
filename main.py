from __future__ import annotations

import argparse
import logging
from pathlib import Path

from geomag_dataset.builder import DatasetBuilder
from geomag_dataset.config import load_config
from geomag_dataset.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a geomagnetic storm-event window index from IAGA-2002 data."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--stage",
        choices=("all", "audit", "convert", "symh", "windows"),
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--stations",
        help="Optional comma-separated station override, for example FUR,NGK.",
    )
    parser.add_argument(
        "--years",
        help="Optional year or inclusive range, for example 2015 or 2008:2022.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs that already exist.",
    )
    return parser.parse_args()


def parse_year_override(value: str | None) -> list[int] | None:
    if not value:
        return None
    if ":" not in value:
        return [int(value)]
    start_text, end_text = value.split(":", maxsplit=1)
    start, end = int(start_text), int(end_text)
    if end < start:
        raise ValueError("Year range end must be greater than or equal to start.")
    return list(range(start, end + 1))


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    output_root = Path(config["paths"]["output_root"])
    configure_logging(output_root, config["runtime"]["log_level"])

    stations = None
    if args.stations:
        stations = [item.strip().upper() for item in args.stations.split(",") if item.strip()]
    years = parse_year_override(args.years)

    builder = DatasetBuilder(
        config=config,
        stations=stations,
        years=years,
        overwrite=args.overwrite or config["runtime"]["overwrite"],
    )
    builder.run(args.stage)
    logging.getLogger(__name__).info("Stage '%s' completed.", args.stage)


if __name__ == "__main__":
    main()
