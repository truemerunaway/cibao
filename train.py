from __future__ import annotations

import argparse

from training.config import load_training_config
from training.pipeline import run_cross_validation, run_final_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train experiment 1: 1D CNN + raw_delta.")
    parser.add_argument(
        "stage",
        choices=("cv", "final", "all"),
        help="'all' runs cross-validation and final development-set retraining.",
    )
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoints and reuse fitted normalization files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    if args.stage in ("cv", "all"):
        run_cross_validation(config, resume=args.resume)
    if args.stage in ("final", "all"):
        run_final_training(config, resume=args.resume)


if __name__ == "__main__":
    main()
