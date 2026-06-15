from __future__ import annotations

import argparse

from training.config import load_training_config
from training.pipeline import run_final_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the locked final model on permanent 2020-2022 test data."
    )
    parser.add_argument("--config", default="train_config.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_training_config(parse_args().config)
    run_final_test(config)


if __name__ == "__main__":
    main()
