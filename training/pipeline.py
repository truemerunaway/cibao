from __future__ import annotations

import json
import logging
import platform
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .dataset import RawDeltaDataset
from .event_postprocess import (
    event_metrics,
    event_stratified_report,
    filter_reference_events,
    fuse_station_scores,
    predictions_to_events,
    smooth_station_scores,
    timeline_to_events,
)
from .folds import (
    annotate_groups,
    boundary_event_ids,
    load_events,
    load_window_index,
    make_fold_assignment,
)
from .inference import load_model_checkpoint
from .metrics import full_window_report, summarize_fold_metrics, window_metrics
from .normalization import StationRobustNormalizer
from .sampler import BalancedGroupBatchSampler
from .trainer import (
    predict_loader,
    predictions_frame,
    require_torch,
    resolve_device,
    seed_everything,
    train_fixed_epochs,
    train_with_early_stopping,
)

LOGGER = logging.getLogger(__name__)


def _new_model(model_config: dict[str, Any]):
    from .models import CNN1D

    return CNN1D(**model_config)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(output)


def configure_training_logging(output_root: Path, level: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_root / "experiment.log", encoding="utf-8"),
        ],
        force=True,
    )


def save_run_metadata(config: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    (output_root / "train_config_used.yaml").write_text(
        yaml.safe_dump(serializable, sort_keys=False),
        encoding="utf-8",
    )
    torch = require_torch()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(config["_config_path"]).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "unavailable"
    device = resolve_device(config["runtime"]["device"])
    metadata = {
        "git_commit": commit,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "seed": int(config["training"]["seed"]),
    }
    write_json(output_root / "run_metadata.json", metadata)


def _data_loader(dataset, batch_size: int, num_workers: int, pin_memory: bool):
    torch = require_torch()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
    )


def _training_loader(dataset, index: pd.DataFrame, config: dict[str, Any]):
    torch = require_torch()
    training = config["training"]
    sampler = BalancedGroupBatchSampler(
        index,
        batch_size=int(training["batch_size"]),
        samples_per_epoch=int(training["samples_per_epoch"]),
        seed=int(training["seed"]),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=bool(training["pin_memory"]),
        persistent_workers=int(training["num_workers"]) > 0,
    )


def _normalizer_for(
    path: Path,
    config: dict[str, Any],
    training_index: pd.DataFrame,
    training_years: list[int],
    resume: bool,
) -> StationRobustNormalizer:
    if resume and path.exists():
        normalizer = StationRobustNormalizer.load(path)
        expected_years = [int(year) for year in training_years]
        observed_years = [
            int(year) for year in normalizer.metadata.get("training_years", [])
        ]
        if observed_years != expected_years:
            raise ValueError(
                f"Normalization training years do not match resume request: "
                f"{observed_years} != {expected_years}"
            )
        return normalizer
    normalization = config["normalization"]
    normalizer = StationRobustNormalizer.fit(
        dataset_root=config["data"]["dataset_root"],
        training_index=training_index,
        training_years=training_years,
        baseline_minutes=int(config["features"]["baseline_minutes"]),
        max_windows_per_station=int(normalization["max_windows_per_station"]),
        global_max_points_per_station_channel=int(
            normalization["global_max_points_per_station_channel"]
        ),
        seed=int(config["training"]["seed"]),
        epsilon=float(normalization["epsilon"]),
        clip=float(normalization["clip"]),
    )
    normalizer.save(path)
    return normalizer


def _fold_reference_events(
    events: pd.DataFrame,
    years: list[int],
    excluded_events: list[str],
) -> pd.DataFrame:
    return filter_reference_events(events, years, excluded_events)


def _event_report(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    predicted_events, _, _ = predictions_to_events(
        predictions,
        threshold=float(parameters["threshold"]),
        smoothing_points=int(parameters["smoothing_points"]),
        merge_gap_hours=float(parameters["merge_gap_hours"]),
        fusion=str(parameters["fusion"]),
        min_stations=int(parameters["min_stations"]),
        stride_hours=float(parameters["stride_hours"]),
        boundary_half_width_hours=float(parameters["boundary_half_width_hours"]),
    )
    metrics, matches = event_metrics(
        predicted_events,
        references,
        match_tolerance_hours=float(parameters["match_tolerance_hours"]),
    )
    metrics["stratified"] = event_stratified_report(matches, references)
    by_station: dict[str, Any] = {}
    for station, station_predictions in predictions.groupby("station", sort=True):
        station_events, _, _ = predictions_to_events(
            station_predictions,
            threshold=float(parameters["threshold"]),
            smoothing_points=int(parameters["smoothing_points"]),
            merge_gap_hours=float(parameters["merge_gap_hours"]),
            fusion="median",
            min_stations=1,
            stride_hours=float(parameters["stride_hours"]),
            boundary_half_width_hours=float(
                parameters["boundary_half_width_hours"]
            ),
        )
        station_metrics, _ = event_metrics(
            station_events,
            references,
            match_tolerance_hours=float(parameters["match_tolerance_hours"]),
        )
        by_station[str(station)] = station_metrics
    metrics["by_station"] = by_station
    return metrics, predicted_events, matches


def select_oof_parameters(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    postprocess = config["postprocess"]
    records: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], dict[str, Any]] | None = None
    for smoothing in postprocess["smoothing_grid"]:
        smoothed = smooth_station_scores(
            predictions,
            smoothing_points=int(smoothing),
            stride_hours=float(postprocess["stride_hours"]),
        )
        fused = fuse_station_scores(
            smoothed,
            threshold=0.0,
            min_stations=int(postprocess["min_stations"]),
            method=str(postprocess["fusion"]),
        )
        for threshold in postprocess["threshold_grid"]:
            window = window_metrics(
                predictions["label"],
                predictions["score"],
                float(threshold),
            )
            for merge_gap in postprocess["merge_gap_hours_grid"]:
                predicted_events = timeline_to_events(
                    fused,
                    threshold=float(threshold),
                    merge_gap_hours=float(merge_gap),
                    stride_hours=float(postprocess["stride_hours"]),
                    boundary_half_width_hours=float(
                        postprocess["boundary_half_width_hours"]
                    ),
                )
                event, _ = event_metrics(
                    predicted_events,
                    references,
                    match_tolerance_hours=float(
                        postprocess["match_tolerance_hours"]
                    ),
                )
                record = {
                    "threshold": float(threshold),
                    "smoothing_points": int(smoothing),
                    "merge_gap_hours": float(merge_gap),
                    "window_f1": window["f1"],
                    "event_f1": event["event_f1"],
                    "event_detection_rate": event["event_detection_rate"],
                    "false_alarm_event_count": event["false_alarm_event_count"],
                }
                records.append(record)
                event_f1 = event["event_f1"]
                objective = (
                    float(event_f1) if event_f1 is not None else -1.0,
                    float(window["f1"]),
                )
                if best is None or objective > best[0]:
                    best = (objective, record)
    if best is None:
        raise RuntimeError("OOF parameter search produced no candidates.")
    selected = dict(best[1])
    selected.update(
        {
            "fusion": str(postprocess["fusion"]),
            "min_stations": int(postprocess["min_stations"]),
            "stride_hours": float(postprocess["stride_hours"]),
            "boundary_half_width_hours": float(
                postprocess["boundary_half_width_hours"]
            ),
            "match_tolerance_hours": float(postprocess["match_tolerance_hours"]),
            "selection_source": "out_of_fold_only",
        }
    )
    return selected, pd.DataFrame.from_records(records)


def load_locked_postprocess(output_root: str | Path) -> dict[str, Any]:
    path = Path(output_root) / "selected_postprocess.json"
    if not path.exists():
        raise FileNotFoundError(f"OOF-selected post-processing file not found: {path}")
    parameters = json.loads(path.read_text(encoding="utf-8"))
    if parameters.get("selection_source") != "out_of_fold_only":
        raise ValueError("Final-test parameters were not selected from OOF predictions.")
    required = {
        "threshold",
        "smoothing_points",
        "merge_gap_hours",
        "fusion",
        "min_stations",
        "stride_hours",
        "boundary_half_width_hours",
        "match_tolerance_hours",
    }
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError(f"Locked post-processing parameters are incomplete: {missing}")
    return parameters


def run_cross_validation(config: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    output_root = Path(config["data"]["output_root"])
    configure_training_logging(output_root, config["runtime"]["log_level"])
    save_run_metadata(config, output_root)
    seed_everything(int(config["training"]["seed"]))
    device = resolve_device(config["runtime"]["device"])
    dataset_root = config["data"]["dataset_root"]
    development_years = config["data"]["development_years"]
    final_test_years = config["data"]["final_test_years"]
    events = load_events(dataset_root)
    raw_index = load_window_index(dataset_root, development_years)
    index = annotate_groups(raw_index, events)
    oof_frames: list[pd.DataFrame] = []
    fold_training_results: dict[str, dict[str, Any]] = {}

    for number in (1, 2, 3):
        name = f"fold_{number}"
        fold_dir = output_root / name
        validation_years = config["folds"][f"{name}_val_years"]
        assignment = make_fold_assignment(
            index,
            events,
            development_years,
            final_test_years,
            validation_years,
            name,
        )
        write_json(fold_dir / "audit.json", assignment.audit)
        normalization_path = fold_dir / "normalization.json"
        normalizer = _normalizer_for(
            normalization_path,
            config,
            assignment.train,
            assignment.audit["training_years"],
            resume=resume,
        )
        train_dataset = RawDeltaDataset(
            dataset_root,
            assignment.train,
            normalizer,
            baseline_minutes=int(config["features"]["baseline_minutes"]),
        )
        validation_dataset = RawDeltaDataset(
            dataset_root,
            assignment.validation,
            normalizer,
            baseline_minutes=int(config["features"]["baseline_minutes"]),
        )
        train_loader = _training_loader(train_dataset, assignment.train, config)
        validation_loader = _data_loader(
            validation_dataset,
            batch_size=int(config["evaluation"]["batch_size"]),
            num_workers=int(config["evaluation"]["num_workers"]),
            pin_memory=bool(config["training"]["pin_memory"]),
        )
        model = _new_model(config["model"])
        result = train_with_early_stopping(
            model,
            train_loader,
            validation_loader,
            validation_dataset,
            fold_dir,
            config["training"],
            config["model"],
            device,
            resume=resume,
        )
        predictions = result["predictions"]
        predictions["fold"] = name
        predictions.to_parquet(fold_dir / "val_predictions.parquet", index=False)
        oof_frames.append(predictions)
        fold_training_results[name] = {
            "best_epoch": int(result["best_epoch"]),
            "best_pr_auc": float(result["best_pr_auc"]),
        }
        write_json(fold_dir / "training_result.json", fold_training_results[name])

    oof = pd.concat(oof_frames, ignore_index=True)
    oof.to_parquet(output_root / "oof_predictions.parquet", index=False)
    excluded = boundary_event_ids(events, development_years, final_test_years)
    references = filter_reference_events(events, development_years, excluded)
    selected, search = select_oof_parameters(oof, references, config)
    search.to_csv(output_root / "oof_parameter_search.csv", index=False)
    write_json(output_root / "selected_postprocess.json", selected)

    fold_reports: dict[str, dict[str, Any]] = {}
    for number in (1, 2, 3):
        name = f"fold_{number}"
        fold_predictions = oof.loc[oof["fold"].eq(name)].copy()
        reference = _fold_reference_events(
            events,
            config["folds"][f"{name}_val_years"],
            sorted(excluded),
        )
        event_report, predicted_events, matches = _event_report(
            fold_predictions,
            reference,
            selected,
        )
        report = {
            "window": full_window_report(
                fold_predictions,
                threshold=float(selected["threshold"]),
            ),
            "event": event_report,
            **fold_training_results[name],
        }
        fold_reports[name] = report
        fold_dir = output_root / name
        write_json(fold_dir / "metrics.json", report)
        predicted_events.to_csv(fold_dir / "predicted_events.csv", index=False)
        matches.to_csv(fold_dir / "event_matches.csv", index=False)

    oof_event_report, oof_events, oof_matches = _event_report(
        oof,
        references,
        selected,
    )
    best_epochs = [value["best_epoch"] for value in fold_training_results.values()]
    summary = summarize_fold_metrics(fold_reports)
    summary["oof_window"] = full_window_report(
        oof,
        threshold=float(selected["threshold"]),
    )
    summary["oof_event"] = oof_event_report
    summary["selected_postprocess"] = selected
    summary["final_training_epochs"] = int(median(best_epochs))
    write_json(output_root / "cv_summary.json", summary)
    oof_events.to_csv(output_root / "oof_predicted_events.csv", index=False)
    oof_matches.to_csv(output_root / "oof_event_matches.csv", index=False)
    return summary


def _development_training_index(
    config: dict[str, Any],
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    development_years = config["data"]["development_years"]
    final_test_years = config["data"]["final_test_years"]
    index = annotate_groups(
        load_window_index(config["data"]["dataset_root"], development_years),
        events,
    )
    excluded = boundary_event_ids(events, development_years, final_test_years)
    selected = index.loc[
        index["group_year"].isin(development_years)
        & index["label"].isin([0, 1])
        & ~index["event_id"].isin(excluded)
    ].reset_index(drop=True)
    return selected, excluded


def run_final_training(config: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    output_root = Path(config["data"]["output_root"])
    summary_path = output_root / "cv_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run three-fold cross-validation before final training.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    epochs = int(summary["final_training_epochs"])
    events = load_events(config["data"]["dataset_root"])
    training_index, excluded = _development_training_index(config, events)
    final_dir = output_root / "final"
    write_json(
        final_dir / "audit.json",
        {
            "training_years": config["data"]["development_years"],
            "training_window_count": len(training_index),
            "excluded_boundary_event_ids": sorted(excluded),
            "test_years_loaded": [],
        },
    )
    normalizer = _normalizer_for(
        final_dir / "normalization.json",
        config,
        training_index,
        config["data"]["development_years"],
        resume=resume,
    )
    dataset = RawDeltaDataset(
        config["data"]["dataset_root"],
        training_index,
        normalizer,
        baseline_minutes=int(config["features"]["baseline_minutes"]),
    )
    loader = _training_loader(dataset, training_index, config)
    seed_everything(int(config["training"]["seed"]))
    device = resolve_device(config["runtime"]["device"])
    result = train_fixed_epochs(
        _new_model(config["model"]),
        loader,
        final_dir,
        config["training"],
        config["model"],
        epochs,
        device,
        resume=resume,
    )
    write_json(
        final_dir / "training_result.json",
        {"epochs": epochs, "training_window_count": len(training_index)},
    )
    return {"epochs": epochs, "history": result["history"]}


def run_final_test(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["data"]["output_root"])
    final_dir = output_root / "final"
    required = [
        final_dir / "final_model.pt",
        final_dir / "normalization.json",
        output_root / "selected_postprocess.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing final-test artifacts: {missing}")
    parameters = load_locked_postprocess(output_root)

    dataset_root = config["data"]["dataset_root"]
    events = load_events(dataset_root)
    test_years = config["data"]["final_test_years"]
    index = annotate_groups(load_window_index(dataset_root, test_years), events)
    excluded = boundary_event_ids(
        events,
        config["data"]["development_years"],
        test_years,
    )
    test_index = index.loc[
        index["group_year"].isin(test_years)
        & index["label"].isin([0, 1])
        & ~index["event_id"].isin(excluded)
    ].reset_index(drop=True)
    normalizer = StationRobustNormalizer.load(required[1])
    dataset = RawDeltaDataset(
        dataset_root,
        test_index,
        normalizer,
        baseline_minutes=int(config["features"]["baseline_minutes"]),
    )
    loader = _data_loader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=bool(config["training"]["pin_memory"]),
    )
    device = resolve_device(config["runtime"]["device"])
    model, _ = load_model_checkpoint(required[0], device)
    positions, _, scores, loss = predict_loader(model, loader, device)
    predictions = predictions_frame(dataset, positions, scores)
    predictions.to_parquet(final_dir / "test_predictions.parquet", index=False)
    references = filter_reference_events(events, test_years, excluded)
    event_report, predicted_events, matches = _event_report(
        predictions,
        references,
        parameters,
    )
    report = {
        "window": full_window_report(
            predictions,
            threshold=float(parameters["threshold"]),
        ),
        "event": event_report,
        "loss": float(loss),
        "parameters_locked_from_oof": True,
        "threshold_search_performed": False,
        "test_years": test_years,
        "excluded_boundary_event_ids": sorted(excluded),
    }
    write_json(final_dir / "test_metrics.json", report)
    predicted_events.to_csv(final_dir / "test_predicted_events.csv", index=False)
    matches.to_csv(final_dir / "test_event_matches.csv", index=False)
    return report
