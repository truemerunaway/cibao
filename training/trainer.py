from __future__ import annotations

import logging
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import window_metrics

LOGGER = logging.getLogger(__name__)


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required. Use the CUDA-enabled PyTorch already installed "
            "in the AutoDL image."
        ) from exc
    return torch


def seed_everything(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str):
    torch = require_torch()
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _scaler(enabled: bool):
    torch = require_torch()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device, enabled: bool):
    torch = require_torch()
    if not enabled:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=True,
    )


def _torch_load(path: Path, device):
    torch = require_torch()
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _rng_state() -> dict[str, Any]:
    torch = require_torch()
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch = require_torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    torch = require_torch()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    model.eval()
    positions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            logits = model(inputs)
            loss_sum += float(criterion(logits, targets).item())
            positions.append(batch["position"].cpu().numpy())
            labels.append(targets.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    if not positions:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float32),
            np.nan,
        )
    all_positions = np.concatenate(positions)
    all_labels = np.concatenate(labels).astype(np.int64)
    all_scores = np.concatenate(scores).astype(np.float32)
    return (
        all_positions,
        all_labels,
        all_scores,
        loss_sum / len(all_labels),
    )


def predictions_frame(dataset, positions: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    frame = dataset.index.iloc[positions].reset_index(drop=True).copy()
    frame["score"] = scores
    return frame


def _checkpoint_payload(
    model,
    optimizer,
    scaler,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    patience_count: int,
    history: list[dict[str, Any]],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "patience_count": int(patience_count),
        "history": history,
        "rng_state": _rng_state(),
        "model_config": model_config,
    }


def train_with_early_stopping(
    model,
    train_loader,
    validation_loader,
    validation_dataset,
    output_dir: str | Path,
    training_config: dict[str, Any],
    model_config: dict[str, Any],
    device,
    resume: bool = False,
) -> dict[str, Any]:
    torch = require_torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    amp_enabled = bool(training_config["amp"]) and device.type == "cuda"
    scaler = _scaler(amp_enabled)
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_model.pt"
    start_epoch = 0
    best_epoch = -1
    best_metric = -np.inf
    patience_count = 0
    history: list[dict[str, Any]] = []

    if resume and last_path.exists():
        checkpoint = _torch_load(last_path, device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_metric = float(checkpoint["best_metric"])
        patience_count = int(checkpoint["patience_count"])
        history = list(checkpoint.get("history", []))
        _restore_rng_state(checkpoint["rng_state"])
        LOGGER.info("Resuming %s from epoch %d.", output_dir, start_epoch + 1)

    max_epochs = int(training_config["max_epochs"])
    already_stopped = patience_count >= int(training_config["patience"])
    for epoch in range(start_epoch, max_epochs):
        if already_stopped:
            break
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_count = int(targets.numel())
            loss_sum += float(loss.item()) * batch_count
            sample_count += batch_count

        positions, labels, scores, validation_loss = predict_loader(
            model,
            validation_loader,
            device,
        )
        validation_metrics = window_metrics(labels, scores, threshold=0.5)
        current_metric = validation_metrics["pr_auc"]
        if current_metric is None:
            raise ValueError("Validation PR-AUC is undefined.")
        record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(sample_count, 1),
            "validation_loss": validation_loss,
            "validation_pr_auc": current_metric,
            "validation_roc_auc": validation_metrics["roc_auc"],
        }
        history.append(record)
        improved = current_metric > best_metric
        if improved:
            best_metric = float(current_metric)
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        payload = _checkpoint_payload(
            model,
            optimizer,
            scaler,
            epoch,
            best_epoch,
            best_metric,
            patience_count,
            history,
            model_config,
        )
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        LOGGER.info(
            "Epoch %d/%d train_loss=%.5f val_loss=%.5f val_pr_auc=%.5f",
            epoch + 1,
            max_epochs,
            record["train_loss"],
            validation_loss,
            current_metric,
        )
        if patience_count >= int(training_config["patience"]):
            LOGGER.info("Early stopping at epoch %d.", epoch + 1)
            break

    best = _torch_load(best_path, device)
    model.load_state_dict(best["model_state"])
    positions, labels, scores, validation_loss = predict_loader(
        model,
        validation_loader,
        device,
    )
    predictions = predictions_frame(validation_dataset, positions, scores)
    return {
        "model": model,
        "predictions": predictions,
        "best_epoch": int(best["best_epoch"]) + 1,
        "best_pr_auc": float(best["best_metric"]),
        "validation_loss": float(validation_loss),
        "history": list(best.get("history", history)),
    }


def train_fixed_epochs(
    model,
    train_loader,
    output_dir: str | Path,
    training_config: dict[str, Any],
    model_config: dict[str, Any],
    epochs: int,
    device,
    resume: bool = False,
) -> dict[str, Any]:
    torch = require_torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    amp_enabled = bool(training_config["amp"]) and device.type == "cuda"
    scaler = _scaler(amp_enabled)
    checkpoint_path = output_dir / "last_checkpoint.pt"
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if resume and checkpoint_path.exists():
        checkpoint = _torch_load(checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
        _restore_rng_state(checkpoint["rng_state"])

    for epoch in range(start_epoch, int(epochs)):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = int(targets.numel())
            loss_sum += float(loss.item()) * count
            sample_count += count
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / max(sample_count, 1),
            }
        )
        payload = _checkpoint_payload(
            model,
            optimizer,
            scaler,
            epoch,
            epoch,
            0.0,
            0,
            history,
            model_config,
        )
        _atomic_torch_save(payload, checkpoint_path)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    final_payload = _torch_load(checkpoint_path, device)
    _atomic_torch_save(final_payload, output_dir / "final_model.pt")
    return {"model": model, "history": history, "epochs": int(epochs)}
