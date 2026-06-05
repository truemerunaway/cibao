from __future__ import annotations

from typing import Any

import numpy as np


def raw_delta(raw: np.ndarray, baseline_minutes: int = 60) -> np.ndarray:
    """Subtract each channel's median over the leading baseline interval."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("raw must have shape [channels, time]")
    baseline_length = min(max(int(baseline_minutes), 1), raw.shape[1])
    baseline = np.median(raw[:, :baseline_length], axis=1, keepdims=True)
    return (raw - baseline).astype(np.float32, copy=False)


def raw_diff(raw: np.ndarray) -> np.ndarray:
    """Return the first temporal difference with a zero-valued first sample."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("raw must have shape [channels, time]")
    result = np.zeros_like(raw, dtype=np.float32)
    result[:, 1:] = np.diff(raw, axis=1)
    return result


def build_time_domain_input(
    raw: np.ndarray,
    baseline_minutes: int = 60,
    include_delta: bool = True,
    include_diff: bool = True,
) -> np.ndarray:
    channels: list[np.ndarray] = []
    if include_delta:
        channels.append(raw_delta(raw, baseline_minutes))
    if include_diff:
        channels.append(raw_diff(raw))
    if not channels:
        channels.append(np.asarray(raw, dtype=np.float32))
    return np.concatenate(channels, axis=0).astype(np.float32, copy=False)


def welch_log_psd(
    signal: np.ndarray,
    nperseg: int = 128,
    noverlap: int = 64,
    sampling_interval_minutes: float = 1.0,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import welch
    except ImportError as exc:
        raise ImportError("SciPy is required when PSD features are enabled.") from exc

    signal = np.asarray(signal, dtype=np.float32)
    sampling_frequency = 1.0 / sampling_interval_minutes
    segment_length = min(int(nperseg), signal.shape[-1])
    overlap = min(int(noverlap), max(segment_length - 1, 0))
    frequencies, power = welch(
        signal,
        fs=sampling_frequency,
        nperseg=segment_length,
        noverlap=overlap,
        axis=-1,
    )
    return frequencies.astype(np.float32), np.log10(power + epsilon).astype(np.float32)


def morlet_cwt_power(
    signal: np.ndarray,
    wavelet: str = "morl",
    num_scales: int = 32,
    min_period_minutes: float = 4,
    max_period_minutes: float = 180,
    sampling_interval_minutes: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("PyWavelets is required when CWT features are enabled.") from exc

    signal = np.asarray(signal, dtype=np.float32)
    periods = np.geomspace(min_period_minutes, max_period_minutes, int(num_scales))
    central_frequency = pywt.central_frequency(wavelet)
    scales = central_frequency * periods / sampling_interval_minutes
    channel_power: list[np.ndarray] = []

    for channel in signal:
        coefficients, _ = pywt.cwt(
            channel,
            scales,
            wavelet,
            sampling_period=sampling_interval_minutes,
        )
        channel_power.append(np.log1p(np.abs(coefficients) ** 2).astype(np.float32))

    return periods.astype(np.float32), np.stack(channel_power, axis=0)


def build_representations(
    raw: np.ndarray,
    feature_config: dict[str, Any],
    baseline_minutes: int,
) -> dict[str, np.ndarray]:
    time_domain = build_time_domain_input(
        raw,
        baseline_minutes=baseline_minutes,
        include_delta=bool(feature_config.get("raw_delta", True)),
        include_diff=bool(feature_config.get("raw_diff", True)),
    )
    result: dict[str, np.ndarray] = {"time_domain": time_domain}

    psd_config = feature_config.get("psd", {})
    if psd_config.get("enabled", False):
        frequencies, power = welch_log_psd(
            raw_delta(raw, baseline_minutes),
            nperseg=int(psd_config.get("nperseg", 128)),
            noverlap=int(psd_config.get("noverlap", 64)),
        )
        result["psd_frequencies"] = frequencies
        result["psd"] = power

    cwt_config = feature_config.get("cwt", {})
    if cwt_config.get("enabled", False):
        periods, power = morlet_cwt_power(
            raw_delta(raw, baseline_minutes),
            wavelet=str(cwt_config.get("wavelet", "morl")),
            num_scales=int(cwt_config.get("num_scales", 32)),
            min_period_minutes=float(cwt_config.get("min_period_minutes", 4)),
            max_period_minutes=float(cwt_config.get("max_period_minutes", 180)),
        )
        result["cwt_periods"] = periods
        result["cwt"] = power
    return result
