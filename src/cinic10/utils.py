"""Common utility helpers for reproducible experiments."""

import json
import logging
import random
import resource
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from safetensors.torch import save_file as save_safetensors_file

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds across libraries.

    Args:
        seed: Seed value.
    """
    logger.debug("set_seed: %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return the same path.

    Args:
        path: Target directory.

    Returns:
        Created or existing directory path.
    """
    logger.debug("ensure_dir: %s", path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, data: Any) -> None:
    """Write JSON file with UTF-8 encoding.

    Args:
        path: Destination file path.
        data: Serializable object.
    """
    serializable: Any = asdict(data) if is_dataclass(data) else data
    logger.info("Writing JSON to %s", path)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def atomic_torch_save(data: Any, path: Path) -> None:
    """Persist torch-serializable object using atomic rename.

    Args:
        data: Serializable object.
        path: Destination file path.
    """
    logger.info("atomic_torch_save -> %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(data, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def save_model_weights_optimized(model: torch.nn.Module, path: Path) -> Path:
    """Save model weights in compact format suitable for versioning.

    Args:
        model: Model to export.
        path: Destination path, typically with `.safetensors` suffix.

    Returns:
        Actual path written.
    """
    logger.info("save_model_weights_optimized -> %s", path)
    state_dict = {
        name: tensor.detach().cpu().to(torch.float16)
        if tensor.is_floating_point()
        else tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }

    save_safetensors_file(state_dict, str(path))
    return path


def pick_device(device_hint: str) -> torch.device:
    """Resolve target torch device from hint.

    Args:
        device_hint: Preferred device name.

    Returns:
        Resolved torch device.
    """
    logger.debug("pick_device: hint=%s", device_hint)
    if device_hint == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_hint == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    """Synchronize pending accelerator work for accurate timing."""
    logger.debug("synchronize_device: %s", device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        return
    if device.type == "mps" and torch.backends.mps.is_available():
        mps_module = getattr(torch, "mps", None)
        synchronize = getattr(mps_module, "synchronize", None)
        if callable(synchronize):
            synchronize()


def reset_device_peak_memory_stats(device: torch.device) -> None:
    """Reset device peak memory counters when available."""
    logger.debug("reset_device_peak_memory_stats: %s", device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def process_memory_snapshot() -> dict[str, int | None]:
    """Collect process RAM usage snapshot in bytes."""
    current_rss: int | None = None
    current_rss = int(psutil.Process().memory_info().rss)

    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_raw = int(usage.ru_maxrss)
    max_rss_bytes = max_rss_raw if sys.platform == "darwin" else max_rss_raw * 1024
    snapshot = {
        "ram_current_bytes": current_rss,
        "ram_peak_bytes": max_rss_bytes,
    }
    logger.debug("process_memory_snapshot: %s", snapshot)
    return snapshot


def device_memory_snapshot(device: torch.device) -> dict[str, int | None]:
    """Collect accelerator memory usage snapshot in bytes when available."""
    snapshot: dict[str, int | None] = {
        "device_memory_current_bytes": None,
        "device_memory_peak_bytes": None,
        "device_memory_reserved_current_bytes": None,
        "device_memory_reserved_peak_bytes": None,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        snapshot["device_memory_current_bytes"] = int(torch.cuda.memory_allocated(device))
        snapshot["device_memory_peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
        snapshot["device_memory_reserved_current_bytes"] = int(torch.cuda.memory_reserved(device))
        snapshot["device_memory_reserved_peak_bytes"] = int(torch.cuda.max_memory_reserved(device))
        return snapshot

    if device.type == "mps" and torch.backends.mps.is_available():
        mps_module = getattr(torch, "mps", None)
        current_allocated = getattr(mps_module, "current_allocated_memory", None)
        driver_allocated = getattr(mps_module, "driver_allocated_memory", None)
        if callable(current_allocated):
            snapshot["device_memory_current_bytes"] = int(current_allocated())
            snapshot["device_memory_peak_bytes"] = int(current_allocated())
        if callable(driver_allocated):
            snapshot["device_memory_reserved_current_bytes"] = int(driver_allocated())
            snapshot["device_memory_reserved_peak_bytes"] = int(driver_allocated())

    logger.debug("device_memory_snapshot (%s): %s", device, snapshot)
    return snapshot


def wall_time_seconds() -> float:
    """Return monotonic wall time in seconds."""
    return time.perf_counter()


def cpu_time_seconds() -> float:
    """Return process CPU time in seconds."""
    return time.process_time()
