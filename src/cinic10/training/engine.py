"""Training and evaluation loops."""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cinic10.config import AugmentationMode, TrainingConfig
from cinic10.utils import (
    atomic_torch_save,
    cpu_time_seconds,
    device_memory_snapshot,
    dump_json,
    ensure_dir,
    process_memory_snapshot,
    reset_device_peak_memory_stats,
    save_model_weights_optimized,
    synchronize_device,
    wall_time_seconds,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Metrics captured for one epoch.

    Attributes:
        loss: Mean loss over dataset.
        accuracy: Classification accuracy in range [0, 1].
    """

    loss: float
    accuracy: float


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    progress_desc: str = "eval",
    verbose: bool = True,
) -> EpochMetrics:
    """Evaluate model on a dataloader.

    Args:
        model: Evaluated model.
        dataloader: Validation/test dataloader.
        criterion: Loss function.
        device: Computation device.

    Returns:
        Aggregated epoch metrics.
    """
    model.eval()
    logger.debug("evaluate: %s batches=%d", getattr(dataloader, "__len__", lambda: -1)(), 0)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in tqdm(
            dataloader,
            desc=progress_desc,
            leave=False,
            disable=not verbose,
        ):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            # Handle both hard labels (1D) and soft labels (2D from MixUp/CutMix)
            labels_for_accuracy = labels.argmax(dim=1) if labels.ndim > 1 else labels
            batch_size = labels_for_accuracy.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels_for_accuracy).sum().item())
            total_examples += batch_size

    return EpochMetrics(
        loss=total_loss / max(1, total_examples),
        accuracy=total_correct / max(1, total_examples),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    augmentation: AugmentationMode,
    nas_entropy_weight: float,
    verbose: bool = True,
    architecture_optimizer: Optimizer | None = None,
    architecture_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> EpochMetrics:
    """Run one training epoch.

    Args:
        model: Trained model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Computation device.
        augmentation: Active augmentation strategy.
        nas_entropy_weight: Coefficient for architecture entropy regularization.
        architecture_optimizer: Optional optimizer for NAS architecture parameters.
        architecture_dataloader: Optional validation loader for bilevel NAS updates.

    Returns:
        Aggregated training metrics.
    """
    model.train()
    logger.debug(
        "train_one_epoch: model=%s augmentation=%s",
        model.__class__.__name__,
        augmentation,
    )
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # list of architecture parameters to exclude from optimizer step if using bilevel NAS
    architecture_params: list[torch.nn.Parameter] = []
    if hasattr(model, "architecture_parameters") and callable(model.architecture_parameters):
        architecture_params = [p for p in model.architecture_parameters() if p.requires_grad]

    # iterator for architecture updates in bilevel NAS;
    # if enabled, one val batch is used for architecture step per train batch
    architecture_iter: Any = None
    use_bilevel = architecture_optimizer is not None and architecture_dataloader is not None
    if use_bilevel:
        architecture_iter = iter(architecture_dataloader)

    for images, labels in tqdm(dataloader, desc="train", leave=False, disable=not verbose):
        images = images.to(device)
        labels = labels.to(device)

        # If using bilevel NAS, perform architecture update
        # on a validation batch before each training step
        if use_bilevel and architecture_iter is not None:
            try:
                val_images, val_labels = next(architecture_iter)
            except StopIteration:  # reached end of architecture dataloader, restart for next epoch
                architecture_iter = iter(architecture_dataloader)
                val_images, val_labels = next(architecture_iter)

            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            # clear architecture gradients before backward pass
            architecture_optimizer.zero_grad(set_to_none=True)

            val_logits = model(val_images)
            architecture_loss = criterion(val_logits, val_labels)

            # Optionally add entropy regularization for NAS models to encourage exploration
            if nas_entropy_weight > 0.0 and hasattr(model, "architecture_entropy_loss"):
                entropy_loss = model.architecture_entropy_loss()
                if isinstance(entropy_loss, torch.Tensor):
                    architecture_loss = architecture_loss + nas_entropy_weight * entropy_loss

            architecture_loss.backward()
            architecture_optimizer.step()

        # zero training gradients after architecture step to avoid interference
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        # Optionally add entropy regularization for NAS models to encourage exploration
        if nas_entropy_weight > 0.0 and hasattr(model, "architecture_entropy_loss"):
            entropy_loss = model.architecture_entropy_loss()
            if isinstance(entropy_loss, torch.Tensor):
                loss = loss + nas_entropy_weight * entropy_loss

        loss.backward()
        # exclude architecture parameters
        # from optimizer step if using bilevel NAS
        for parameter in architecture_params:
            parameter.grad = None
        optimizer.step()

        labels_for_accuracy = labels.argmax(dim=1) if labels.ndim > 1 else labels
        batch_size = labels_for_accuracy.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels_for_accuracy).sum().item())
        total_examples += batch_size

    return EpochMetrics(
        loss=total_loss / max(1, total_examples),
        accuracy=total_correct / max(1, total_examples),
    )


def _capture_rng_state() -> dict[str, Any]:
    """Capture RNG states for reproducible resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured with `_capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    architecture_optimizer: Optimizer | None,
    metrics: EpochMetrics,
    epoch: int,
    best_val_acc: float,
    status: str,
) -> None:
    """Save full training state for resume and reproducibility.

    Args:
        path: Destination checkpoint path.
        model: Model to serialize.
        optimizer: Optimizer state to serialize.
        scheduler: Scheduler state to serialize.
        metrics: Validation metrics.
        epoch: Epoch number.
        best_val_acc: Best validation accuracy so far.
        status: Run status, e.g. `running`, `interrupted`, `completed`.
    """
    logger.info("Saving checkpoint %s status=%s epoch=%d", path, status, epoch)
    atomic_torch_save(
        {
            "epoch": epoch,
            "status": status,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "architecture_optimizer_state_dict": (
                architecture_optimizer.state_dict() if architecture_optimizer is not None else None
            ),
            "val_loss": metrics.loss,
            "val_accuracy": metrics.accuracy,
            "best_val_accuracy": best_val_acc,
            "rng_state": _capture_rng_state(),
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    architecture_optimizer: Optimizer | None,
) -> tuple[int, EpochMetrics, float]:
    """Load training state from a checkpoint.

    Args:
        path: Checkpoint path.
        model: Model to load into.
        optimizer: Optimizer to restore.
        scheduler: Scheduler to restore.

    Returns:
        Last completed epoch, last epoch metrics, and best validation accuracy.
    """
    logger.info("Loading checkpoint %s", path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    architecture_optimizer_state = state.get("architecture_optimizer_state_dict")
    if architecture_optimizer is not None and architecture_optimizer_state is not None:
        architecture_optimizer.load_state_dict(architecture_optimizer_state)
    if "rng_state" in state:
        _restore_rng_state(state["rng_state"])

    metrics = EpochMetrics(
        loss=float(state.get("val_loss", float("inf"))),
        accuracy=float(state.get("val_accuracy", 0.0)),
    )
    return int(state["epoch"]), metrics, float(state.get("best_val_accuracy", 0.0))


def fit(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    device: torch.device,
    resume: bool = False,
    verbose: bool = True,
    architecture_optimizer: Optimizer | None = None,
    architecture_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, float]:
    """Train model and persist run artifacts.

    Args:
        model: Model to train.
        train_loader: Train dataloader.
        val_loader: Validation dataloader.
        optimizer: Optimizer.
        scheduler: Learning-rate scheduler.
        config: Run configuration.
        device: Computation device.
        resume: Resume from `last.ckpt` in output directory if available.
        architecture_optimizer: Optional architecture optimizer used for bilevel NAS.
        architecture_loader: Optional validation loader for architecture updates.

    Returns:
        Final metrics dictionary.
    """
    ensure_dir(config.output_dir)
    logger.info(
        "fit: starting run output=%s epochs=%d device=%s", config.output_dir, config.epochs, device
    )
    # CrossEntropyLoss with reduction='mean' handles both hard labels (integers)
    # and soft labels (probability distributions from MixUp/CutMix).
    # With soft targets (2D), it computes KL divergence: -sum(target * log(softmax(input)))
    criterion = nn.CrossEntropyLoss(reduction="mean")
    last_checkpoint_path = config.output_dir / "last.ckpt"
    best_checkpoint_path = config.output_dir / "best.pt"

    best_val_acc = -1.0
    best_metrics = EpochMetrics(loss=float("inf"), accuracy=0.0)
    start_epoch = 1
    resumed_from_epoch = 0
    nas_diagnostics: list[dict[str, Any]] = []
    epoch_metrics: list[dict[str, Any]] = []
    epoch_resource_stats: list[dict[str, Any]] = []

    if resume and last_checkpoint_path.exists():
        resumed_from_epoch, last_metrics, best_val_acc = _load_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            architecture_optimizer,
        )
        best_metrics = last_metrics
        start_epoch = resumed_from_epoch + 1

        if best_checkpoint_path.exists():
            best_state = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
            best_metrics = EpochMetrics(
                loss=float(best_state.get("val_loss", best_metrics.loss)),
                accuracy=float(best_state.get("val_accuracy", best_metrics.accuracy)),
            )
            best_val_acc = max(best_val_acc, best_metrics.accuracy)

        if resumed_from_epoch >= config.epochs:
            results = {
                "best_val_loss": best_metrics.loss,
                "best_val_accuracy": best_val_acc,
                "resumed_from_epoch": float(resumed_from_epoch),
                "completed_epochs": float(resumed_from_epoch),
            }
            dump_json(config.output_dir / "metrics.json", results)
            return results

    last_completed_epoch = start_epoch - 1

    def _nas_temperature(epoch_idx: int) -> float:
        """Compute NAS softmax temperature for given epoch index."""
        if config.epochs <= 1:
            return float(config.nas_temperature_end)
        progress = (epoch_idx - 1) / (config.epochs - 1)
        return float(
            config.nas_temperature_start
            + progress * (config.nas_temperature_end - config.nas_temperature_start)
        )

    try:
        epoch_progress = tqdm(
            range(start_epoch, config.epochs + 1),
            desc="epochs",
            leave=True,
            disable=not verbose,
        )
        for epoch in epoch_progress:
            synchronize_device(device)
            reset_device_peak_memory_stats(device)
            epoch_wall_start = wall_time_seconds()
            epoch_cpu_start = cpu_time_seconds()
            memory_start = process_memory_snapshot()

            # If model supports setting NAS temperature, update it for current epoch
            if hasattr(model, "set_arch_temperature"):
                model.set_arch_temperature(_nas_temperature(epoch))

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                augmentation=config.augmentation,
                nas_entropy_weight=config.nas_entropy_weight,
                verbose=verbose,
                architecture_optimizer=architecture_optimizer,
                architecture_dataloader=architecture_loader,
            )
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                progress_desc="validate",
                verbose=verbose,
            )
            scheduler.step()
            epoch_metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_metrics.loss),
                    "train_accuracy": float(train_metrics.accuracy),
                    "val_loss": float(val_metrics.loss),
                    "val_accuracy": float(val_metrics.accuracy),
                }
            )
            dump_json(config.output_dir / "epoch_metrics.json", epoch_metrics)

            synchronize_device(device)
            epoch_wall_end = wall_time_seconds()
            epoch_cpu_end = cpu_time_seconds()
            memory_end = process_memory_snapshot()
            accelerator_memory = device_memory_snapshot(device)
            epoch_resource_stats.append(
                {
                    "epoch": epoch,
                    "wall_time_seconds": float(epoch_wall_end - epoch_wall_start),
                    "cpu_time_seconds": float(epoch_cpu_end - epoch_cpu_start),
                    "ram_current_bytes_start": memory_start["ram_current_bytes"],
                    "ram_current_bytes_end": memory_end["ram_current_bytes"],
                    "ram_peak_bytes_end": memory_end["ram_peak_bytes"],
                    "device_type": device.type,
                    **accelerator_memory,
                }
            )
            dump_json(config.output_dir / "epoch_resource_stats.json", epoch_resource_stats)

            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                best_metrics = val_metrics
                _save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    architecture_optimizer,
                    val_metrics,
                    epoch,
                    best_val_acc,
                    status="best",
                )
                save_model_weights_optimized(
                    model,
                    config.output_dir / "best.safetensors",
                )
                logger.info(
                    "New best checkpoint at epoch %d best_val_acc=%.4f", epoch, best_val_acc
                )

            if epoch % max(1, config.checkpoint_interval) == 0:
                _save_checkpoint(
                    last_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    architecture_optimizer,
                    val_metrics,
                    epoch,
                    best_val_acc,
                    status="running",
                )
                logger.debug("Saved running checkpoint for epoch %d", epoch)

            # If model has architecture diagnostics (e.g. for NAS),
            # capture and persist them after each epoch
            if hasattr(model, "architecture_diagnostics") and callable(
                model.architecture_diagnostics
            ):
                diagnostics = model.architecture_diagnostics()
                mean_entropy = 0.0
                if diagnostics:
                    mean_entropy = float(
                        sum(float(edge["entropy"]) for edge in diagnostics) / len(diagnostics)
                    )
                nas_diagnostics.append(
                    {
                        "epoch": epoch,
                        "temperature": _nas_temperature(epoch),
                        "mean_entropy": mean_entropy,
                        "edges": diagnostics,
                    }
                )
                dump_json(config.output_dir / "nas_diagnostics.json", nas_diagnostics)

            if verbose:
                epoch_progress.set_postfix(
                    {
                        "train_loss": f"{train_metrics.loss:.4f}",
                        "train_acc": f"{train_metrics.accuracy:.4f}",
                        "val_loss": f"{val_metrics.loss:.4f}",
                        "val_acc": f"{val_metrics.accuracy:.4f}",
                        "wall_s": f"{epoch_resource_stats[-1]['wall_time_seconds']:.2f}",
                    }
                )
            last_completed_epoch = epoch

    except KeyboardInterrupt:
        _save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            architecture_optimizer,
            best_metrics,
            max(last_completed_epoch, 0),
            best_val_acc,
            status="interrupted",
        )
        logger.warning("Training interrupted by user at epoch %d", last_completed_epoch)
        raise

    _save_checkpoint(
        last_checkpoint_path,
        model,
        optimizer,
        scheduler,
        architecture_optimizer,
        best_metrics,
        config.epochs,
        best_val_acc,
        status="completed",
    )

    logger.info("fit: completed run output=%s best_val_acc=%.4f", config.output_dir, best_val_acc)

    results = {
        "best_val_loss": best_metrics.loss,
        "best_val_accuracy": best_metrics.accuracy,
        "resumed_from_epoch": float(resumed_from_epoch),
        "completed_epochs": float(config.epochs),
    }

    # If model supports reporting selected architecture (e.g. for NAS),
    # capture and persist it at the end of training
    if hasattr(model, "selected_architecture") and callable(model.selected_architecture):
        selected_ops = model.selected_architecture()
        selected_operation_names: list[str] = []
        if hasattr(model, "selected_operation_names") and callable(model.selected_operation_names):
            selected_operation_names = model.selected_operation_names()
        dump_json(
            config.output_dir / "architecture.json",
            {
                "selected_operations": selected_ops,
                "selected_operation_names": selected_operation_names,
            },
        )

    if nas_diagnostics:
        dump_json(config.output_dir / "nas_diagnostics.json", nas_diagnostics)

    dump_json(config.output_dir / "metrics.json", results)
    return results
