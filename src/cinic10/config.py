"""Typed configuration objects for CINIC-10 experiments."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ArchitectureName = Literal[
    "nas_cnn",
    "mobilenet_v3_small",
    "squeezenet1_0",
    "resnet18",
    "densenet121",
    "convkan_mobilenet_v3_small",
    "convkan_squeezenet1_0",
]
"""Identifier for model architecture."""

OptimizerName = Literal["sgd", "adamw"]
"""Identifier for optimizer type."""

AugmentationMode = Literal[
    "none",
    "standard",
    "standard_mixup",
    "standard_cutmix",
    "autoaugment",
]
"""Identifier for supervised augmentation strategy."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for supervised training.

    Attributes:
        data_root: Directory containing CINIC-10 splits.
        output_dir: Directory where metrics and checkpoints are saved.
        architecture: Model architecture name.
        seed: Random seed.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial optimizer learning rate.
        optimizer: Optimizer name.
        momentum: Momentum used by SGD.
        weight_decay: Weight decay regularization coefficient.
        dropout: Dropout probability for custom heads.
        num_workers: Number of dataloader workers.
        augmentation: Data augmentation strategy.
        mix_alpha: Alpha parameter for Beta distribution used in MixUp/CutMix.
        pretrained: Whether to initialize torchvision models with pretrained weights.
        train_fraction: Fraction of train split to use, e.g. 0.05 for reduction analysis.
        checkpoint_interval: Save resume checkpoint every N epochs.
        nas_entropy_weight: Weight for architecture entropy regularization (NAS models).
        nas_temperature_start: Initial NAS softmax temperature.
        nas_temperature_end: Final NAS softmax temperature at last epoch.
        arch_learning_rate: Learning rate for NAS architecture parameters.
        arch_weight_decay: Weight decay for NAS architecture parameters.
        device: Target device string, e.g. "cpu", "cuda", "mps".
    """

    data_root: Path
    output_dir: Path
    architecture: ArchitectureName = "mobilenet_v3_small"
    seed: int = 42
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 3e-4
    optimizer: OptimizerName = "adamw"
    momentum: float = 0.9
    weight_decay: float = 1e-4
    dropout: float = 0.1
    num_workers: int = 4
    augmentation: AugmentationMode = "standard"
    mix_alpha: float = 1.0
    pretrained: bool = True
    train_fraction: float = 1.0
    checkpoint_interval: int = 1
    nas_entropy_weight: float = 1e-3
    nas_temperature_start: float = 5.0
    nas_temperature_end: float = 0.5
    arch_learning_rate: float = 3e-4
    arch_weight_decay: float = 0.0
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class FewShotConfig:
    """Configuration for episodic few-shot training.

    Attributes:
        data_root: Directory containing CINIC-10 splits.
        output_dir: Directory where outputs are saved.
        seed: Random seed.
        ways: Number of classes per episode (N-way).
        shots: Number of support examples per class (K-shot).
        queries: Number of query examples per class.
        episodes: Number of training episodes.
        eval_episodes: Number of evaluation episodes.
        eval_interval: Evaluate every N training episodes.
        checkpoint_interval: Save resume checkpoint every N training episodes.
        learning_rate: Optimizer learning rate.
        embedding_dim: Feature dimension of encoder output.
        device: Target device string.
    """

    data_root: Path
    output_dir: Path
    seed: int = 42
    ways: int = 5
    shots: int = 5
    queries: int = 15
    episodes: int = 2000
    eval_episodes: int = 400
    eval_interval: int = 100
    checkpoint_interval: int = 100
    learning_rate: float = 1e-3
    embedding_dim: int = 128
    device: str = "cpu"


def build_mobilenet_grid(
    data_root: Path,
    output_root: Path,
    seed: int,
    epochs: int = 30,
) -> list[TrainingConfig]:
    """Build a single-seed MobileNetV3-Small hyperparameter grid.

    Args:
        data_root: Path to CINIC-10 root.
        output_root: Directory where each run subdirectory is created.
        seed: Random seed used for all runs in this grid invocation.
        epochs: Number of epochs per run.

    Returns:
        List of 24 training configurations for the provided seed.
    """
    optimizers: tuple[OptimizerName, ...] = ("sgd", "adamw")
    batch_sizes: tuple[int, ...] = (128, 256)
    dropouts: tuple[float, ...] = (0.0, 0.1, 0.5)
    decays: tuple[float, ...] = (0.0, 1e-4)

    logger.info("build_mobilenet_grid: data_root=%s seed=%d epochs=%d", data_root, seed, epochs)
    runs: list[TrainingConfig] = []
    run_id: int = 0
    for optimizer in optimizers:
        for batch_size in batch_sizes:
            for dropout in dropouts:
                for weight_decay in decays:
                    run_dir = output_root / f"run_{run_id:03d}"
                    runs.append(
                        TrainingConfig(
                            data_root=data_root,
                            output_dir=run_dir,
                            architecture="mobilenet_v3_small",
                            seed=seed,
                            epochs=epochs,
                            batch_size=batch_size,
                            optimizer=optimizer,
                            dropout=dropout,
                            weight_decay=weight_decay,
                        )
                    )
                    run_id += 1
    return runs
