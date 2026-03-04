"""CINIC-10 dataset and dataloader utilities."""

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import kagglehub
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

SplitName = Literal["train", "validate", "test"]
"""Identifier for dataset split."""

EXPECTED_SPLITS: tuple[str, str, str] = ("train", "validate", "test")
"""Expected CINIC-10 split directory names."""

VALIDATE_SPLIT_ALIASES: tuple[str, str] = ("validate", "valid")
"""Accepted directory aliases for validation split."""

KAGGLE_CINIC10_DATASETS: tuple[str, ...] = ("mengcius/cinic10",)
"""Candidate Kaggle dataset references for auto-download."""


@dataclass(frozen=True, slots=True)
class Batch:
    """Container for a mini-batch.

    Attributes:
        images: Image tensor.
        labels: Label tensor.
    """

    images: torch.Tensor
    labels: torch.Tensor


def _normalize() -> transforms.Normalize:
    """Build normalization transform for CINIC-10/CIFAR-like data.

    Returns:
        Normalization transform.
    """
    return transforms.Normalize(
        mean=(0.47889522, 0.47227842, 0.43047404),
        std=(0.24205776, 0.23828046, 0.25874835),
    )


@dataclass(frozen=True, slots=True)
class DatasetResolver:
    """Resolve CINIC-10 data root with optional Kaggle auto-download.

    Attributes:
        expected_splits: Required CINIC-10 split directory names.
        kaggle_datasets: Candidate Kaggle dataset references to try.
        auto_download_env_var: Environment variable controlling auto-download.
    """

    expected_splits: tuple[str, ...] = EXPECTED_SPLITS
    kaggle_datasets: tuple[str, ...] = KAGGLE_CINIC10_DATASETS
    auto_download_env_var: str = "CINIC10_AUTO_DOWNLOAD"

    def _has_validation_split(self, root: Path) -> bool:
        """Check if root contains a supported validation split directory."""
        return any((root / split_name).is_dir() for split_name in VALIDATE_SPLIT_ALIASES)

    def has_expected_splits(self, root: Path) -> bool:
        """Check whether all expected split folders are present.

        Args:
            root: Candidate dataset root.

        Returns:
            True when all split folders exist.
        """
        return (
            (root / "train").is_dir()
            and (root / "test").is_dir()
            and self._has_validation_split(root)
        )

    def find_cinic10_root(self, base_dir: Path) -> Path | None:
        """Find CINIC-10 root under a directory tree.

        Args:
            base_dir: Directory to inspect recursively.

        Returns:
            Resolved CINIC-10 root path when found, otherwise None.
        """
        if self.has_expected_splits(base_dir):
            return base_dir

        for candidate in base_dir.rglob("*"):
            if candidate.is_dir() and self.has_expected_splits(candidate):
                return candidate
        return None

    def _download_from_kaggle(self, output_dir: Path | None = None) -> Path:
        """Download CINIC-10 with KaggleHub and resolve extracted root.

        When `output_dir` is provided, KaggleHub downloads directly into that
        directory instead of the global `~/.cache` location.

        Returns:
            Directory containing `train`, `validate`, and `test`.

        Raises:
            RuntimeError: If KaggleHub is unavailable or download fails.
        """
        download_errors: list[str] = []
        for dataset_ref in self.kaggle_datasets:
            try:
                download_kwargs: dict[str, str] = {}
                if output_dir is not None:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    download_kwargs["output_dir"] = str(output_dir)
                downloaded_path = Path(kagglehub.dataset_download(dataset_ref, **download_kwargs))
                resolved = self.find_cinic10_root(downloaded_path)
                if resolved is not None:
                    return resolved
                raise RuntimeError(
                    f"Downloaded dataset {dataset_ref} but could "
                    f"not find expected splits under {downloaded_path}"
                )
            except Exception as exc:  # pragma: no cover
                download_errors.append(f"{dataset_ref}: {exc}")

        raise RuntimeError(
            "Unable to auto-download CINIC-10 from Kaggle. "
            f"Tried datasets: {', '.join(self.kaggle_datasets)}. "
            f"Errors: {' | '.join(download_errors)}"
        )

    def resolve(self, data_root: Path) -> Path:
        """Resolve local dataset path or auto-download from Kaggle.

        Args:
            data_root: User-provided path.

        Returns:
            Path that contains CINIC-10 split directories.

        Raises:
            FileNotFoundError: If auto-download is disabled and data is missing.
            RuntimeError: If auto-download is enabled but cannot complete.
        """
        if self.has_expected_splits(data_root):
            logger.debug("DatasetResolver.resolve: found expected splits at %s", data_root)
            return data_root

        env_flag = os.getenv(self.auto_download_env_var, "1").strip().lower()
        if env_flag in {"0", "false", "no"}:
            raise FileNotFoundError(
                f"CINIC-10 splits not found under {data_root}. "
                "Enable auto-download or provide valid --data-root containing train/validate/test."
            )

        return self._download_from_kaggle(output_dir=data_root)


def resolve_data_root(data_root: Path, resolver: DatasetResolver | None = None) -> Path:
    """Resolve CINIC-10 root path using a resolver service.

    Args:
        data_root: User-provided root path.
        resolver: Optional resolver strategy object.

    Returns:
        Path containing required CINIC-10 splits.
    """
    active_resolver = resolver or DatasetResolver()
    return active_resolver.resolve(data_root)


def _resolve_split_path(data_root: Path, split: SplitName) -> Path:
    """Resolve split directory path with support for alias names.

    Args:
        data_root: Root containing dataset splits.
        split: Requested logical split.

    Returns:
        Concrete directory path for the split.

    Raises:
        FileNotFoundError: If no matching split directory exists.
    """
    if split != "validate":
        resolved = data_root / split
        if resolved.is_dir():
            return resolved
        raise FileNotFoundError(f"Split directory not found: {resolved}")

    for candidate in VALIDATE_SPLIT_ALIASES:
        resolved = data_root / candidate
        if resolved.is_dir():
            return resolved

    raise FileNotFoundError(
        f"Validation split directory not found under {data_root}. "
        f"Expected one of: {', '.join(VALIDATE_SPLIT_ALIASES)}"
    )


def build_transforms(train: bool, use_autoaugment: bool) -> transforms.Compose:
    """Create input transform pipeline.

    Args:
        train: Whether transform is for train split.
        use_autoaugment: Whether to apply CIFAR10 AutoAugment policy.

    Returns:
        Composed transform pipeline.
    """
    if train:
        train_ops: list[object] = [
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if use_autoaugment:
            train_ops.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
        train_ops.extend([transforms.ToTensor(), _normalize()])
        return transforms.Compose(train_ops)

    return transforms.Compose([transforms.ToTensor(), _normalize()])


def _sample_indices_per_class(
    labels: list[int],
    fraction: float,
    classes: list[int],
    seed: int,
) -> list[int]:
    """Sample a fraction of indices per class.

    Args:
        labels: Dataset labels by index.
        fraction: Fraction per class in `(0, 1]`.
        classes: Class ids.
        seed: Random seed.

    Returns:
        Sampled index list.
    """
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    labels_arr = np.array(labels)
    for class_idx in classes:
        class_indices = np.where(labels_arr == class_idx)[0]
        take = max(1, int(len(class_indices) * fraction))
        chosen = rng.choice(class_indices, size=take, replace=False)
        selected.extend(chosen.tolist())
    return selected


def maybe_reduce_dataset_per_class(
    dataset: datasets.ImageFolder,
    train_fraction: float,
    seed: int,
) -> Dataset[tuple[torch.Tensor, int]]:
    """Reduce dataset to a per-class fraction.

    Args:
        dataset: Input `ImageFolder` dataset.
        train_fraction: Fraction in `(0, 1]`.
        seed: Random seed.

    Returns:
        Original dataset or reduced subset.
    """
    if train_fraction >= 1.0:
        return dataset
    if train_fraction <= 0.0:
        raise ValueError("train_fraction must be in (0, 1]")

    class_ids = list(range(len(dataset.classes)))
    indices = _sample_indices_per_class(
        labels=dataset.targets,
        fraction=train_fraction,
        classes=class_ids,
        seed=seed,
    )
    return Subset(dataset, indices)


def _seed_worker(worker_id: int) -> None:
    """Seed dataloader worker RNGs deterministically.

    Args:
        worker_id: Worker index provided by PyTorch.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(
    data_root: Path,
    split: SplitName,
    batch_size: int,
    num_workers: int,
    use_autoaugment: bool,
    train_fraction: float,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build one CINIC-10 dataloader.

    Args:
        data_root: Root with `train`, `validate`, and `test` subdirectories.
        split: Dataset split name.
        batch_size: Batch size.
        num_workers: Number of workers.
        use_autoaugment: Use autoaugment on train split.
        train_fraction: Fraction of train data for reduction experiments.
        seed: Random seed.

    Returns:
        Configured dataloader.
    """
    resolved_data_root = resolve_data_root(data_root)
    is_train = split == "train"
    split_path = _resolve_split_path(resolved_data_root, split)
    dataset = datasets.ImageFolder(
        root=str(split_path),
        transform=build_transforms(train=is_train, use_autoaugment=use_autoaugment),
    )
    materialized: Dataset[tuple[torch.Tensor, int]]
    if is_train:
        materialized = maybe_reduce_dataset_per_class(
            dataset=dataset,
            train_fraction=train_fraction,
            seed=seed,
        )
    else:
        materialized = dataset

    split_seed = seed + {"train": 0, "validate": 10_000, "test": 20_000}[split]
    generator = torch.Generator()
    generator.manual_seed(split_seed)

    return DataLoader(
        materialized,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=_seed_worker,
    )
