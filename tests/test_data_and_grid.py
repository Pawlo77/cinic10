"""Pytest tests for dataset resolution and grid configuration."""

from pathlib import Path

from cinic10.config import build_mobilenet_grid
from cinic10.data import DatasetResolver, _resolve_split_path


def test_build_mobilenet_grid_single_seed_size_and_seed(tmp_path: Path) -> None:
    """Single-seed grid should contain 24 runs and preserve provided seed."""
    runs = build_mobilenet_grid(
        data_root=tmp_path / "data",
        output_root=tmp_path / "outputs",
        seed=3407,
        epochs=5,
    )

    assert len(runs) == 24
    assert all(run.seed == 3407 for run in runs)
    assert all(run.epochs == 5 for run in runs)


def test_dataset_resolver_accepts_valid_alias(tmp_path: Path) -> None:
    """Resolver should accept 'valid' as validation split alias."""
    (tmp_path / "train").mkdir(parents=True)
    (tmp_path / "valid").mkdir(parents=True)
    (tmp_path / "test").mkdir(parents=True)

    resolver = DatasetResolver()
    assert resolver.has_expected_splits(tmp_path)


def test_resolve_split_path_maps_validate_to_valid(tmp_path: Path) -> None:
    """Logical validate split should resolve to existing valid directory."""
    (tmp_path / "train").mkdir(parents=True)
    (tmp_path / "valid").mkdir(parents=True)
    (tmp_path / "test").mkdir(parents=True)

    assert _resolve_split_path(tmp_path, "validate") == tmp_path / "valid"
    assert _resolve_split_path(tmp_path, "train") == tmp_path / "train"
