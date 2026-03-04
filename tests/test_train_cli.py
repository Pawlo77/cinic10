"""Pytest tests for experiment CLI argument parsing."""

from unittest.mock import patch

import pytest

from cinic10.experiments.run_fewshot import _parse_args as parse_fewshot_args
from cinic10.experiments.run_grid_search import _parse_args as parse_grid_args
from cinic10.experiments.run_nas_two_stage import _parse_args as parse_nas_two_stage_args
from cinic10.experiments.run_train import _parse_args as parse_train_args


def test_train_cli_accepts_known_architecture() -> None:
    """Parser accepts known architecture values."""
    argv = [
        "cinic10-train",
        "--data-root",
        ".",
        "--output-dir",
        "./out",
        "--architecture",
        "nas_cnn",
        "--quiet",
    ]
    with patch("sys.argv", argv):
        args = parse_train_args()
    assert args.architecture == "nas_cnn"
    assert args.quiet is True


def test_train_cli_rejects_unknown_architecture() -> None:
    """Parser rejects unsupported architecture values."""
    argv = [
        "cinic10-train",
        "--data-root",
        ".",
        "--output-dir",
        "./out",
        "--architecture",
        "unknown_model",
    ]
    with patch("sys.argv", argv), pytest.raises(SystemExit):
        parse_train_args()


def test_grid_cli_parses_seed_and_quiet() -> None:
    """Grid CLI exposes seed selection and quiet mode."""
    argv = [
        "cinic10-grid",
        "--data-root",
        ".",
        "--output-root",
        "./out",
        "--seed",
        "3407",
        "--quiet",
    ]
    with patch("sys.argv", argv):
        args = parse_grid_args()
    assert args.seed == 3407
    assert args.quiet is True


def test_fewshot_cli_parses_quiet() -> None:
    """Few-shot CLI supports quiet mode."""
    argv = [
        "cinic10-fewshot",
        "--data-root",
        ".",
        "--output-dir",
        "./out",
        "--quiet",
    ]
    with patch("sys.argv", argv):
        args = parse_fewshot_args()
    assert args.quiet is True


def test_nas_two_stage_cli_parses_quiet() -> None:
    """Two-stage NAS CLI supports quiet mode."""
    argv = [
        "cinic10-nas-two-stage",
        "--data-root",
        ".",
        "--output-root",
        "./out",
        "--quiet",
    ]
    with patch("sys.argv", argv):
        args = parse_nas_two_stage_args()
    assert args.quiet is True
