"""Pytest tests for NAS model helpers."""

from pathlib import Path

import torch

from cinic10.config import TrainingConfig
from cinic10.models.nas_cnn import DiscreteNasCnn, NasCnn
from cinic10.training.optimizer import create_optimizers


def test_architecture_optimizer_split() -> None:
    """Architecture parameters are optimized separately from weights."""
    model = NasCnn(num_classes=10)
    config = TrainingConfig(
        data_root=Path(),
        output_dir=Path(),
        architecture="nas_cnn",
    )

    bundle = create_optimizers(model, config)
    assert bundle.architecture_optimizer is not None

    arch_param_ids = {id(parameter) for parameter in model.architecture_parameters()}
    weight_param_ids = {
        id(parameter)
        for group in bundle.weight_optimizer.param_groups
        for parameter in group["params"]
    }
    assert arch_param_ids
    assert arch_param_ids.isdisjoint(weight_param_ids)


def test_discrete_model_forward() -> None:
    """Discrete model built from searched ops runs a forward pass."""
    operations = [
        "conv3x3",
        "conv5x5",
        "depthwise3x3",
        "maxpool3x3_proj",
        "skip_proj",
        "conv3x3",
    ]
    model = DiscreteNasCnn(selected_operations=operations, num_classes=10)

    inputs = torch.randn(4, 3, 32, 32)
    logits = model(inputs)
    assert logits.shape == (4, 10)


def test_architecture_diagnostics_shape() -> None:
    """Diagnostics contain one record per searchable edge."""
    model = NasCnn(num_classes=10)
    diagnostics = model.architecture_diagnostics()

    assert len(diagnostics) == 6
    for edge in diagnostics:
        assert "top_operation" in edge
        assert "top_probability" in edge
        assert "entropy" in edge
