"""Optimizer builders."""

import logging
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import SGD, AdamW, Optimizer

from cinic10.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OptimizerBundle:
    """Container for weight and architecture optimizers.

    Attributes:
        weight_optimizer: Optimizer for model weights.
        architecture_optimizer: Optional optimizer for NAS architecture logits.
    """

    weight_optimizer: Optimizer
    architecture_optimizer: Optimizer | None


def _split_model_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split model parameters into regular weights and NAS architecture params.

    Args:
        model: Model to inspect.

    Returns:
        A tuple of `(weight_parameters, architecture_parameters)`.
    """
    arch_params: list[nn.Parameter] = []
    if hasattr(model, "architecture_parameters") and callable(model.architecture_parameters):
        arch_params = [p for p in model.architecture_parameters() if p.requires_grad]

    arch_param_ids = {id(param) for param in arch_params}
    weight_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in arch_param_ids
    ]
    return weight_params, arch_params


def _build_optimizer(parameters: list[nn.Parameter], config: TrainingConfig) -> Optimizer:
    """Build configured optimizer for a parameter list.

    Args:
        parameters: Parameters to optimize.
        config: Training configuration.

    Returns:
        Instantiated optimizer.
    """
    if config.optimizer == "sgd":
        logger.debug("_build_optimizer: using SGD lr=%s", config.learning_rate)
        return SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    logger.debug("_build_optimizer: using AdamW lr=%s", config.learning_rate)
    return AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def create_optimizers(model: nn.Module, config: TrainingConfig) -> OptimizerBundle:
    """Create optimizers for model weights and optional NAS architecture params.

    Args:
        model: Model parameters source.
        config: Training configuration.

    Returns:
        Bundle with weight optimizer and optional architecture optimizer.
    """
    weight_params, arch_params = _split_model_parameters(model)
    weight_optimizer = _build_optimizer(weight_params, config)
    architecture_optimizer: Optimizer | None = None

    if arch_params:
        logger.info(
            "create_optimizers: creating architecture optimizer lr=%s", config.arch_learning_rate
        )
        architecture_optimizer = AdamW(
            arch_params,
            lr=config.arch_learning_rate,
            weight_decay=config.arch_weight_decay,
        )

    return OptimizerBundle(
        weight_optimizer=weight_optimizer,
        architecture_optimizer=architecture_optimizer,
    )


def create_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Create optimizer from training config.

    Args:
        model: Model parameters source.
        config: Training configuration.

    Returns:
        Instantiated optimizer.
    """
    return create_optimizers(model, config).weight_optimizer


def create_scheduler(optimizer: Optimizer, epochs: int) -> torch.optim.lr_scheduler.LRScheduler:
    """Create cosine LR scheduler.

    Args:
        optimizer: Optimizer instance.
        epochs: Number of epochs.

    Returns:
        Learning rate scheduler.
    """
    logger.debug("create_scheduler: epochs=%d", epochs)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
