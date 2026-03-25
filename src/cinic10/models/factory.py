"""Factory for model construction."""

import logging
from collections.abc import Iterator
from typing import cast

from convkan import ConvKAN
from torch import nn
from torchvision.models import (
    DenseNet121_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    SqueezeNet1_0_Weights,
    densenet121,
    mobilenet_v3_small,
    resnet18,
    squeezenet1_0,
)

from cinic10.config import ArchitectureName
from cinic10.models.nas_cnn import NasCnn

logger = logging.getLogger(__name__)


def _iter_named_children(module: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield immediate named children for recursive replacement."""
    yield from module.named_children()


def replace_conv2d_with_convkan(
    module: nn.Module,
    min_kernel_size: int = 3,
) -> nn.Module:
    """Recursively replace Conv2d layers with KAN convolutional layers.

    Args:
        module: Input module to transform.
        min_kernel_size: Replace only convolutions with kernel size >= this threshold.
        max_channels: Optional upper bound for both in/out channels of converted layers.
            If set, larger convolutions are left as standard Conv2d to avoid memory blowups.

    Returns:
        Transformed module.
    """
    for name, child in _iter_named_children(module):
        if isinstance(child, nn.Conv2d):
            kernel_h, kernel_w = (
                child.kernel_size
                if isinstance(child.kernel_size, tuple)
                else (child.kernel_size, child.kernel_size)
            )
            if kernel_h < min_kernel_size or kernel_w < min_kernel_size:
                continue

            setattr(
                module,
                name,
                ConvKAN(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=(kernel_h, kernel_w),
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                ),
            )
            logger.debug("Replaced Conv2d with ConvKAN in %s.%s", module.__class__.__name__, name)
        else:
            replace_conv2d_with_convkan(
                child,
                min_kernel_size=min_kernel_size,
            )
    return module


def create_model(
    architecture: ArchitectureName,
    num_classes: int,
    dropout: float,
    pretrained: bool,
) -> nn.Module:
    """Create model by architecture name.

    Args:
        architecture: Architecture identifier.
        num_classes: Number of output classes.
        dropout: Dropout probability for custom heads.
        pretrained: Whether to use pretrained torchvision weights.

    Returns:
        Initialized model.
    """
    if architecture == "nas_cnn":
        logger.info("create_model: architecture=%s (nas_cnn)", architecture)
        return NasCnn(num_classes=num_classes, dropout=dropout)

    if architecture in {"mobilenet_v3_small", "convkan_mobilenet_v3_small"}:
        model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        logger.info(
            "create_model: mobilenet_v3_small pretrained=%s dropout=%s", pretrained, dropout
        )
        last = cast(nn.Linear, model.classifier[-1])
        model.classifier = nn.Sequential(
            model.classifier[0],
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(last.in_features, num_classes),
        )
        if architecture == "convkan_mobilenet_v3_small":
            replace_conv2d_with_convkan(model.features)
        return model

    if architecture in {"squeezenet1_0", "convkan_squeezenet1_0"}:
        model = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT if pretrained else None)
        logger.info("create_model: squeezenet1_0 pretrained=%s dropout=%s", pretrained, dropout)
        model.classifier = nn.Sequential(
            model.classifier[0],
            nn.Dropout(p=dropout),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )
        if architecture == "convkan_squeezenet1_0":
            replace_conv2d_with_convkan(model.features)
        return model

    if architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        logger.info("create_model: resnet18 pretrained=%s dropout=%s", pretrained, dropout)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    if architecture == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        logger.info("create_model: densenet121 pretrained=%s dropout=%s", pretrained, dropout)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.classifier.in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")
