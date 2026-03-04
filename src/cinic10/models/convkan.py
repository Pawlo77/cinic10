"""ConvKAN conversion utilities powered by `torchkan`."""

import importlib
import logging
from collections.abc import Iterable, Iterator
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)


def _iter_named_children(module: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield immediate named children.

    Args:
        module: Parent module.

    Yields:
        Tuples of child name and child module.
    """
    yield from module.named_children()


def _candidate_torchkan_module_names() -> tuple[str, ...]:
    """Return module paths where torchkan conv layers are commonly exposed.

    Returns:
        Candidate import paths.
    """
    return (
        "torchkan",
        "torchkan.layers",
        "torchkan.convolution",
        "torchkan.conv",
    )


def _candidate_torchkan_class_names() -> tuple[str, ...]:
    """Return likely ConvKAN class names across torchkan versions.

    Returns:
        Candidate class names.
    """
    return (
        "KANConv2d",
        "KANConv2D",
        "KANConv2DLayer",
        "ConvKAN2d",
        "ConvKAN2D",
        "ConvKAN",
    )


def _load_torchkan_conv_class() -> type[nn.Module]:
    """Locate a ConvKAN class from the installed `torchkan` package.

    Returns:
        ConvKAN layer class.

    Raises:
        RuntimeError: If torchkan is missing or no supported class is found.
    """
    loaded_modules: list[Any] = []
    for module_name in _candidate_torchkan_module_names():
        loaded_modules.append(importlib.import_module(module_name))

    for module in loaded_modules:
        for class_name in _candidate_torchkan_class_names():
            cls = getattr(module, class_name, None)
            if isinstance(cls, type) and issubclass(cls, nn.Module):
                logger.debug("Found torchkan Conv class: %s.%s", module.__name__, class_name)
                return cls

    searched_modules = ", ".join(module.__name__ for module in loaded_modules)
    searched_classes = ", ".join(_candidate_torchkan_class_names())
    raise RuntimeError(
        "Installed `torchkan` does not expose a supported ConvKAN layer class. "
        f"Searched modules: {searched_modules}. Searched classes: {searched_classes}."
    )


def _instantiate_torchkan_from_conv(
    conv_layer: nn.Conv2d,
    convkan_cls: type[nn.Module],
) -> nn.Module:
    """Create a torchkan layer from a Conv2d layer configuration.

    Args:
        conv_layer: Source convolution layer.
        convkan_cls: Target torchkan layer class.

    Returns:
        Instantiated torchkan layer.

    Raises:
        RuntimeError: If class constructor signature is incompatible.
    """
    constructor_candidates: Iterable[dict[str, Any]] = (
        {
            "in_channels": conv_layer.in_channels,
            "out_channels": conv_layer.out_channels,
            "kernel_size": conv_layer.kernel_size,
            "stride": conv_layer.stride,
            "padding": conv_layer.padding,
            "dilation": conv_layer.dilation,
            "groups": conv_layer.groups,
            "bias": conv_layer.bias is not None,
        },
        {
            "in_channels": conv_layer.in_channels,
            "out_channels": conv_layer.out_channels,
            "kernel_size": conv_layer.kernel_size,
            "stride": conv_layer.stride,
            "padding": conv_layer.padding,
            "bias": conv_layer.bias is not None,
        },
        {
            "in_channels": conv_layer.in_channels,
            "out_channels": conv_layer.out_channels,
            "kernel_size": conv_layer.kernel_size,
        },
    )

    last_error: Exception | None = None
    for kwargs in constructor_candidates:
        try:
            return convkan_cls(**kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Could not instantiate torchkan layer `{convkan_cls.__name__}` from Conv2d settings."
    ) from last_error


def replace_conv2d_with_convkan(module: nn.Module, min_kernel_size: int = 3) -> nn.Module:
    """Recursively replace Conv2d layers with torchkan ConvKAN layers.

    Args:
        module: Input module to transform.
        min_kernel_size: Replace only convolutions with kernel size >= this threshold.

    Returns:
        Transformed module.

    Raises:
        RuntimeError: If torchkan is unavailable or incompatible.
    """
    convkan_cls = _load_torchkan_conv_class()

    for name, child in _iter_named_children(module):
        if isinstance(child, nn.Conv2d):
            kernel_h, kernel_w = (
                child.kernel_size
                if isinstance(child.kernel_size, tuple)
                else (child.kernel_size, child.kernel_size)
            )
            if kernel_h >= min_kernel_size and kernel_w >= min_kernel_size:
                setattr(
                    module,
                    name,
                    _instantiate_torchkan_from_conv(child, convkan_cls),
                )
                logger.debug(
                    "Replaced Conv2d with ConvKAN in %s.%s", module.__class__.__name__, name
                )
        else:
            replace_conv2d_with_convkan(child, min_kernel_size=min_kernel_size)
    return module
