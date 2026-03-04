"""NAS CNN architecture for CINIC-10."""

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _build_edge_operation(
    name: str,
    in_channels: int,
    out_channels: int,
    stride: int,
) -> nn.Module:
    """Build one candidate operation for a searchable edge.

    Args:
        name: Candidate operation name.
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Spatial stride.

    Returns:
        Instantiated operation module.
    """
    if name == "conv3x3":
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
    if name == "conv5x5":
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                padding=2,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
    if name == "depthwise3x3":
        return _DepthwiseSeparableConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
        )
    if name == "maxpool3x3_proj":
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
            _Projection(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
            ),
        )
    if name == "skip_proj":
        return _Projection(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
        )
    raise ValueError(f"Unsupported NAS edge operation: {name}")


class _DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution block with configurable stride."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        """Initialize block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
        """
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.net(x)


class _Projection(nn.Module):
    """Projection layer used for shape-aligned skip connections."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Initialize projection layer.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            stride: Spatial stride.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply projection.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.net(x)


class _MixedOpEdge(nn.Module):
    """Differentiable NAS edge with learnable operation mixture."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Initialize candidate operations for one searchable edge.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            stride: Edge stride.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.choice_names: tuple[str, ...] = (
            "conv3x3",
            "conv5x5",
            "depthwise3x3",
            "maxpool3x3_proj",
            "skip_proj",
        )
        self.choices = nn.ModuleList(
            [
                _build_edge_operation(
                    name=choice_name,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
                for choice_name in self.choice_names
            ]
        )
        self.arch_logits = nn.Parameter(torch.zeros(len(self.choice_names), dtype=torch.float32))
        self.temperature = 1.0

    def set_temperature(self, temperature: float) -> None:
        """Set softmax temperature for architecture mixing.

        Args:
            temperature: Positive temperature value.
        """
        self.temperature = max(1e-6, float(temperature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of candidate operations.

        Args:
            x: Input tensor.

        Returns:
            Mixed output tensor.
        """
        weights = torch.softmax(self.arch_logits / self.temperature, dim=0)
        sample = self.choices[0](x)
        mixed = torch.zeros_like(sample)
        for idx, op in enumerate(self.choices):
            mixed = mixed + weights[idx] * op(x)
        return mixed

    def best_operation(self) -> str:
        """Return currently selected operation name (argmax)."""
        best_idx = int(torch.argmax(self.arch_logits).item())
        return self.choice_names[best_idx]


class NasCnn(nn.Module):
    """Differentiable NAS supernet with fully searched backbone.

    The feature extractor is composed only of searchable edges. Each edge learns
    a weighted mixture of candidate operations, so channel transitions and
    downsampling decisions are optimized end-to-end together with weights.

    Args:
        num_classes: Number of target classes.
        dropout: Dropout probability in classifier.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.1,
        arch_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        logger.info("NasCnn: init num_classes=%d dropout=%s", num_classes, dropout)
        edge_specs: tuple[tuple[int, int, int], ...] = (
            (3, 32, 1),
            (32, 32, 1),
            (32, 64, 2),
            (64, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
        )
        self.search_edges = nn.ModuleList(
            [
                _MixedOpEdge(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
                for in_channels, out_channels, stride in edge_specs
            ]
        )
        self.set_arch_temperature(arch_temperature)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor shaped `(batch, channels, height, width)`.

        Returns:
            Class logits tensor shaped `(batch, num_classes)`.
        """
        features = x
        for edge in self.search_edges:
            features = edge(features)
        features = self.pool(features)
        return self.classifier(features)

    def selected_architecture(self) -> list[str]:
        """Get currently selected operation for each NAS block.

        Returns:
            List of operation names chosen by argmax architecture weights.
        """
        selected: list[str] = []
        for idx, edge in enumerate(self.search_edges):
            op = edge.best_operation()
            edge_desc = (
                f"edge_{idx}: {op} "
                f"(in={edge.in_channels}, out={edge.out_channels}, stride={edge.stride})"
            )
            selected.append(edge_desc)
        logger.debug("selected_architecture: %s", selected)
        return selected

    def selected_operation_names(self) -> list[str]:
        """Return only selected operation names in edge order.

        Returns:
            Ordered operation names selected by argmax per edge.
        """
        return [edge.best_operation() for edge in self.search_edges]

    def architecture_parameters(self) -> list[nn.Parameter]:
        """Return trainable NAS architecture parameters.

        Returns:
            List of architecture logits tensors.
        """
        return [edge.arch_logits for edge in self.search_edges]

    def set_arch_temperature(self, temperature: float) -> None:
        """Set architecture softmax temperature for all searchable edges.

        Args:
            temperature: Positive temperature value.
        """
        for edge in self.search_edges:
            edge.set_temperature(temperature)

    def architecture_entropy_loss(self) -> torch.Tensor:
        """Compute mean entropy of edge architecture distributions.

        Lower entropy encourages sharper one-operation selections.

        Returns:
            Scalar entropy tensor.
        """
        entropies: list[torch.Tensor] = []
        for edge in self.search_edges:
            probs = torch.softmax(edge.arch_logits / edge.temperature, dim=0)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
            entropies.append(entropy)
        return torch.stack(entropies).mean()

    def architecture_diagnostics(self) -> list[dict[str, Any]]:
        """Return per-edge architecture diagnostics.

        Returns:
            List of dictionaries with top operation probability and entropy.
        """
        diagnostics: list[dict[str, Any]] = []
        for idx, edge in enumerate(self.search_edges):
            probs = torch.softmax(edge.arch_logits / edge.temperature, dim=0)
            top_index = int(torch.argmax(probs).item())
            entropy = float((-(probs * torch.log(probs.clamp_min(1e-8))).sum()).item())
            diagnostics.append(
                {
                    "edge": idx,
                    "top_operation": edge.choice_names[top_index],
                    "top_probability": float(probs[top_index].item()),
                    "entropy": entropy,
                    "temperature": float(edge.temperature),
                }
            )
        return diagnostics


class DiscreteNasCnn(nn.Module):
    """Discrete CNN materialized from searched NAS operations."""

    def __init__(
        self,
        selected_operations: list[str],
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        """Initialize discrete NAS model.

        Args:
            selected_operations: One operation name per edge.
            num_classes: Number of target classes.
            dropout: Classifier dropout probability.
        """
        super().__init__()
        edge_specs: tuple[tuple[int, int, int], ...] = (
            (3, 32, 1),
            (32, 32, 1),
            (32, 64, 2),
            (64, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
        )
        if len(selected_operations) != len(edge_specs):
            raise ValueError(
                "selected_operations length must match the number of NAS edges "
                f"({len(edge_specs)}), got {len(selected_operations)}"
            )

        self.backbone = nn.Sequential(
            *[
                _build_edge_operation(
                    name=selected_operations[idx],
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
                for idx, (in_channels, out_channels, stride) in enumerate(edge_specs)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for discrete NAS model."""
        features = self.backbone(x)
        features = self.pool(features)
        return self.classifier(features)
