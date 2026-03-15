import math

import torch

from . import convolution
from .kan_linear import KANLinear


class KANConvolutionalLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: tuple[int, int] = (2, 2),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups
        n_grouped_convs = groups * out_channels_per_group * in_channels_per_group

        self.convs = torch.nn.ModuleList()
        for _ in range(n_grouped_convs):
            self.convs.append(
                KANConvolution(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return convolution.multiple_convs_kan_conv2d(
            x,
            self.convs,
            kernel_size=self.kernel_size,
            out_channels=self.out_channels,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=self.groups,
            device=x.device,
        )


class KANConvolution(torch.nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] = (2, 2),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv = KANLinear(
            in_features=math.prod(kernel_size),
            out_features=1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return convolution.multiple_convs_kan_conv2d(
            x,
            [self],
            kernel_size=self.kernel_size,
            out_channels=1,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=1,
            device=x.device,
        )

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        return self.conv.regularization_loss(regularize_activation, regularize_entropy)
