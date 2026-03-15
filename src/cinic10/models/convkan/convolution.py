# Credits to: https://github.com/detkov/Convolution-From-Scratch/

import numpy as np
import torch


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size, n_channels, n, m = matrix.shape

    kernel_h, kernel_w = (
        kernel_side if isinstance(kernel_side, tuple) else (kernel_side, kernel_side)
    )

    h_out = (
        np.floor(
            (n + 2 * padding[0] - kernel_h - (kernel_h - 1) * (dilation[0] - 1)) / stride[0]
        ).astype(int)
        + 1
    )
    w_out = (
        np.floor(
            (m + 2 * padding[1] - kernel_w - (kernel_w - 1) * (dilation[1] - 1)) / stride[1]
        ).astype(int)
        + 1
    )
    return h_out, w_out, batch_size, n_channels


def multiple_convs_kan_conv2d(
    matrix,  # but as torch tensors. Kernel side asume q el kernel es cuadrado
    kernels,
    kernel_size,
    out_channels,
    stride=(1, 1),
    dilation=(1, 1),
    padding=(0, 0),
    groups=1,
    device="cuda",
) -> torch.Tensor:
    """Apply KAN-based 2D convolution over a 4D tensor.

    Args:
        matrix: Input tensor in (batch, channels, height, width).
        kernels: List of KAN convolution kernels.
        kernel_size: Kernel size as int or (h, w).
        out_channels: Number of output channels.
        stride: Convolution stride tuple.
        dilation: Convolution dilation tuple.
        padding: Convolution padding tuple.
        groups: Group count.
        device: Output device.

    Returns:
        Output tensor of shape (batch, out_channels, out_h, out_w).
    """
    kernel_h, kernel_w = (
        kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    )
    h_out, w_out, batch_size, n_channels = calc_out_dims(
        matrix,
        (kernel_h, kernel_w),
        stride,
        dilation,
        padding,
    )
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size, out_channels, h_out, w_out)).to(
        device
    )  # estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold(
        (kernel_h, kernel_w), dilation=dilation, padding=padding, stride=stride
    )
    conv_groups = (
        unfold(matrix[:, :, :, :])
        .view(
            batch_size,
            n_channels,
            kernel_h * kernel_w,
            h_out * w_out,
        )
        .transpose(2, 3)
    )

    if n_channels % groups != 0:
        raise ValueError("Input channels must be divisible by groups")
    if out_channels % groups != 0:
        raise ValueError("Output channels must be divisible by groups")

    in_channels_per_group = n_channels // groups
    out_channels_per_group = out_channels // groups
    expected_kernels = groups * out_channels_per_group * in_channels_per_group
    if n_convs != expected_kernels:
        raise ValueError(
            f"Invalid kernel count: got {n_convs}, expected {expected_kernels} for groups={groups}"
        )

    for c_out in range(out_channels):
        out_channel_accum = torch.zeros((batch_size, h_out, w_out), device=device)

        group_idx = c_out // out_channels_per_group
        out_idx_in_group = c_out % out_channels_per_group
        kernel_block_offset = group_idx * out_channels_per_group * in_channels_per_group
        kernel_row_offset = kernel_block_offset + out_idx_in_group * in_channels_per_group
        input_channel_offset = group_idx * in_channels_per_group

        # Aggregate outputs from each kernel assigned to this output channel
        for in_idx in range(in_channels_per_group):
            kernel = kernels[kernel_row_offset + in_idx]
            channel_idx = input_channel_offset + in_idx
            conv_result = kernel.conv.forward(conv_groups[:, channel_idx, :, :].flatten(0, 1))
            out_channel_accum += conv_result.view(batch_size, h_out, w_out)

        matrix_out[:, c_out, :, :] = out_channel_accum  # Store results in output tensor

    return matrix_out


def add_padding(matrix: np.ndarray, padding: tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix.

    Args:
        matrix: Matrix to pad.
        padding: Number of rows and columns to pad on both sides.

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding

    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix

    return padded_matrix
