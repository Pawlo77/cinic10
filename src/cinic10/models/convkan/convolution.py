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

    return _multiple_convs_kan_conv2d_batched(
        conv_groups=conv_groups,
        kernels=kernels,
        out_channels=out_channels,
        groups=groups,
        in_channels_per_group=in_channels_per_group,
        out_channels_per_group=out_channels_per_group,
        batch_size=batch_size,
        h_out=h_out,
        w_out=w_out,
        device=device,
    )


def _batched_b_splines(
    x: torch.Tensor,
    grid: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Compute B-spline bases for a batched set of KANLinear kernels.

    Args:
        x: Tensor of shape (n_kernels, n_samples, in_features).
        grid: Tensor of shape (n_kernels, in_features, grid_points).
        spline_order: Spline order.

    Returns:
        Bases tensor with shape (n_kernels, n_samples, in_features, n_basis).
    """
    x_expanded = x.unsqueeze(-1)
    grid_left = grid[:, :, :-1].unsqueeze(1)
    grid_right = grid[:, :, 1:].unsqueeze(1)
    bases = ((x_expanded >= grid_left) & (x_expanded < grid_right)).to(x.dtype)

    for order_idx in range(1, spline_order + 1):
        left_num = x_expanded - grid[:, :, : -(order_idx + 1)].unsqueeze(1)
        left_den = (grid[:, :, order_idx:-1] - grid[:, :, : -(order_idx + 1)]).unsqueeze(1)
        right_num = grid[:, :, order_idx + 1 :].unsqueeze(1) - x_expanded
        right_den = (grid[:, :, order_idx + 1 :] - grid[:, :, 1:(-order_idx)]).unsqueeze(1)
        bases = (left_num / left_den) * bases[..., :-1] + (right_num / right_den) * bases[..., 1:]

    return bases.contiguous()


def _multiple_convs_kan_conv2d_batched(
    conv_groups: torch.Tensor,
    kernels,
    out_channels: int,
    groups: int,
    in_channels_per_group: int,
    out_channels_per_group: int,
    batch_size: int,
    h_out: int,
    w_out: int,
    device: str,
) -> torch.Tensor:
    """Vectorized ConvKAN forward over all kernel-channel pairs.

    This eliminates Python loops over channels and evaluates all grouped KAN
    kernels in batched tensor ops.
    """
    del device

    n_kernels = len(kernels)
    first_conv = kernels[0].conv

    # All conv-kernels in one KANConvolutionalLayer are built with matching layout.
    # If not, fall back to the slow reference implementation.
    same_layout = all(
        k.conv.in_features == first_conv.in_features
        and k.conv.grid_size == first_conv.grid_size
        and k.conv.spline_order == first_conv.spline_order
        and k.conv.enable_standalone_scale_spline == first_conv.enable_standalone_scale_spline
        and type(k.conv.base_activation) is type(first_conv.base_activation)
        for k in kernels
    )
    if not same_layout:
        return _multiple_convs_kan_conv2d_slow(
            conv_groups=conv_groups,
            kernels=kernels,
            out_channels=out_channels,
            groups=groups,
            in_channels_per_group=in_channels_per_group,
            out_channels_per_group=out_channels_per_group,
            batch_size=batch_size,
            h_out=h_out,
            w_out=w_out,
        )

    # Stack KANLinear parameters to evaluate all kernels in one pass.
    base_weight = torch.stack([k.conv.base_weight.squeeze(0) for k in kernels], dim=0)
    spline_weight = torch.stack([k.conv.spline_weight.squeeze(0) for k in kernels], dim=0)
    if first_conv.enable_standalone_scale_spline:
        spline_scaler = torch.stack([k.conv.spline_scaler.squeeze(0) for k in kernels], dim=0)
        scaled_spline_weight = spline_weight * spline_scaler.unsqueeze(-1)
    else:
        scaled_spline_weight = spline_weight
    grid = torch.stack([k.conv.grid for k in kernels], dim=0)

    # Build mapping from kernel index -> input channel index.
    block_size = out_channels_per_group * in_channels_per_group
    kernel_indices = torch.arange(n_kernels, device=conv_groups.device)
    group_indices = kernel_indices // block_size
    in_indices = kernel_indices % in_channels_per_group
    channel_indices = group_indices * in_channels_per_group + in_indices

    # Input patches as (channels, batch*spatial, kernel_pixels).
    # conv_groups shape: (batch, channels, spatial, kernel_pixels)
    channel_patches = conv_groups.permute(1, 0, 2, 3).reshape(
        conv_groups.size(1),
        batch_size * h_out * w_out,
        conv_groups.size(3),
    )
    kernel_inputs = channel_patches[channel_indices]

    # KANLinear forward in batch for all kernels.
    activated = first_conv.base_activation(kernel_inputs)
    base_output = torch.einsum("nmk,nk->nm", activated, base_weight)
    spline_bases = _batched_b_splines(kernel_inputs, grid, first_conv.spline_order)
    spline_output = torch.einsum("nmkp,nkp->nm", spline_bases, scaled_spline_weight)
    kernel_output = base_output + spline_output

    # Aggregate per output channel by summing contributions over input channels.
    kernel_output = kernel_output.view(
        groups,
        out_channels_per_group,
        in_channels_per_group,
        batch_size * h_out * w_out,
    )
    output_flat = kernel_output.sum(dim=2).reshape(out_channels, batch_size, h_out * w_out)
    return output_flat.permute(1, 0, 2).reshape(batch_size, out_channels, h_out, w_out)


def _multiple_convs_kan_conv2d_slow(
    conv_groups: torch.Tensor,
    kernels,
    out_channels: int,
    in_channels_per_group: int,
    out_channels_per_group: int,
    batch_size: int,
    h_out: int,
    w_out: int,
) -> torch.Tensor:
    """Reference implementation with explicit Python loops."""
    matrix_out = torch.zeros((batch_size, out_channels, h_out, w_out), device=conv_groups.device)

    for c_out in range(out_channels):
        out_channel_accum = torch.zeros((batch_size, h_out, w_out), device=conv_groups.device)

        group_idx = c_out // out_channels_per_group
        out_idx_in_group = c_out % out_channels_per_group
        kernel_block_offset = group_idx * out_channels_per_group * in_channels_per_group
        kernel_row_offset = kernel_block_offset + out_idx_in_group * in_channels_per_group
        input_channel_offset = group_idx * in_channels_per_group

        for in_idx in range(in_channels_per_group):
            kernel = kernels[kernel_row_offset + in_idx]
            channel_idx = input_channel_offset + in_idx
            conv_result = kernel.conv.forward(conv_groups[:, channel_idx, :, :].flatten(0, 1))
            out_channel_accum += conv_result.view(batch_size, h_out, w_out)

        matrix_out[:, c_out, :, :] = out_channel_accum

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
