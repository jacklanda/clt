import os
from typing import Literal, NamedTuple

import torch
import torch.distributed.tensor as dtensor
import torch.nn.functional as F

try:
    import rtopk
except ImportError:
    rtopk = None

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from .kernels import (
    COODecoder,
    triton_coo_sparse_dense_matmul,
    triton_sparse_transpose_dense_matmul,
)
from .nanogpt import linear
from .utils import decoder_impl

NO_COMPILE = os.environ.get("SPARSIFY_NO_COMPILE", "0") == "1"
NO_RTOPK = os.environ.get("SPARSIFY_NO_RTOPK", "1") == "1"
# Environment variable override for tile size (0=auto, -1=disabled, >0=explicit)
ENV_TILE_SIZE = int(os.environ.get("SPARSIFY_TILE_SIZE", "0"))

MAX_SIZE = 1024


# ---------------------------------------------------------------------------
# Triton kernel: fused tiled matmul + ReLU + top-k
# Processes the encoder weight matrix in tiles so the full [N, num_latents]
# preactivation tensor is never materialized in HBM.
# ---------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _tiled_encode_topk_kernel(
        # Pointers
        x_ptr,
        w_ptr,
        b_ptr,
        out_vals_ptr,
        out_idxs_ptr,
        # Dimensions
        D_in,
        D_latent,
        # Strides
        stride_xn,
        stride_wl,
        # Compile-time constants
        K: tl.constexpr,
        TILE_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused tiled encoder: matmul + ReLU + streaming top-k.

        Each program instance handles one input row. It iterates over tiles
        of the latent dimension, computing dot-products in register, applying
        ReLU, and maintaining a running top-K buffer without ever writing the
        full preactivation vector to global memory.
        """
        row_id = tl.program_id(0)

        # ---- initialise running top-k in registers ----
        k_offs = tl.arange(0, K)
        topk_vals = tl.full([K], value=-1e30, dtype=tl.float32)
        topk_idxs = tl.zeros([K], dtype=tl.int32)

        # ---- iterate over latent tiles ----
        for tile_start in range(0, D_latent, TILE_L):
            l_offs = tl.arange(0, TILE_L)
            l_mask = (tile_start + l_offs) < D_latent

            # Compute dot products for this tile: tile_preacts[j] = x @ w[tile_start+j]
            tile_preacts = tl.zeros([TILE_L], dtype=tl.float32)
            for d_start in range(0, D_in, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D_in

                x_block = tl.load(
                    x_ptr + row_id * stride_xn + d_offs,
                    mask=d_mask,
                    other=0.0,
                ).to(tl.float32)

                # [TILE_L, BLOCK_D]
                w_block = tl.load(
                    w_ptr + (tile_start + l_offs[:, None]) * stride_wl + d_offs[None, :],
                    mask=l_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                tile_preacts += tl.sum(w_block * x_block[None, :], axis=1)

            # Add bias + ReLU
            tile_bias = tl.load(b_ptr + tile_start + l_offs, mask=l_mask, other=0.0).to(
                tl.float32
            )
            tile_preacts = tl.maximum(tile_preacts + tile_bias, 0.0)
            tile_preacts = tl.where(l_mask, tile_preacts, -1e30)

            # ---- merge tile into running top-k ----
            # Concatenate [topk_vals, tile_preacts] and [topk_idxs, tile_idxs]
            # then pick the top-K.  We use an iterative replacement strategy:
            # for each tile element, if it beats the current min of topk, swap.
            cur_min = tl.min(topk_vals)
            for j in tl.static_range(TILE_L):
                # Extract scalar from tile_preacts[j]
                val = tl.sum(
                    tl.where(
                        l_offs == j,
                        tile_preacts,
                        tl.zeros([TILE_L], dtype=tl.float32),
                    )
                )
                if val > cur_min:
                    # Find the first position holding the minimum
                    is_min_mask = topk_vals == cur_min
                    # Build a replacement mask: only the first True position
                    # Use cumsum trick: cumsum of is_min_mask, replace where cumsum==1
                    cum = tl.cumsum(is_min_mask.to(tl.int32), axis=0)
                    replace_mask = is_min_mask & (cum == 1)
                    topk_vals = tl.where(replace_mask, val, topk_vals)
                    topk_idxs = tl.where(
                        replace_mask, (tile_start + j).to(tl.int32), topk_idxs
                    )
                    cur_min = tl.min(topk_vals)

        # ---- store results ----
        tl.store(out_vals_ptr + row_id * K + k_offs, topk_vals, mask=k_offs < K)
        tl.store(out_idxs_ptr + row_id * K + k_offs, topk_idxs, mask=k_offs < K)


def _tiled_topk_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    tile_l: int = 64,
    block_d: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused tiled-encode top-k Triton kernel."""
    N, D_in = x.shape
    D_latent = weight.shape[0]

    out_vals = torch.empty(N, k, device=x.device, dtype=torch.float32)
    out_idxs = torch.empty(N, k, device=x.device, dtype=torch.int32)

    # Constexpr parameters must be powers of two for Triton
    tile_l_po2 = triton.next_power_of_2(min(tile_l, D_latent))
    block_d_po2 = triton.next_power_of_2(min(block_d, D_in))
    k_po2 = triton.next_power_of_2(k)

    grid = (N,)
    _tiled_encode_topk_kernel[grid](
        x,
        weight,
        bias,
        out_vals,
        out_idxs,
        D_in,
        D_latent,
        x.stride(0),
        weight.stride(0),
        K=k_po2,
        TILE_L=tile_l_po2,
        BLOCK_D=block_d_po2,
    )

    # Trim padding if k was rounded up
    if k_po2 != k:
        # Take actual top-k from the padded result
        topk = out_vals[:, :k_po2].topk(k, dim=-1, sorted=False)
        out_vals = topk.values
        out_idxs = out_idxs.gather(-1, topk.indices)
    return out_vals.to(x.dtype), out_idxs.long()


def _tiled_topk_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-efficient top-k via PyTorch tiling (fallback when Triton unavailable
    or for large tile sizes where the Triton kernel's register pressure is too high).

    Processes the encoder weight matrix in tiles of ``tile_size`` latents.
    Peak activation memory drops from O(N * num_latents) to
    O(N * tile_size + N * 2k).
    """
    N = x.shape[0]
    num_latents = weight.shape[0]

    global_vals = x.new_full((N, k), float("-inf"))
    global_idxs = torch.zeros(N, k, device=x.device, dtype=torch.long)

    for start in range(0, num_latents, tile_size):
        end = min(start + tile_size, num_latents)

        # Materialise only one tile of preactivations at a time
        tile_preacts = F.linear(x, weight[start:end], bias[start:end])
        tile_preacts.relu_()

        local_k = min(k, end - start)
        tile_vals, tile_idxs = torch.topk(tile_preacts, local_k, dim=-1, sorted=False)
        tile_idxs = tile_idxs + start  # offset to global latent indices

        del tile_preacts  # free HBM immediately

        # Merge with running global top-k
        merged_vals = torch.cat([global_vals, tile_vals], dim=-1)
        merged_idxs = torch.cat([global_idxs, tile_idxs], dim=-1)

        best = torch.topk(merged_vals, k, dim=-1, sorted=False)
        global_vals = best.values
        global_idxs = merged_idxs.gather(-1, best.indices)

    return global_vals, global_idxs


def _resolve_tile_size(num_latents: int, k: int, cfg_tile_size: int) -> int:
    """Decide the effective tile size.

    Returns 0 when tiling should be skipped (standard path).
    """
    # Explicit env-var override takes priority
    ts = ENV_TILE_SIZE if ENV_TILE_SIZE != 0 else cfg_tile_size
    if ts == -1:
        return 0  # disabled
    if ts > 0:
        return ts
    # Auto-detect: tile when num_latents is large enough to matter
    if num_latents <= 8192:
        return 0
    return max(4096, 4 * k)


@torch.compile
def rtopk_topk(data, k: int, max_iter=10, k_div: int = 1):
    if rtopk is None or NO_RTOPK:
        return torch.topk(data, k, dim=1, sorted=False)
    else:
        if data.shape[-1] < MAX_SIZE:
            return rtopk.ops.rtopk(data, k, max_iter=max_iter)
        if data.shape[-1] % MAX_SIZE != 0:
            data = torch.nn.functional.pad(data, (0, data.shape[-1] % MAX_SIZE))
        data = data.unflatten(-1, (-1, MAX_SIZE))
        if data.shape[-1] <= k:
            indices = torch.arange(
                data.shape[-1], device=data.device, dtype=torch.int32
            )
            indices = indices.unsqueeze(0).expand(data.shape[0], -1)
            values = data
        else:
            values, indices = rtopk.ops.rtopk(data, k=k // k_div, max_iter=max_iter)

        values_l2, indices_l2 = rtopk_topk(
            values.flatten(-2), k=k, max_iter=max_iter, k_div=k_div
        )
        indices = (
            (
                indices
                + torch.arange(
                    data.shape[-2], device=indices.device, dtype=torch.int32
                )[:, None]
                * data.shape[-1]
            )
            .flatten(-2)
            .gather(-1, indices_l2.long())
        )
        return values_l2, indices


class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    """Activations of the top-k latents."""

    top_indices: torch.Tensor
    """Indices of the top-k features."""


class FusedEncoder(torch.autograd.Function):
    @torch.compile(disable=NO_COMPILE)
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        values,
        indices,
        activation: Literal["groupmax"] | str | None = None,
    ):
        # Save tensors needed for the backward pass
        ctx.save_for_backward(input, weight, bias, values, indices)
        ctx.k = values.shape[-1]
        ctx.activation = activation
        return values

    # @torch.compile
    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_values):
        input, weight, bias, values, indices = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        activation = ctx.activation

        grad_values = grad_values * (values > 0).to(grad_values)

        # --- Grad w.r.t. input ---
        if ctx.needs_input_grad[0]:
            grad_input = decoder_impl(
                indices,
                grad_values,
                weight,
            )

        if isinstance(grad_values, dtensor.DTensor):
            mesh = grad_values.device_mesh
            local_size = weight.to_local().shape[0]
            start_feature = mesh.get_local_rank(1) * local_size
            end_feature = start_feature + local_size

        # --- Grad w.r.t. bias ---
        if bias is not None and ctx.needs_input_grad[2]:
            if isinstance(bias, dtensor.DTensor):
                mesh = bias.device_mesh
                grad_bias = torch.zeros_like(bias.to_local())
                all_indices = indices.flatten()
                all_indices = all_indices.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                ).to_local()
                all_values = grad_values.flatten()
                all_values = all_values.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                ).to_local()

                # TODO bespoke all-to-all gradient communication
                # likely won't be necessary, the encoder backward pass is fast
                mask = (all_indices >= start_feature) & (all_indices < end_feature)
                all_indices = all_indices[mask] - start_feature
                all_values = all_values[mask]

                grad_bias.index_add_(
                    0, all_indices, all_values.type_as(bias.to_local())
                )
                grad_bias = dtensor.DTensor.from_local(
                    grad_bias, mesh, (dtensor.Replicate(), dtensor.Shard(0))
                )
            else:
                grad_bias = torch.zeros_like(bias)
                grad_bias.index_add_(
                    0, indices.flatten(), grad_values.flatten().type_as(bias)
                )

        # --- Grad w.r.t. weight ---
        if ctx.needs_input_grad[1]:
            # Accumulate contributions into the correct rows of grad_weight.
            _, D = input.shape
            if not isinstance(grad_values, dtensor.DTensor):
                grad_weight = triton_sparse_transpose_dense_matmul(
                    indices,
                    grad_values.float(),
                    input,
                    N=weight.shape[0],
                )
            else:
                mesh = grad_values.device_mesh
                local_weight = weight.to_local()
                gathered_input = input.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                ).to_local()
                if activation == "groupmax":
                    indices = indices.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Shard(1))
                    ).to_local()
                    values = grad_values.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Shard(1))
                    ).to_local()
                    local_k = ctx.k // mesh.shape[1]
                    start_f = mesh.get_local_rank(1) * local_k
                    indices = indices - start_f
                else:
                    gathered_indices = indices.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Replicate())
                    ).to_local()
                    gathered_values = grad_values.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Replicate())
                    ).to_local()

                    indices = gathered_indices.view(-1, ctx.k)
                    values = gathered_values.view(-1, ctx.k)

                    mask = (indices >= start_feature) & (indices < end_feature)
                    values *= mask.type_as(values)
                    indices = (indices - start_feature).clamp(
                        0, local_weight.shape[0] - 1
                    )
                local_grad_weight = triton_sparse_transpose_dense_matmul(
                    indices,
                    values.float(),
                    gathered_input,
                    N=local_weight.shape[0],
                )
                grad_weight = dtensor.DTensor.from_local(
                    local_grad_weight,
                    mesh,
                    (dtensor.Replicate(), dtensor.Shard(0)),
                )

        # The k parameter is an int, so return None for its gradient.
        return grad_input, grad_weight, grad_bias, None, None, None


def batch_topk(preacts, k, return_indices=False):
    expected_k = k * preacts.shape[0]
    if isinstance(preacts, dtensor.DTensor):
        mesh = preacts.device_mesh
        local_preacts = preacts.to_local()
        expected_local_k = k * local_preacts.shape[0]
        local_values_less, original_indices = rtopk_topk(local_preacts, k=k * 4)
        original_indices = original_indices.long()
        local_values = torch.topk(
            local_values_less.flatten(), expected_local_k, sorted=False
        ).values
        all_values = dtensor.DTensor.from_local(
            local_values,
            mesh,
            (dtensor.Shard(0), dtensor.Shard(0)),
        )
        all_values = all_values.redistribute(
            mesh, (dtensor.Shard(0), dtensor.Replicate())
        )
        combined_values = all_values.to_local()
        values, indices = torch.topk(combined_values, expected_k, sorted=False)
        values, indices = values[0], indices[0]
        if return_indices:
            return values, indices
        else:
            threshold = values.min()
            local_preacts[local_preacts < threshold] = 0
            return preacts, original_indices
    else:
        values_less, original_indices = rtopk_topk(preacts, k=k * 4)
        values, indices = torch.topk(values_less.flatten(), expected_k, sorted=False)
        if return_indices:
            return values, indices
        else:
            threshold = values.min()
            preacts[preacts < threshold] = 0
            return preacts, original_indices


class FusedEncoderCOO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, k: int, use_fp8: bool = False):
        """Forward pass for BatchTopK with COO sparse kernels."""
        preacts = F.relu(linear(input, weight, bias, use_fp8))
        preact_values, preact_indices = preacts.topk(k * 4, dim=-1, sorted=False)
        values, indices = batch_topk(preact_values, k, return_indices=True)
        indices = preact_indices.flatten()[indices]
        row_indices, col_indices = (
            indices // preacts.shape[1],
            indices % preacts.shape[1],
        )
        ctx.save_for_backward(input, weight, bias, row_indices, col_indices, values)
        return values, row_indices, col_indices

    @staticmethod
    def backward(ctx, grad_values, _grad_row_indices, _grad_col_indices):
        input, weight, bias, row_indices, col_indices, values = ctx.saved_tensors
        grad_values = grad_values * (values > 0).to(grad_values)
        grad_input = grad_weight = grad_bias = None

        # --- Grad w.r.t. input ---
        if ctx.needs_input_grad[0]:
            grad_input = COODecoder.apply(
                row_indices,
                col_indices,
                grad_values,
                weight,
            )

        # --- Grad w.r.t. bias ---
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.zeros_like(bias)
            grad_bias.index_add_(
                0, row_indices.flatten(), grad_values.flatten().type_as(bias)
            )
            grad_bias = grad_bias

        # --- Grad w.r.t. weight ---
        if ctx.needs_input_grad[1]:
            grad_weight = triton_coo_sparse_dense_matmul(
                torch.stack([row_indices, col_indices]),
                grad_values.float(),
                input,
                N=weight.shape[0],
            )

        return grad_input, grad_weight, grad_bias, None, None, None


def fused_encoder(
    input,
    weight,
    bias,
    k: int,
    activation: Literal["groupmax", "topk"],
    use_fp8: bool = False,
    tile_size: int = 0,
) -> EncoderOutput:
    """
    Convenience wrapper that performs an nn.Linear followed by `activation` with
    a backward pass optimized using index_add.

    input:  (N, D)
    weight: (M, D)
    bias:   (M,)
    k:      int (number of top elements to select along dim=1)
    tile_size: int (0=auto, -1=disabled, >0=explicit tile size for memory-efficient encoding)
    """
    # ---- Memory-efficient tiled path ----
    # Only for topk activation, non-FP8
    effective_tile = _resolve_tile_size(weight.shape[0], k, tile_size)
    if (
        effective_tile > 0
        and activation == "topk"
        and not use_fp8
    ):
        # --- DTensor tiled path ---
        if isinstance(input, dtensor.DTensor):
            mesh = input.device_mesh
            local_input = input.to_local()
            local_weight = weight.to_local()
            local_bias = bias.to_local()
            local_num_latents = local_weight.shape[0]
            local_tile = _resolve_tile_size(local_num_latents, k, tile_size)

            if local_tile > 0:
                with torch.no_grad():
                    local_values, local_indices = _tiled_topk_pytorch(
                        local_input, local_weight, local_bias, k,
                        tile_size=local_tile,
                    )
                    # Offset to global latent indices
                    local_indices += mesh.get_local_rank(1) * local_num_latents

                    # Gather across TP ranks and do final top-k
                    values_dt = dtensor.DTensor.from_local(
                        local_values, mesh,
                        (dtensor.Shard(0), dtensor.Shard(1)),
                    ).redistribute(mesh, (dtensor.Shard(0), dtensor.Replicate()))
                    indices_dt = dtensor.DTensor.from_local(
                        local_indices, mesh,
                        (dtensor.Shard(0), dtensor.Shard(1)),
                    ).redistribute(mesh, (dtensor.Shard(0), dtensor.Replicate()))

                    lv, li = values_dt.to_local(), indices_dt.to_local()
                    lv, li_ = rtopk_topk(lv, k=k)
                    li = torch.gather(li, 1, li_.long())

                    values = dtensor.DTensor.from_local(
                        lv, mesh, (dtensor.Shard(0), dtensor.Replicate()),
                    )
                    indices = dtensor.DTensor.from_local(
                        li, mesh, (dtensor.Shard(0), dtensor.Replicate()),
                    )

                values = FusedEncoder.apply(
                    input, weight, bias, values, indices, activation
                )
                return EncoderOutput(top_acts=values, top_indices=indices)
            # else: fall through to standard path

        # --- Non-DTensor tiled path ---
        elif not isinstance(input, dtensor.DTensor):
            with torch.no_grad():
                use_triton_tiled = (
                    HAS_TRITON
                    and effective_tile <= 128
                    and k <= 128
                    and input.is_cuda
                )
                if use_triton_tiled:
                    values, indices = _tiled_topk_triton(
                        input, weight, bias, k, tile_l=effective_tile
                    )
                else:
                    values, indices = _tiled_topk_pytorch(
                        input, weight, bias, k, tile_size=effective_tile
                    )

            values = FusedEncoder.apply(
                input, weight, bias, values, indices, activation
            )
            return EncoderOutput(top_acts=values, top_indices=indices)

    # ---- Standard (non-tiled) path ----
    with torch.no_grad():
        preacts = linear(input, weight, bias, use_fp8)
        preacts.relu_()

        original_indices = None
        if activation == "batchtopk":
            preacts, original_indices = batch_topk(preacts, k)
            k *= 4
            activation = "topk"

        # Get top-k values and indices for each row
        if activation == "topk":
            if (
                isinstance(preacts, dtensor.DTensor)
                and preacts.device_mesh.shape[1] == 1
            ):
                mesh = preacts.device_mesh
                local_acts = preacts.to_local()
                local_values, local_indices = rtopk_topk(local_acts, k=k)
                values = dtensor.DTensor.from_local(
                    local_values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
                indices = dtensor.DTensor.from_local(
                    local_indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
            elif isinstance(preacts, dtensor.DTensor):
                mesh = preacts.device_mesh
                local_acts = preacts.to_local()
                if original_indices is not None:
                    local_indices = original_indices
                    local_values = torch.gather(local_acts, 1, original_indices)
                else:
                    local_values, local_indices = rtopk_topk(local_acts, k=k)
                local_indices += mesh.get_local_rank(1) * local_acts.shape[1]
                values = dtensor.DTensor.from_local(
                    local_values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Shard(1)),
                ).redistribute(mesh, (dtensor.Shard(0), dtensor.Replicate()))
                indices = dtensor.DTensor.from_local(
                    local_indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Shard(1)),
                ).redistribute(mesh, (dtensor.Shard(0), dtensor.Replicate()))
                local_values, local_indices = values.to_local(), indices.to_local()
                local_values, local_indices_ = rtopk_topk(local_values, k=k)
                local_indices = torch.gather(local_indices, 1, local_indices_.long())
                values = dtensor.DTensor.from_local(
                    local_values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
                indices = dtensor.DTensor.from_local(
                    local_indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
            else:
                values, indices = rtopk_topk(preacts, k=k)
        elif activation == "groupmax":
            if isinstance(preacts, dtensor.DTensor):
                mesh = preacts.device_mesh
                local_acts = preacts.to_local()
                assert k % mesh.shape[1] == 0
                local_k = k // mesh.shape[1]
                local_values, local_indices = local_acts.unflatten(
                    -1, (local_k, -1)
                ).max(dim=-1)
                offsets = torch.arange(
                    0,
                    local_acts.shape[1],
                    local_acts.shape[1] // local_k,
                    device=preacts.device,
                )
                mesh_offset = mesh.get_local_rank(1) * local_k
                indices = mesh_offset + offsets + local_indices
                values = local_values
                values = dtensor.DTensor.from_local(
                    values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Shard(1)),
                )
                indices = dtensor.DTensor.from_local(
                    indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Shard(1)),
                )
                # values = values.redistribute(
                #     mesh, (dtensor.Shard(0), dtensor.Replicate())
                # )
                # indices = indices.redistribute(
                #     mesh, (dtensor.Shard(0), dtensor.Replicate())
                # )
            else:
                num_latents = preacts.shape[1]
                values, indices = preacts.unflatten(-1, (k, -1)).max(dim=-1)
                offsets = torch.arange(
                    0, num_latents, num_latents // k, device=preacts.device
                )
                indices = offsets + indices
        else:
            raise ValueError(f"Unknown activation: {activation}")

    values = FusedEncoder.apply(input, weight, bias, values, indices, activation)

    return EncoderOutput(
        top_acts=values,
        top_indices=indices,
    )


class DeadLatentLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, loss_weight):
        ctx.save_for_backward(inputs, weight, bias, loss_weight)
        return inputs.new_tensor(0.0)

    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    @staticmethod
    def backward(ctx, grad):
        inputs, weight, bias, loss_weight_mul = ctx.saved_tensors
        was_dtensor = False
        lwmul_was_dtensor = isinstance(loss_weight_mul, dtensor.DTensor)
        if isinstance(weight, dtensor.DTensor):
            was_dtensor = True
            mesh = weight.device_mesh
            placements = weight.placements
            if not lwmul_was_dtensor:
                loss_weight_mul = dtensor.DTensor.from_local(
                    loss_weight_mul, mesh, (dtensor.Replicate(), dtensor.Replicate())
                )
            loss_weight_mul = loss_weight_mul.redistribute(mesh, placements)
        inputs, weight, bias, loss_weight_mul, grad = (
            inputs.to_local(),
            weight.to_local(),
            bias.to_local(),
            loss_weight_mul.to_local(),
            grad.to_local(),
        )
        loss_weight = grad * loss_weight_mul
        grad_weight = None
        grad_bias = None
        if ctx.needs_input_grad[1]:
            grad_weight = -torch.einsum("...x,yx,y->yx", inputs, weight, loss_weight)
            if was_dtensor:
                grad_weight = dtensor.DTensor.from_local(grad_weight, mesh, placements)
        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = -loss_weight.broadcast_to(bias.shape)
            if was_dtensor:
                grad_bias = dtensor.DTensor.from_local(grad_bias, mesh, placements)
        grad_weight_mul = None
        if ctx.needs_input_grad[3]:
            grad_weight_mul = -torch.einsum("...x,yx,->y", inputs, weight, grad)
            if was_dtensor:
                grad_weight_mul = dtensor.DTensor.from_local(
                    grad_weight_mul, mesh, placements
                )
                grad_weight_mul = grad_weight_mul.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                )
            if not lwmul_was_dtensor:
                grad_weight_mul = grad_weight_mul.to_local()
        return None, grad_weight, grad_bias, grad_weight_mul
