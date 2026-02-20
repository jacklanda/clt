import os
import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, List

import einops
import torch
import torch.distributed as dist
import einops
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.distributed import tensor as dtensor
from torch.distributed.tensor.device_mesh import DeviceMesh
from transformers import AutoModel
from nnsight import LanguageModel

from .config import SparseCoderConfig
from .fused_encoder import NO_COMPILE, EncoderOutput, fused_encoder
from .utils import decoder_impl, load_sharded, save_sharded



from torch.utils._python_dispatch import _disable_current_modes


def _all_reduce_detached(t: Tensor) -> Tensor:
    """All_reduce a detached DTensor with Partial placements.

    Bypasses DTensor dispatch by:
    1. Creating a fresh plain tensor via torch.empty (never a DTensor subclass)
    2. Disabling all Python dispatch modes during dist.all_reduce

    Only use on tensors that do NOT need gradients.
    """
    if not isinstance(t, dtensor.DTensor):
        return t
    has_partial = any(isinstance(p, dtensor.Partial) for p in t.placements)
    if not has_partial:
        return t
    # torch.empty always creates a plain torch.Tensor, never a subclass.
    src = t._local_tensor
    local = torch.empty(src.shape, dtype=src.dtype, device=src.device)
    local.copy_(src)
    for dim_idx, p in enumerate(t.placements):
        if isinstance(p, dtensor.Partial):
            pg = t.device_mesh.get_group(dim_idx)
            # Disable all Python dispatch modes so dist.all_reduce uses
            # the standard NCCL path without DTensor interception.
            with _disable_current_modes():
                dist.all_reduce(local, group=pg)
    return local


@dataclass
class ForwardOutput:
    y_hat: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    explained_variance: Tensor
    """Explained variance."""

    explained_variance_legacy: Tensor
    """Explained variance (legacy computation)."""

    unexplained_variance: Tensor
    """Fraction of variance unexplained (1 - explained variance)."""

    unexplained_variance_legacy: Tensor
    """Fraction of variance unexplained (1 - explained variance, legacy computation)."""

    is_last: bool = False
    """Whether this is the last target in a multi-target setup."""

    per_token_l0: float = 0.0
    """L0 sparsity (number of latents used) per token."""

    per_sequence_l0: float = 0.0
    """L0 sparsity (number of latents used) per sequence."""

    per_batch_l0: float = 0.0
    """L0 sparsity (number of latents used) per batch."""

    per_feature_l0: float = 0.0
    """L0 sparsity (number of latents used) per feature."""

    per_token_l1: float = 0.0
    """L1 sparsity (sum of absolute latent activations) per token."""

    per_sequence_l1: float = 0.0
    """L1 sparsity (sum of absolute latent activations) per sequence."""

    per_batch_l1: float = 0.0
    """L1 sparsity (sum of absolute latent activations) per batch."""

    per_feature_l1: float = 0.0
    """L1 sparsity (sum of absolute latent activations) per feature."""

    l2_loss: float = 0.0
    """L2 loss."""

    l2_ratio: float = 0.0
    """L2 ratio between reconstruction and target."""

    mse_loss: float = 0.0
    """MSE loss."""

    norm_mse_loss: float = 0.0
    """Normalized MSE loss."""

    frac_dead: float = 0.0
    """Fraction of dead latents."""

    frac_alive: float = 0.0
    """Fraction of alive latents."""

    cossim: float = 0.0
    """Cosine similarity between target and reconstruction."""

    relative_reconstruction_bias: float = 0.0
    """Relative reconstruction bias."""

    loss_original: float = 0.0
    """Original model cross-entropy loss (for language models)."""

    loss_reconstructed: float = 0.0
    """Cross-entropy loss with sparse coder reconstruction (for language models)."""

    loss_zero: float = 0.0
    """Cross-entropy loss with component zeroed out (for language models)."""

    frac_recovered: float = 0.0
    """Fraction of loss recovered: (loss_reconstructed - loss_zero) / (loss_original - loss_zero)."""


class MidDecoder:
    def __init__(
        self,
        sparse_coder: "SparseCoder",
        x: Tensor,
        activations: Tensor,
        indices: Tensor,
        dead_mask: Optional[Tensor] = None,
    ):
        self.sparse_coder = sparse_coder
        self.x = x
        self.latent_acts = activations
        self.latent_indices = indices
        self.dead_mask = dead_mask
        self.index = 0

    def copy(
        self,
        x: Tensor | None = None,
        activations: Tensor | None = None,
        indices: Tensor | None = None,
        dead_mask: Tensor | None = None,
        texts: List[str] | None = None,
        lm: AutoModel | None = None,
        submodule: str | None = None,
    ):
        if x is None:
            x = self.x
        if activations is None:
            activations = self.latent_acts
        if indices is None:
            indices = self.latent_indices
        if dead_mask is None:
            dead_mask = self.dead_mask
        return MidDecoder(
            self.sparse_coder,
            x,
            activations,
            indices,
            dead_mask,
        )

    def detach(self):
        if not hasattr(self, "original_activations"):
            self.original_activations = self.latent_acts
            self.latent_acts = self.latent_acts.detach()
            self.latent_acts.requires_grad = True

    def restore(self, is_last: bool = False):
        grad = self.latent_acts.grad
        assert grad is not None, "Activations have no gradient."
        self.latent_acts = self.original_activations
        self.latent_acts.backward(grad, retain_graph=not is_last)
        del self.original_activations
        if is_last:
            # Free tensors no longer needed after backward
            del self.latent_acts, self.latent_indices
            if hasattr(self, "x"):
                del self.x

    def next(self):
        self.index += 1

    def prev(self):
        self.index = max(0, self.index - 1)

    @property
    def will_be_last(self):
        return self.index + 1 >= self.sparse_coder.cfg.n_targets

    @property
    def current_w_dec(self):
        return (
            self.sparse_coder.W_decs[self.index]
            if self.sparse_coder.multi_target
            else self.sparse_coder.W_dec
        )

    @property
    def current_latent_acts(self):
        post_enc = (
            self.sparse_coder.post_encs[self.index]
            if self.sparse_coder.multi_target
            else self.sparse_coder.post_enc
        )

        if self.sparse_coder.cfg.post_encoder_scale:
            post_enc_scale = (
                self.sparse_coder.post_enc_scales[self.index]
                if self.sparse_coder.multi_target
                else self.sparse_coder.post_enc_scale
            )
            if isinstance(post_enc_scale, dtensor.DTensor):
                post_enc_scale = post_enc_scale.to_local()
        else:
            post_enc_scale = None

        latent_acts = self.latent_acts

        if isinstance(latent_acts, dtensor.DTensor):
            latent_acts = latent_acts.to_local()
            latent_indices = self.latent_indices.to_local()
            post_enc = post_enc.to_local()
            latent_acts = latent_acts + post_enc[latent_indices] * (latent_acts > 0)
            if post_enc_scale is not None:
                latent_acts = latent_acts * post_enc_scale[latent_indices]
            latent_acts = dtensor.DTensor.from_local(
                latent_acts,
                self.latent_acts.device_mesh,
                placements=self.latent_acts.placements,
            )
        else:
            latent_acts = latent_acts + post_enc[self.latent_indices] * (
                latent_acts > 0
            )
            if post_enc_scale is not None:
                latent_acts = latent_acts * post_enc_scale[self.latent_indices]

        return latent_acts

    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    )
    def __call__(
        self,
        y: Tensor | None,
        index: Optional[int] = None,
        addition: float | Tensor = 0,
        no_extras: bool = False,
        denormalize: bool = True,
        add_post_enc: bool = True,
        loss_mask: Tensor | None = None,
        # Optional parameters for cross-entropy loss calculation
        compute_nll_loss: bool = False,
        lm: LanguageModel = None,  # nnsight LanguageModel
        submodule: str = None,  # submodule to intervene on
        texts: List[str] = None,  # text batch for loss calculation
        compute_metrics: bool = True,
    ) -> ForwardOutput:
        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = self.x
            if isinstance(y, dtensor.DTensor):
                y = y.redistribute(
                    y.device_mesh, (dtensor.Replicate(), dtensor.Shard(1))
                )

        assert isinstance(y, Tensor), "y must be a tensor."
        if add_post_enc:
            latent_acts = self.current_latent_acts
        else:
            latent_acts = self.latent_acts
        if index is None:
            index = self.index
            self.next()
        elif self.sparse_coder.cfg.n_targets > 0:
            assert 0 <= index < self.sparse_coder.cfg.n_targets, "Index out of bounds."
        is_last = self.index >= self.sparse_coder.cfg.n_targets

        # Decode
        if latent_acts is None and self.latent_indices is None:
            y_hat = torch.zeros_like(self.x)
        else:
            latent_indices = self.latent_indices
            y_hat = self.sparse_coder.decode(latent_acts, latent_indices, index).clone()
        W_skip = (
            self.sparse_coder.W_skips[index]
            if hasattr(self.sparse_coder, "W_skips")
            else self.sparse_coder.W_skip
        )
        if W_skip is not None:
            dtype = self.sparse_coder.dtype
            x = self.x
            needs_cast = x.dtype != dtype
            _CHUNK = 256
            if isinstance(x, dtensor.DTensor):
                # DTensor slicing + .to(dtype) triggers expensive all_gather
                # that causes OOM. Work with local tensors instead.
                x_local = x.to_local()
                y_hat_placements = y_hat.placements if isinstance(y_hat, dtensor.DTensor) else None
                y_hat_local = y_hat.to_local().clone() if isinstance(y_hat, dtensor.DTensor) else y_hat.clone()
                W_skip_local = W_skip.to_local() if isinstance(W_skip, dtensor.DTensor) else W_skip
                if needs_cast and x_local.shape[0] > _CHUNK:
                    for i in range(0, x_local.shape[0], _CHUNK):
                        x_chunk = x_local[i : i + _CHUNK].to(dtype)
                        y_hat_local[i : i + _CHUNK] += x_chunk @ W_skip_local.mT
                        del x_chunk
                else:
                    y_hat_local += x_local.to(dtype) @ W_skip_local.mT
                if y_hat_placements is not None:
                    y_hat = dtensor.DTensor.from_local(
                        y_hat_local, x.device_mesh, y_hat_placements
                    )
                else:
                    y_hat = y_hat_local
            elif needs_cast and x.shape[0] > _CHUNK:
                # Chunk to avoid materializing the full dtype-cast copy of x
                for i in range(0, x.shape[0], _CHUNK):
                    x_chunk = x[i : i + _CHUNK].to(dtype)
                    y_hat[i : i + _CHUNK] += x_chunk @ W_skip.mT
                    del x_chunk
            else:
                y_hat += x.to(dtype) @ W_skip.mT
        y_hat += addition

        if denormalize:
            y_hat = self.sparse_coder.denormalize_output(y_hat)

        if no_extras:
            return ForwardOutput(
                y_hat=y_hat,
                latent_acts=self.latent_acts,
                latent_indices=self.latent_indices,
                explained_variance=y_hat.new_tensor(0.0),
                explained_variance_legacy=y_hat.new_tensor(0.0),
                unexplained_variance=y_hat.new_tensor(0.0),
                unexplained_variance_legacy=y_hat.new_tensor(0.0),
                is_last=is_last,
            )
        else:
            # Compute the residual
            error = y - y_hat
            if loss_mask is not None:
                error = error * loss_mask[..., None]

            # ---- Compute the loss (unexplained_variance) — always via the
            #      DTensor-compatible fast path that preserves autograd. ----
            resid_sum_of_squares = error.pow(2).sum(dim=-1)
            mean_sum_of_squares = y.pow(2).sum(dim=-1).mean(dim=0)
            mean_act_per_dimension = y.pow(2).mean()
            residual_variance = resid_sum_of_squares.mean(dim=0)
            total_variance_new = mean_sum_of_squares - mean_act_per_dimension.pow(2)
            # Work around PyTorch 2.7 DTensor bug: dividing two Partial
            # DTensors triggers an all_reduce whose sharding strategy is
            # not registered.  total_variance_new only depends on y (no
            # grad needed), so we detach & all_reduce it to a plain scalar
            # via the NCCL backend.  residual_variance keeps its DTensor
            # placement and autograd graph; multiplying a Partial DTensor
            # by a plain scalar needs no collective and preserves grads.
            total_variance_resolved = _all_reduce_detached(total_variance_new.detach())
            unexplained_variance = residual_variance * (1.0 / total_variance_resolved)
            del resid_sum_of_squares, mean_sum_of_squares, mean_act_per_dimension
            del residual_variance, total_variance_new, total_variance_resolved

            if not compute_metrics:
                _zero = y_hat.new_tensor(0.0)
                return ForwardOutput(
                    y_hat=y_hat,
                    latent_acts=self.latent_acts,
                    latent_indices=self.latent_indices,
                    explained_variance=_zero,
                    explained_variance_legacy=_zero,
                    unexplained_variance=unexplained_variance,
                    unexplained_variance_legacy=unexplained_variance,
                    is_last=is_last,
                )

            # ---- Full metrics (logging steps only) ----
            # Compute in no_grad using detached copies so metric intermediates
            # don't bloat the autograd graph or keep large DTensor replicas alive.
            with torch.no_grad():
                y_det = y.detach()
                y_hat_det = y_hat.detach()

                # Convert DTensors to plain (replicated) tensors for metrics.
                if isinstance(y_det, dtensor.DTensor):
                    y_det = y_det.full_tensor()
                if isinstance(y_hat_det, dtensor.DTensor):
                    y_hat_det = y_hat_det.full_tensor()

                error_det = y_det - y_hat_det

                # (per-token) MSE loss
                squared_error = error_det.pow(2)
                squared_error_per_batch = einops.reduce(
                    squared_error, "bsz neuron -> bsz", "sum"
                )
                mse_loss = squared_error_per_batch.mean()

                # Norm MSE
                y_centered = y_det - y_det.mean(0, keepdim=True)
                normalization = y_centered.norm(dim=-1, keepdim=True)
                norm_mse_loss = (error_det / (normalization + 1e-6)).pow(2).sum(dim=-1).mean()
                del y_centered, normalization

                # explained_variance (legacy & new)
                resid_ss = squared_error.sum(dim=-1)
                batched_variance_sum = (y_det - y_det.mean(dim=0)).pow(2).sum(dim=-1)
                explained_variance_legacy = 1 - (
                    resid_ss / batched_variance_sum
                ).mean(dim=0)
                del batched_variance_sum

                y_squared = y_det.pow(2)
                ms_of_sq = y_squared.sum(dim=-1).mean(dim=0)
                ma_per_dim = y_squared.mean()
                res_var = resid_ss.mean(dim=0)
                tot_var = ms_of_sq - ma_per_dim.pow(2)
                explained_variance = 1 - res_var / tot_var
                del y_squared, resid_ss

                unexplained_variance_legacy = 1.0 - explained_variance_legacy

                # L2 loss
                l2_loss = torch.linalg.norm(error_det, dim=-1).mean()

                # L2 norm ratio
                l2_norm_in = torch.norm(y_det, dim=-1)
                l2_norm_out = torch.norm(y_hat_det, dim=-1)
                l2_ratio = (l2_norm_out / l2_norm_in.clone()).mean()
                del l2_norm_in, l2_norm_out

                # Relative reconstruction bias
                y_hat_norm_squared = torch.norm(y_hat_det, dim=-1).pow(2)
                y_dot_y_hat = (y_det * y_hat_det).sum(dim=-1)
                relative_reconstruction_bias = (
                    y_hat_norm_squared.mean() / y_dot_y_hat.mean()
                )
                del y_hat_norm_squared, y_dot_y_hat

                # Cosine similarity
                y_normed = y_det / torch.linalg.norm(y_det, dim=-1, keepdim=True)
                y_hat_normed = y_hat_det / torch.linalg.norm(y_hat_det, dim=-1, keepdim=True)
                cossim = (y_normed * y_hat_normed).sum(dim=-1).mean()
                del y_normed, y_hat_normed, y_det, y_hat_det, error_det, squared_error

                # L0 & L1 sparsity
                context_stripe = 128
                per_token_l0 = (latent_acts != 0).float().sum(dim=-1).mean()

                latent_acts_reshaped = latent_acts.view(
                    latent_acts.shape[0] // context_stripe,
                    context_stripe,
                    latent_acts.shape[1],
                )
                per_sequence_l0 = (
                    (latent_acts_reshaped != 0).float().sum(dim=(1, 2)).mean()
                )
                per_batch_l0 = (latent_acts != 0).float().sum()
                per_feature_l0 = (latent_acts != 0).float().sum(dim=0)

                per_token_l1 = latent_acts.abs().sum().mean()
                per_sequence_l1 = latent_acts_reshaped.abs().sum(dim=(1, 2)).mean()
                per_batch_l1 = latent_acts.abs().sum()
                per_feature_l1 = latent_acts.abs().sum(dim=0)

                assert len(latent_acts.shape) == 2, "latent_acts must be 2D"
                frac_alive = (
                    latent_acts.sum(dim=0) != 0
                ).float().sum() / latent_acts.shape[1]
                frac_dead = 1.0 - frac_alive

            # Cross-entropy losses of language models (optional)
            loss_original = torch.tensor(0.0)
            loss_reconstructed = torch.tensor(0.0)
            loss_zero = torch.tensor(0.0)
            frac_recovered = torch.tensor(0.0)

            if compute_nll_loss:
                if lm is None or submodule is None or texts is None:
                    raise ValueError(
                        "compute_nll_loss=True requires lm, submodule, and texts to be provided"
                    )

                try:
                    from .evaluation import (
                        loss_recovered as compute_loss_recovered,
                        compute_frac_recovered,
                    )

                    batched_loss_original = []
                    batched_loss_reconstructed = []
                    batched_loss_zero = []
                    batched_frac_recovered = []

                    # Compute cross-entropy losses
                    for text in texts:
                        loss_original, loss_reconstructed, loss_zero = (
                            compute_loss_recovered(
                                text=text,
                                model=lm,
                                submodule=submodule,
                                sparse_coder=self.sparse_coder,
                                normalize_batch=True,
                                y_recon=y_hat,
                            )
                        )
                        # Compute fraction recovered
                        frac_recovered = compute_frac_recovered(
                            loss_original, loss_reconstructed, loss_zero
                        )
                        batched_loss_original.append(loss_original)
                        batched_loss_reconstructed.append(loss_reconstructed)
                        batched_loss_zero.append(loss_zero)
                        batched_frac_recovered.append(frac_recovered)
                    loss_original = torch.stack(batched_loss_original).mean()
                    loss_reconstructed = torch.stack(batched_loss_reconstructed).mean()
                    loss_zero = torch.stack(batched_loss_zero).mean()
                    frac_recovered = torch.stack(batched_frac_recovered).mean()
                except ImportError as e:
                    print(f"Warning: Could not import loss_recovered function: {e}")
                    print("Cross-entropy losses will not be computed.")

                print(
                    "loss_original:",
                    loss_original,
                    "loss_reconstructed:",
                    loss_reconstructed,
                    "loss_zero:",
                    loss_zero,
                    "frac_recovered:",
                    frac_recovered,
                )

            return ForwardOutput(
                y_hat=y_hat,
                latent_acts=self.latent_acts,
                latent_indices=self.latent_indices,
                explained_variance=explained_variance,
                explained_variance_legacy=explained_variance_legacy,
                unexplained_variance=unexplained_variance,
                unexplained_variance_legacy=unexplained_variance_legacy,
                is_last=is_last,
                per_token_l0=per_token_l0,
                per_sequence_l0=per_sequence_l0,
                per_batch_l0=per_batch_l0,
                per_feature_l0=per_feature_l0,
                per_token_l1=per_token_l1,
                per_sequence_l1=per_sequence_l1,
                per_batch_l1=per_batch_l1,
                per_feature_l1=per_feature_l1,
                l2_loss=l2_loss,
                l2_ratio=l2_ratio,
                mse_loss=mse_loss,
                norm_mse_loss=norm_mse_loss,
                frac_dead=frac_dead,
                frac_alive=frac_alive,
                cossim=cossim,
                relative_reconstruction_bias=relative_reconstruction_bias,
                loss_original=loss_original.item(),
                loss_reconstructed=loss_reconstructed.item(),
                loss_zero=loss_zero.item(),
                frac_recovered=frac_recovered.item(),
            )


class SparseCoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
        mesh: Optional[DeviceMesh] = None,
        d_out: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        # Support separate output dimension for transcoders with dimension change
        self.d_out = (
            d_out if d_out is not None else (cfg.d_out if cfg.d_out > 0 else d_in)
        )
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        self.multi_target = cfg.n_targets > 0 and cfg.transcode
        self.mesh = mesh
        weight_dtype = cfg.torch_dtype
        if weight_dtype is None:
            weight_dtype = dtype
        encoder_dtype = weight_dtype
        decoder_dtype = weight_dtype
        self.encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=encoder_dtype
        )
        self.encoder.bias.data.zero_()
        if mesh is None:
            self.encoder = nn.Linear(
                d_in, self.num_latents, device=device, dtype=encoder_dtype
            )
            self.encoder.bias.data.zero_()
        else:
            self.encoder = nn.Linear(
                d_in,
                self.num_latents // mesh.shape[1],
                device=device,
                dtype=encoder_dtype,
            )
            self.encoder.bias.data.zero_()
            scaling = 1 / self.encoder.weight.shape[1] ** 0.5
            self.encoder.register_parameter(
                "weight",
                nn.Parameter(
                    # default torch initialization
                    dtensor.rand(
                        (self.num_latents, d_in),
                        dtype=encoder_dtype,
                        device_mesh=mesh,
                        placements=[dtensor.Replicate(), dtensor.Shard(0)],
                    )
                    * (2.0 * scaling)
                    - scaling
                ),
            )
            self.encoder.register_parameter(
                "bias",
                nn.Parameter(
                    dtensor.DTensor.from_local(
                        self.encoder.bias.data,
                        mesh,
                        placements=[
                            dtensor.Replicate(),
                            dtensor.Shard(0),
                        ],
                    )
                ),
            )

        if decoder:
            # Transcoder initialization: use zeros
            if cfg.transcode:

                def create_W_dec():
                    num_latents = self.num_latents
                    if self.cfg.coalesce_topk == "per-layer":
                        num_latents *= max(1, cfg.n_sources)
                    if mesh is not None:
                        result = dtensor.zeros(
                            (num_latents, self.d_out),
                            dtype=decoder_dtype,
                            device_mesh=mesh,
                            placements=[
                                dtensor.Replicate(),
                                dtensor.Shard(1) if cfg.tp_output else dtensor.Shard(0),
                            ],
                        )
                    else:
                        result = torch.zeros(
                            num_latents, self.d_out, device=device, dtype=decoder_dtype
                        )
                    return nn.Parameter(result)

                if (
                    self.multi_target
                    and self.cfg.coalesce_topk
                    not in (
                        "concat",
                        "per-layer",
                    )
                    and not cfg.per_source_tied
                ):
                    self.W_decs = nn.ParameterList()
                    for i in range(cfg.n_targets):
                        self.W_decs.append(create_W_dec())
                    self.W_dec = self.W_decs[0]
                else:
                    self.W_dec = create_W_dec()

            # Sparse autoencoder initialization: use the transpose of encoder weights
            else:
                self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
                if self.cfg.normalize_decoder:
                    self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        def create_bias():
            if mesh is not None:
                result = dtensor.zeros(
                    (self.d_out,),
                    dtype=dtype,
                    device_mesh=mesh,
                    placements=[
                        dtensor.Replicate(),
                        dtensor.Shard(0) if cfg.tp_output else dtensor.Replicate(),
                    ],
                )
            else:
                result = torch.zeros(self.d_out, device=device, dtype=dtype)
            return nn.Parameter(result)

        def create_W_skip():
            if not cfg.skip_connection:
                return None
            if mesh is not None:
                result = dtensor.zeros(
                    (self.d_out, self.d_in),
                    dtype=dtype,
                    device_mesh=mesh,
                    placements=[dtensor.Replicate(), dtensor.Shard(0)],
                )
            else:
                result = torch.zeros(self.d_out, self.d_in, device=device, dtype=dtype)
            return nn.Parameter(result)

        if self.multi_target and self.cfg.coalesce_topk not in ("concat", "per-layer"):
            self.b_decs = nn.ParameterList()
            self.W_skips = nn.ParameterList()
            for _ in range(cfg.n_targets):
                self.b_decs.append(create_bias())
                self.W_skips.append(create_W_skip())
            self.W_skip = self.W_skips[0]
            self.b_dec = self.b_decs[0]
        else:
            self.b_dec = create_bias()
            self.W_skip = create_W_skip()

        if cfg.normalize_io:
            if mesh is not None:
                self.register_buffer(
                    "in_norm",
                    dtensor.ones(
                        1,
                        dtype=dtype,
                        device_mesh=mesh,
                        placements=[dtensor.Replicate(), dtensor.Replicate()],
                    ),
                )
                self.register_buffer(
                    "out_norm",
                    dtensor.ones(
                        1,
                        dtype=dtype,
                        device_mesh=mesh,
                        placements=[dtensor.Replicate(), dtensor.Replicate()],
                    ),
                )
            else:
                self.register_buffer(
                    "in_norm", torch.ones(1, device=device, dtype=dtype)
                )
                self.register_buffer(
                    "out_norm", torch.ones(1, device=device, dtype=dtype)
                )

        def make_post_enc(is_zeros: bool = True):
            if mesh is not None:
                post_enc = (dtensor.zeros if is_zeros else dtensor.ones)(
                    (self.num_latents,),
                    dtype=dtype,
                    device_mesh=mesh,
                    placements=[dtensor.Replicate(), dtensor.Replicate()],
                )
            else:
                post_enc = (torch.zeros if is_zeros else torch.ones)(
                    self.num_latents, device=device, dtype=dtype
                )
            post_enc = nn.Parameter(post_enc, requires_grad=cfg.train_post_encoder)
            return post_enc

        if self.multi_target:
            self.post_encs = nn.ParameterList()
            for _ in range(cfg.n_targets):
                self.post_encs.append(make_post_enc())
        else:
            self.post_enc = make_post_enc()

        if self.cfg.post_encoder_scale:
            if self.multi_target:
                self.post_enc_scales = nn.ParameterList()
                for i in range(cfg.n_targets):
                    self.post_enc_scales.append(make_post_enc(is_zeros=i > 0))
            else:
                self.post_enc_scale = make_post_enc(is_zeros=False)

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "SparseCoder"]:
        """Load sparse coders for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: SparseCoder.load_from_disk(
                    repo_path / layer, device=device, decoder=decoder
                )
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: SparseCoder.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return SparseCoder.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        mesh: Optional[DeviceMesh] = None,
    ) -> "SparseCoder":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            d_out = cfg_dict.pop(
                "d_out", None
            )  # Support legacy checkpoints without d_out
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = SparseCoder(
            d_in, cfg, device=device, decoder=decoder, mesh=mesh, d_out=d_out
        )
        sae.load_state(path)
        return sae

    def load_state(self, path: os.PathLike, strict: bool = False):
        current_state_dict = self.state_dict()

        filename = str(Path(path) / "sae.safetensors")
        if self.mesh is None:
            state_dict = load_file(
                filename,
                device=str(self.device),
            )
        else:
            state_dict = load_sharded(
                filename,
                current_state_dict,
                self.mesh,
            )
        if hasattr(self, "post_enc") and "post_enc" not in state_dict:
            print("Imputing post_enc")
            state_dict["post_enc"] = self.post_enc.clone()
        if hasattr(self, "post_encs") and not any(
            f"post_encs.{i}" in state_dict for i in range(len(self.post_encs))
        ):
            print("Imputing post_encs")
            for i, post_enc in enumerate(self.post_encs):
                state_dict[f"post_encs.{i}"] = post_enc.clone()
        if hasattr(self, "post_enc_scale") and "post_enc_scale" not in state_dict:
            print("Imputing post_enc_scale")
            state_dict["post_enc_scale"] = self.post_enc_scale.clone()
        if hasattr(self, "post_enc_scales") and not any(
            f"post_enc_scales.{i}" in state_dict
            for i in range(len(self.post_enc_scales))
        ):
            print("Imputing post_enc_scales")
            for i, post_enc_scale in enumerate(self.post_enc_scales):
                if f"post_enc_scales.{i}" not in state_dict:
                    state_dict[f"post_enc_scales.{i}"] = post_enc_scale.clone()
        if hasattr(self, "W_decs") and not any(
            f"W_decs.{i}" in state_dict for i in range(len(self.W_decs))
        ):
            print("Imputing W_decs")
            state_dict["W_decs.0"] = state_dict.pop("W_dec")
            for i, W_dec in enumerate(self.W_decs):
                if i > 0:
                    state_dict[f"W_decs.{i}"] = W_dec.clone()
        if (
            self.cfg.skip_connection
            and hasattr(self, "W_skips")
            and not any(f"W_skips.{i}" in state_dict for i in range(len(self.W_skips)))
        ):
            state_dict["W_skips.0"] = state_dict.pop("W_skip")
            for i, W_skip in enumerate(self.W_skips):
                if i > 0:
                    state_dict[f"W_skips.{i}"] = W_skip.clone()
        if hasattr(self, "b_decs") and not any(
            f"b_decs.{i}" in state_dict for i in range(len(self.b_decs))
        ):
            print("Imputing b_decs")
            state_dict["b_decs.0"] = state_dict.pop("b_dec")
            for i, b_dec in enumerate(self.b_decs):
                if i > 0:
                    state_dict[f"b_decs.{i}"] = b_dec.clone()

        items = [(k, v) for k, v in state_dict.items() if k in current_state_dict]
        current_keys = list(current_state_dict.keys())
        items.sort(key=lambda x: current_keys.index(x[0]))
        state_dict = dict()
        for k, v in items:
            state_dict[k] = v
        self.load_state_dict(state_dict, strict=strict)

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        if (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0:
            path.mkdir(parents=True, exist_ok=True)

        filename = str(path / "sae.safetensors")
        current_state_dict = self.state_dict()
        is_main = save_sharded(current_state_dict, filename, mesh=self.mesh)

        if is_main:
            with open(path / "cfg.json", "w") as f:
                json.dump(
                    {
                        **self.cfg.to_dict(),
                        "d_in": self.d_in,
                        "d_out": self.d_out,
                    },
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def normalize_input(self, x: Tensor) -> Tensor:
        if self.cfg.normalize_io:
            return x * (x.shape[-1] ** 0.5 / self.in_norm)
        return x

    def denormalize_output(self, x: Tensor) -> Tensor:
        if self.cfg.normalize_io:
            return x * (self.out_norm / (x.shape[-1] ** 0.5))
        return x

    @torch.compile(disable=NO_COMPILE)
    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        x = self.normalize_input(x)

        if not self.cfg.transcode:
            x = x - self.b_dec

        return fused_encoder(
            x,
            self.encoder.weight,
            self.encoder.bias,
            self.cfg.k,
            self.cfg.activation,
            self.cfg.use_fp8,
            self.cfg.encoder_tile_size,
        )

    def decode(
        self,
        top_acts: Tensor | dtensor.DTensor,
        top_indices: Tensor | dtensor.DTensor,
        index: int = 0,
    ) -> Tensor:
        W_dec = self.W_decs[index] if hasattr(self, "W_decs") else self.W_dec
        b_dec = self.b_decs[index] if hasattr(self, "b_decs") else self.b_dec

        assert W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), W_dec)
        return y + b_dec

    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        *,
        dead_mask: Tensor | None = None,
        return_mid_decoder: bool = False,
        loss_mask: Tensor | None = None,
    ) -> ForwardOutput | MidDecoder:
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16 if self.cfg.dtype != "float16" else torch.float16,
            enabled=torch.cuda.is_bf16_supported(),
        ):
            top_acts, top_indices = self.encode(x)

            x = self.normalize_input(x)

            mid_decoder = MidDecoder(self, x, top_acts, top_indices, dead_mask)
            if self.multi_target or return_mid_decoder:
                return mid_decoder
            else:
                return mid_decoder(y, 0, loss_mask=loss_mask)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        for W_dec in self.W_decs if self.multi_target else (self.W_dec,):
            assert W_dec is not None, "Decoder weight was not initialized."

            eps = torch.finfo(W_dec.dtype).eps
            norm = torch.norm(W_dec.data, dim=1, keepdim=True)
            W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        for W_dec in self.W_decs if self.multi_target else (self.W_dec,):
            assert W_dec is not None, "Decoder weight was not initialized."
            assert W_dec.grad is not None  # keep pyright happy

            parallel_component = einops.einsum(
                W_dec.grad,
                W_dec.data,
                "d_sae d_in, d_sae d_in -> d_sae",
            )
            W_dec.grad -= einops.einsum(
                parallel_component,
                W_dec.data,
                "d_sae, d_sae d_in -> d_sae d_in",
            )


# Allow for alternate naming conventions
Sae = SparseCoder
