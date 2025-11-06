"""
Utilities for evaluating sparse coders on language models with ce loss metrics.
"""

from typing import Optional

import torch
from nnsight import LanguageModel


cache_original_logits = None
cache_input_data = None


def get_nested_attr(obj, path):
    parts = path.split(".")
    current = obj
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def loss_recovered(
    text: str,  # a batch of text
    model: LanguageModel,  # an nnsight LanguageModel or compatible model
    submodule: str,  # submodule of model to intervene on
    sparse_coder,  # SparseCoder instance
    max_len: Optional[int] = None,  # max context length for loss recovered
    normalize_batch: bool = False,  # normalize batch before passing to sparse coder
    tracer_args: dict = None,  # minimize cache during model trace
    y_recon: torch.Tensor = None,  # reconstructed output from sparse coder
):
    """
    Compute how much of the model's loss is recovered by replacing the component output
    with the reconstruction by the sparse coder (autoencoder/transcoder).

    Returns:
        tuple: (loss_original, loss_reconstructed, loss_zero)
            - loss_original: Loss with unmodified model
            - loss_reconstructed: Loss with sparse coder reconstruction
            - loss_zero: Loss with component zeroed out
    """
    submodule_name = submodule
    submodule = get_nested_attr(
        model.model, submodule
    )  # model.model.layers[0].mlp.gate

    if tracer_args is None:
        tracer_args = {
            "use_cache": True,
            "output_attentions": False,
        }

    invoker_args = {"truncation": True, "max_length": 131072}

    # 1. Get unmodified logits (baseline)
    global cache_original_logits, cache_input_data
    if ".0." not in submodule_name:
        logits_original = cache_original_logits
        input_data = cache_input_data
        print("use cache.")
    else:
        with model.trace(text, invoker_args=invoker_args):
            input_data = model.inputs.save()
            logits_original = model.output.save()
        logits_original = logits_original.values()
        cache_original_logits = logits_original
        cache_input_data = input_data

    # 2. Get logits with sparse coder reconstruction
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        # Extract activations from the submodule
        x = submodule.input
        if normalize_batch:
            scale = (sparse_coder.d_in**0.5) / x.norm(dim=-1).mean()
            x = x * scale

        x = x.save()

    # Reconstruct using sparse coder
    x_hat = y_recon

    # Intervene with reconstruction
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        x = submodule.input
        if x.device != x_hat.device:
            x_hat = x_hat.to(x.device)
        if normalize_batch:
            scale = (sparse_coder.d_in**0.5) / x.norm(dim=-1).mean()
            submodule.output[:] = x_hat / scale
        else:
            submodule.output[:] = x_hat

        logits_reconstructed = model.output.save()
    logits_reconstructed = logits_reconstructed.values()

    # 3. Get logits when component is zeroed out (worst case)
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        x = submodule.output
        submodule.output[:] = torch.zeros_like(x)

        logits_zero = model.output.save()

    logits_zero = logits_zero.values()

    # Extract logits from model output
    logits_original = list(logits_original)[0]
    logits_reconstructed = list(logits_reconstructed)[0]
    logits_zero = list(logits_zero)[0]

    # Get tokens for loss calculation
    if isinstance(text, torch.Tensor):
        tokens = text
    else:
        try:
            tokens = input_data[1]["input_ids"]
        except Exception as _:
            tokens = input_data[1]["input"]

    # Compute cross-entropy losses
    losses = []
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        loss_kwargs = {"ignore_index": model.tokenizer.pad_token_id}
    else:
        loss_kwargs = {}

    for logits in [logits_original, logits_reconstructed, logits_zero]:
        loss = torch.nn.CrossEntropyLoss(**loss_kwargs)(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
        )
        losses.append(loss)

    return tuple(losses)


def compute_frac_recovered(
    loss_original: torch.Tensor,
    loss_reconstructed: torch.Tensor,
    loss_zero: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the fraction of loss recovered by the sparse coder.

    Args:
        loss_original: Original model loss
        loss_reconstructed: Loss with sparse coder reconstruction
        loss_zero: Loss with component zeroed out

    Returns:
        FLR: (loss_reconstructed - loss_zero) / (loss_original - loss_zero)
    """
    return (loss_reconstructed - loss_zero) / (loss_original - loss_zero)
