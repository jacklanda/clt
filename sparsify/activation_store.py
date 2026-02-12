"""Activation caching for two-phase training.

Phase 1: Run base model forward pass, save (input, output, bos_mask) per hookpoint.
Phase 2: Train SAEs from cached activations without loading the base model.
"""

import json
import os
from fnmatch import fnmatchcase
from functools import partial

import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .utils import get_layer_list


def _resolve_hookpoints(model: PreTrainedModel, hookpoint_patterns: list[str],
                        layers: list[int], layer_stride: int) -> list[str]:
    """Resolve wildcard hookpoint patterns against the model."""
    if hookpoint_patterns:
        raw = []
        for name, _ in model.base_model.named_modules():
            if any(fnmatchcase(name, pat) for pat in hookpoint_patterns):
                raw.append(name)
        return natsorted(raw)[::layer_stride]
    else:
        if not layers:
            N = model.config.num_hidden_layers
            layers = list(range(0, N))
        layers_name, _ = get_layer_list(model)
        return [f"{layers_name}.{i}" for i in layers][::layer_stride]


def cache_activations(
    model: PreTrainedModel,
    dataset,
    hookpoint_patterns: list[str],
    layers: list[int],
    layer_stride: int,
    save_dir: str,
    batch_size: int = 32,
    ctx_len: int = 2048,
    transcode: bool = True,
    filter_bos: bool = False,
    remove_first_token: bool = False,
    max_examples: int | None = None,
):
    """Run base model forward pass and cache activations to disk as memmap files.

    Saves per hookpoint:
      - {hookpoint}/inputs.bin  (float16 memmap, shape [N_tokens, d_in])
      - {hookpoint}/outputs.bin (float16 memmap, shape [N_tokens, d_out])
      - {hookpoint}/bos_mask.bin (bool memmap, shape [N_tokens])
    Plus a metadata.json with shapes and hookpoint list.
    """
    model.eval()
    model.requires_grad_(False)
    device = model.device

    hookpoints = _resolve_hookpoints(model, hookpoint_patterns, layers, layer_stride)
    print(f"Caching activations for {len(hookpoints)} hookpoints")

    os.makedirs(save_dir, exist_ok=True)

    # First pass: count total tokens
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_tokens = len(dataset) * ctx_len

    # Resolve dimensions with a dummy forward pass
    name_to_module = {
        name: model.base_model.get_submodule(name) for name in hookpoints
    }
    dims = {}  # hookpoint -> (d_in, d_out)

    cached_io = {}

    def probe_hook(module, inputs, outputs, name=None):
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        out = outputs[0] if isinstance(outputs, tuple) else outputs
        cached_io[name] = (inp, out)

    # Run one batch to get dimensions
    sample_batch = next(iter(dl))
    x = sample_batch["input_ids"].to(device)
    handles = [
        mod.register_forward_hook(partial(probe_hook, name=name))
        for name, mod in name_to_module.items()
    ]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    for name in hookpoints:
        inp, out = cached_io[name]
        dims[name] = (inp.shape[-1], out.shape[-1])
    cached_io.clear()

    # Create memmap files
    memmaps = {}
    for name in hookpoints:
        d_in, d_out = dims[name]
        hp_dir = os.path.join(save_dir, name.replace(".", "_"))
        os.makedirs(hp_dir, exist_ok=True)

        memmaps[name] = {
            "inputs": np.memmap(
                os.path.join(hp_dir, "inputs.bin"),
                dtype=np.float16, mode="w+", shape=(total_tokens, d_in),
            ),
            "outputs": np.memmap(
                os.path.join(hp_dir, "outputs.bin"),
                dtype=np.float16, mode="w+", shape=(total_tokens, d_out),
            ),
            "bos_mask": np.memmap(
                os.path.join(hp_dir, "bos_mask.bin"),
                dtype=np.bool_, mode="w+", shape=(total_tokens,),
            ),
        }

    # Second pass: actually cache
    offset = 0

    def cache_hook(module, inputs, outputs, name=None):
        nonlocal cached_io
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        out = outputs[0] if isinstance(outputs, tuple) else outputs
        cached_io[name] = (inp.detach(), out.detach())

    pbar = tqdm(dl, desc="Caching activations")
    for batch in pbar:
        x = batch["input_ids"].to(device)
        B, S = x.shape

        # Compute bos_mask
        if model.config.bos_token_id is not None:
            bos_mask = x == model.config.bos_token_id
        else:
            bos_mask = torch.zeros_like(x, dtype=torch.bool)
        if not filter_bos:
            bos_mask[:] = False
        if remove_first_token:
            bos_mask[:, 0] = True

        handles = [
            mod.register_forward_hook(partial(cache_hook, name=name))
            for name, mod in name_to_module.items()
        ]
        with torch.no_grad():
            model(x)
        for h in handles:
            h.remove()

        n_tokens = B * S
        bos_flat = bos_mask.flatten().cpu().numpy()

        for name in hookpoints:
            inp, out = cached_io[name]
            # Flatten batch*seq -> tokens
            inp_flat = inp.flatten(0, 1).cpu().to(torch.float16).numpy()
            out_flat = out.flatten(0, 1).cpu().to(torch.float16).numpy()

            memmaps[name]["inputs"][offset:offset + n_tokens] = inp_flat
            memmaps[name]["outputs"][offset:offset + n_tokens] = out_flat
            memmaps[name]["bos_mask"][offset:offset + n_tokens] = bos_flat

        cached_io.clear()
        offset += n_tokens

    # Flush and save metadata
    for name in hookpoints:
        for mm in memmaps[name].values():
            mm.flush()

    metadata = {
        "hookpoints": hookpoints,
        "total_tokens": int(offset),
        "ctx_len": ctx_len,
        "dims": {name: list(dims[name]) for name in hookpoints},
        "dtype": "float16",
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Cached {offset:_} tokens across {len(hookpoints)} hookpoints to {save_dir}")


class CachedActivationDataset(TorchDataset):
    """Dataset that loads cached activations from memmap files.

    Each item returns a dict with:
      - inputs: dict[hookpoint_name -> Tensor[ctx_len, d_in]]
      - outputs: dict[hookpoint_name -> Tensor[ctx_len, d_out]]
      - bos_mask: Tensor[ctx_len] (from first hookpoint, same for all)
    """

    def __init__(self, cache_dir: str, ctx_len: int,
                 max_examples: int | None = None):
        with open(os.path.join(cache_dir, "metadata.json")) as f:
            self.metadata = json.load(f)

        self.cache_dir = cache_dir
        self.ctx_len = ctx_len
        self.hookpoints = self.metadata["hookpoints"]
        total_tokens = self.metadata["total_tokens"]
        self.num_sequences = total_tokens // ctx_len
        if max_examples is not None:
            self.num_sequences = min(self.num_sequences, max_examples)

        # Load memmap references (lazy, no memory cost)
        self.memmaps = {}
        for name in self.hookpoints:
            hp_dir = os.path.join(cache_dir, name.replace(".", "_"))
            d_in, d_out = self.metadata["dims"][name]
            self.memmaps[name] = {
                "inputs": np.memmap(
                    os.path.join(hp_dir, "inputs.bin"),
                    dtype=np.float16, mode="r",
                    shape=(total_tokens, d_in),
                ),
                "outputs": np.memmap(
                    os.path.join(hp_dir, "outputs.bin"),
                    dtype=np.float16, mode="r",
                    shape=(total_tokens, d_out),
                ),
                "bos_mask": np.memmap(
                    os.path.join(hp_dir, "bos_mask.bin"),
                    dtype=np.bool_, mode="r",
                    shape=(total_tokens,),
                ),
            }

    def __len__(self):
        return self.num_sequences

    def select(self, rng: range) -> "CachedActivationDataset":
        """Select a subset of the dataset."""
        ds = CachedActivationDataset.__new__(CachedActivationDataset)
        ds.metadata = self.metadata
        ds.cache_dir = self.cache_dir
        ds.ctx_len = self.ctx_len
        ds.hookpoints = self.hookpoints
        ds.memmaps = self.memmaps
        # Adjust the offset by slicing the memmaps
        ds.num_sequences = rng.stop - rng.start
        # Store offset for __getitem__
        ds._offset = getattr(self, "_offset", 0) + rng.start
        return ds

    def shard(self, num_shards: int, shard_id: int) -> "CachedActivationDataset":
        """Split the dataset into shards for distributed training."""
        ds = CachedActivationDataset.__new__(CachedActivationDataset)
        ds.metadata = self.metadata
        ds.cache_dir = self.cache_dir
        ds.ctx_len = self.ctx_len
        ds.hookpoints = self.hookpoints
        ds.memmaps = self.memmaps
        shard_size = self.num_sequences // num_shards
        ds._offset = getattr(self, "_offset", 0) + shard_id * shard_size
        ds.num_sequences = shard_size
        return ds

    def __getitem__(self, idx):
        actual_idx = getattr(self, "_offset", 0) + idx
        start = actual_idx * self.ctx_len
        end = start + self.ctx_len

        inputs = {}
        outputs = {}
        for name in self.hookpoints:
            inputs[name] = torch.from_numpy(
                self.memmaps[name]["inputs"][start:end].copy()
            ).float()
            outputs[name] = torch.from_numpy(
                self.memmaps[name]["outputs"][start:end].copy()
            ).float()

        bos_mask = torch.from_numpy(
            self.memmaps[self.hookpoints[0]]["bos_mask"][start:end].copy()
        )
        return {"inputs": inputs, "outputs": outputs, "bos_mask": bos_mask}
