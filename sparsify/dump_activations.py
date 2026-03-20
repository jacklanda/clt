"""Dump the first N steps' activations from the base model.

For each step, saves per-layer triplets:
  - mlp_in:        [batch_size, ctx_len, hidden_dim]   gate input (hidden states)
  - router_logits: [batch_size, ctx_len, num_experts]  gate output (pre/post softmax)
  - token_ids:     [batch_size, ctx_len]               input token IDs

Usage (single GPU):
    python -m clt.sparsify.dump_activations \
        --model /share/nlp/share/plm/OLMoE-1B-7B-0125 \
        --dataset /share/nlp/liuyang/workspace/circuit_tracing/RouterScope/data/openwebtext \
        --hookpoints "layers.*.mlp.gate" \
        --ctx_len 128 --batch_size 32 --num_steps 10 \
        --post_softmax --output_dir ./dumped_activations
"""

import argparse
import os
import sys
from fnmatch import fnmatchcase
from pathlib import Path

# Allow running as `python clt/sparsify/dump_activations.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from natsort import natsorted
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from clt.sparsify.data import MemmapDataset, chunk_and_tokenize


def parse_args():
    p = argparse.ArgumentParser(description="Dump base-model activations for CLT analysis")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--hookpoints", type=str, default="layers.*.mlp.gate",
                   help="Glob pattern for hookpoint module names")
    p.add_argument("--ctx_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--post_softmax", action="store_true", default=True,
                   help="Apply softmax to gate outputs before saving")
    p.add_argument("--no_post_softmax", action="store_false", dest="post_softmax")
    p.add_argument("--output_dir", type=str, default="./dumped_activations")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--shuffle_seed", type=int, default=42)
    p.add_argument("--filter_bos", action="store_true", default=True)
    p.add_argument("--remove_first_token", action="store_true", default=True)
    return p.parse_args()


def load_model_and_data(args):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(
        args.model,
        device_map={"": "cuda:0"},
        dtype=dtype,
        torch_dtype=dtype,
    )
    model.config.use_cache = False
    model.eval()
    model.requires_grad_(False)

    print(f"Loading dataset: {args.dataset}")
    if args.dataset.endswith(".bin"):
        dataset = MemmapDataset(args.dataset, args.ctx_len)
    else:
        try:
            dataset = load_dataset(args.dataset, split="train")
        except ValueError as e:
            if "load_from_disk" in str(e):
                dataset = Dataset.load_from_disk(args.dataset, keep_in_memory=False)
            else:
                raise
        if "input_ids" not in dataset.column_names:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            dataset = chunk_and_tokenize(
                dataset, tokenizer, max_seq_len=args.ctx_len,
            )
        dataset = dataset.shuffle(args.shuffle_seed)
        dataset = dataset.with_format("torch")

    return model, dataset


def resolve_hookpoints(model, pattern: str):
    """Resolve wildcard hookpoint pattern to actual module names."""
    resolved = []
    # model might be wrapped; get the base
    base = model
    if hasattr(model, "base_model"):
        base = model.base_model
    for name, _ in base.named_modules():
        if fnmatchcase(name, pattern):
            resolved.append(name)
    return natsorted(resolved)


def main():
    args = parse_args()
    model, dataset = load_model_and_data(args)

    base = model.base_model if hasattr(model, "base_model") else model
    hookpoints = resolve_hookpoints(model, args.hookpoints)
    print(f"Resolved {len(hookpoints)} hookpoints: {hookpoints[0]} ... {hookpoints[-1]}")

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Storage: step -> {hookpoint -> {"mlp_in": tensor, "router_logits": tensor}}
    for step, batch in enumerate(dl):
        if step >= args.num_steps:
            break

        token_ids = batch["input_ids"].to("cuda:0")  # [B, ctx_len]
        B, S = token_ids.shape

        # Dict to collect activations from hooks
        captured = {}

        def make_hook(hp_name):
            def _hook(module, inputs, outputs):
                inp = inputs[0] if isinstance(inputs, tuple) else inputs
                out = outputs[0] if isinstance(outputs, tuple) else outputs
                # gate input is already flattened to [B*S, hidden] by MoE block
                captured[hp_name] = {
                    "mlp_in": inp.detach().float().cpu(),
                    "router_logits": out.detach().float().cpu(),
                }
            return _hook

        # Register hooks
        handles = []
        for hp in hookpoints:
            mod = base.get_submodule(hp)
            handles.append(mod.register_forward_hook(make_hook(hp)))

        # Forward pass
        with torch.no_grad():
            model(token_ids)

        # Remove hooks
        for h in handles:
            h.remove()

        # Save this step
        step_dir = out_dir / f"step_{step:04d}"
        step_dir.mkdir(exist_ok=True)

        # Save token_ids: [B, S]
        torch.save(token_ids.cpu(), step_dir / "token_ids.pt")

        for hp in hookpoints:
            hp_data = captured[hp]
            mlp_in = hp_data["mlp_in"]          # [B*S, hidden_dim]
            router_logits = hp_data["router_logits"]  # [B*S, num_experts]

            # Reshape back to [B, S, ...]
            mlp_in = mlp_in.view(B, S, -1)
            router_logits = router_logits.view(B, S, -1)

            if args.post_softmax:
                router_logits = router_logits.softmax(dim=-1)

            # Use sanitized hookpoint name for directory
            hp_safe = hp.replace(".", "_")
            hp_dir = step_dir / hp_safe
            hp_dir.mkdir(exist_ok=True)

            torch.save(mlp_in.to(torch.bfloat16), hp_dir / "mlp_in.pt")
            torch.save(router_logits.to(torch.bfloat16), hp_dir / "router_logits.pt")

        # Free memory
        del captured
        torch.cuda.empty_cache()

        print(f"Step {step}: saved {len(hookpoints)} layers, "
              f"token_ids {list(token_ids.shape)}, "
              f"mlp_in {[B, S, mlp_in.shape[-1]]}, "
              f"router_logits {[B, S, router_logits.shape[-1]]}")

    print(f"\nDone. {args.num_steps} steps dumped to {out_dir}")
    print(f"Structure: {out_dir}/step_XXXX/token_ids.pt")
    print(f"           {out_dir}/step_XXXX/<hookpoint>/mlp_in.pt")
    print(f"           {out_dir}/step_XXXX/<hookpoint>/router_logits.pt")


if __name__ == "__main__":
    main()
