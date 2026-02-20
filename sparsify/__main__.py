import os
import warnings
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import cpu_count
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from simple_parsing import field, parse
from torch.distributed.tensor import distribute_module, init_device_mesh
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)
from torch.distributed.tensor import DTensor
from transformers.models import gpt_oss

from .data import MemmapDataset, chunk_and_tokenize
from .activation_store import cache_activations, CachedActivationDataset
from .trainer import TrainConfig, Trainer
from .utils import DISTRIBUTE_MODEL

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# Suppress Pydantic warnings from simple_parsing's internal implementation
warnings.filterwarnings(
    "ignore",
    message=".*'repr' attribute.*has no effect in the context it was used.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*'frozen' attribute.*has no effect in the context it was used.*",
)

original_apply_rotary_emb = gpt_oss.modeling_gpt_oss._apply_rotary_emb


def _maybe_to_local(x):
    if isinstance(x, DTensor):
        return x.to_local(), x.device_mesh, x.placements
    return x, None, None


def patched_apply_rotary_emb(q, cos, sin):
    q_local, q_mesh, q_places = _maybe_to_local(q)
    cos_local, _, _ = _maybe_to_local(cos)
    sin_local, _, _ = _maybe_to_local(sin)

    out_local = original_apply_rotary_emb(q_local, cos_local, sin_local)

    if q_mesh is not None:
        return DTensor.from_local(out_local, q_mesh, q_places)
    return out_local


gpt_oss.modeling_gpt_oss._apply_rotary_emb = patched_apply_rotary_emb


@dataclass
class RunConfig(TrainConfig):
    model: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        positional=False,
    )
    """Name of the model to train."""

    dataset: str = field(
        default="/share/nlp/liuyang/workspace/RouterScope/data/openwebtext",
        positional=False,
    )
    """Path to the dataset to use for training."""

    split: str = "train"
    """Dataset split to use for training."""

    ctx_len: int = 2048
    """Context length to use for training."""

    return_overflowed_tokens: bool = True
    """Whether to return overflowed tokens from the dataset."""

    # Use a dummy encoding function to prevent the token from being saved
    # to disk in plain text
    hf_token: str | None = field(default=None, encoding_fn=lambda _: None)
    """Huggingface API token for downloading models."""

    revision: str | None = None
    """Model revision to use for training."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_examples: int | None = None
    """Maximum number of examples to use for training."""

    resume: bool = False
    """Whether to try resuming from the checkpoint present at `checkpoints/run_name`."""

    text_column: str = "text"
    """Column name to use for text data."""

    shuffle_seed: int = 42
    """Random seed for shuffling the dataset."""

    data_preprocessing_num_proc: int = 128
    """Number of processes to use for preprocessing data"""


def load_artifacts(
    args: RunConfig, rank: int, limit_before_processing: bool = False
) -> tuple[PreTrainedModel, Dataset | MemmapDataset, AutoTokenizer | None]:
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    # End-to-end training requires a model with a causal LM head
    if args.loss_fn == "fvu":
        load_causal_lm = False
    else:
        load_causal_lm = True

    model_cls = AutoModel if args.loss_fn == "fvu" else AutoModelForCausalLM
    # from liger_kernel.transformers import AutoLigerKernelForCausalLM
    # model_cls = AutoLigerKernelForCausalLM

    model = model_cls.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
            if args.load_in_8bit
            else None
        ),
        revision=args.revision,
        dtype=dtype,
        token=args.hf_token,
    )

    # Disable torch.compile when using DTensor to avoid FakeTensorMode conflicts
    # if not (torch.distributed.is_initialized() and DISTRIBUTE_MODEL):
    # model = torch.compile(model, mode="default", dynamic=True)
    # else:
    # Use flash_attention_2 when available for faster attention, fall back to sdpa
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            model.config._attn_implementation = "flash_attention_2"
        else:
            model.config._attn_implementation = "sdpa"
    except ImportError:
        model.config._attn_implementation = "sdpa"
    model.config.use_cache = False

    # For memmap-style datasets
    if args.dataset.endswith(".bin"):
        dataset = MemmapDataset(args.dataset, args.ctx_len, args.max_examples)
    else:
        # For Huggingface datasets
        try:
            dataset = load_dataset(
                args.dataset,
                split=args.split,
                # TODO: Maybe set this to False by default? But RPJ requires it.
            )
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                dataset = Dataset.load_from_disk(args.dataset, keep_in_memory=False)
            else:
                raise e
        if limit_before_processing:
            dataset = dataset.select(range(args.max_examples))

        assert isinstance(dataset, Dataset)
        if "input_ids" not in dataset.column_names:
            tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
            dataset = chunk_and_tokenize(
                dataset,
                tokenizer,
                max_seq_len=args.ctx_len,
                num_proc=args.data_preprocessing_num_proc,
                text_key=args.text_column,
                return_overflowed_tokens=args.return_overflowed_tokens,
            )
        else:
            print("Dataset already tokenized; skipping tokenization.")

        dataset = dataset.shuffle(args.shuffle_seed)

        dataset = dataset.with_format("torch")
        if limit := args.max_examples:
            dataset = dataset.select(range(limit))

    return model, dataset, tokenizer


def run():
    args = parse(RunConfig)

    local_rank = os.environ.get("LOCAL_RANK")
    distributed = local_rank is not None
    rank = int(local_rank) if distributed else 0

    if distributed:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            device_id=torch.device(rank),
            timeout=timedelta(weeks=1),
        )
        dist.barrier()
        world_size = dist.get_world_size()
        assert world_size % args.tp == 0, "world_size must be divisible by tp"
        mesh = init_device_mesh(
            "cuda",
            (world_size // args.tp, args.tp),
            mesh_dim_names=("dp", "tp"),
        )
        dp_rank = mesh.get_coordinate()[0]  # type: ignore
        dp_size = world_size // args.tp

        if rank == 0:
            print(
                f"Using DP({dp_size}) * TP({args.tp}) across {dist.get_world_size()} GPUs."
            )

        dist.barrier()
    else:
        mesh = None
        dp_rank = 0

    # ── Cached training path ──
    if args.use_cached and args.cache_dir:
        with nullcontext() if rank == 0 else redirect_stdout(None):
            _run_cached(args, rank, distributed, mesh, dp_rank)
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    # ── Phase 1: Cache activations if cache_dir is set but cache doesn't exist ──
    if args.cache_dir and not os.path.exists(os.path.join(args.cache_dir, "metadata.json")):
        with nullcontext() if rank == 0 else redirect_stdout(None):
            if rank == 0:
                print(f"Phase 1: Caching activations to {args.cache_dir}")
                model, dataset, tokenizer = load_artifacts(args, rank)
                cache_activations(
                    model=model,
                    dataset=dataset,
                    hookpoint_patterns=args.hookpoints,
                    layers=args.layers,
                    layer_stride=args.layer_stride,
                    save_dir=args.cache_dir,
                    batch_size=args.batch_size,
                    ctx_len=args.ctx_len,
                    transcode=args.sae.transcode,
                    filter_bos=args.filter_bos,
                    remove_first_token=args.remove_first_token,
                    max_examples=args.max_examples,
                )
                del model
                torch.cuda.empty_cache()
                print("Phase 1 complete. Model deleted, GPU memory freed.")

        if distributed:
            dist.barrier()

        # Now run Phase 2 with cached activations
        with nullcontext() if rank == 0 else redirect_stdout(None):
            _run_cached(args, rank, distributed, mesh, dp_rank)
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    # ── Original (non-cached) training path ──
    with nullcontext() if rank == 0 else redirect_stdout(None):
        if not distributed or rank == 0:
            model, dataset, tokenizer = load_artifacts(args, rank)
        if distributed:
            dist.barrier()
            if rank != 0:
                model, dataset, tokenizer = load_artifacts(args, rank)
            dist.barrier()

            if DISTRIBUTE_MODEL:
                model = distribute_module(
                    model,
                    mesh,
                )

            example_world_size = mesh.shape[0] if DISTRIBUTE_MODEL else world_size
            remainder_examples = len(dataset) % example_world_size
            dataset = dataset.select(range(len(dataset) - remainder_examples))

            dataset = dataset.shard(example_world_size, dp_rank)

            remainder_examples = len(dataset) % example_world_size
            dataset = dataset.select(range(len(dataset) - remainder_examples))

        print(f"Training on '{args.dataset}' (split '{args.split}')")
        print(f"Storing model weights in {model.dtype}")

        trainer = Trainer(args, dataset, model, tokenizer, mesh)
        if args.resume:
            trainer.load_state(f"checkpoints/{args.run_name}")
        elif args.finetune:
            for name, sae in trainer.saes.items():
                if not os.path.exists(f"{args.finetune}/{name}"):
                    repo_path = snapshot_download(
                        args.finetune,
                        allow_patterns=f"{name}/*",
                    )
                    sae.load_state(
                        Path(repo_path) / name,
                    )
                else:
                    sae.load_state(
                        f"{args.finetune}/{name}",
                    )

        trainer.fit()

        if distributed:
            dist.barrier()
            dist.destroy_process_group()


def _run_cached(args: RunConfig, rank: int, distributed: bool,
                mesh, dp_rank: int):
    """Phase 2: Train SAEs from cached activations without loading the base model."""
    print(f"Phase 2: Training from cached activations at {args.cache_dir}")

    cached_dataset = CachedActivationDataset(
        args.cache_dir, args.ctx_len, args.max_examples
    )

    if distributed:
        world_size = dist.get_world_size()
        example_world_size = mesh.shape[0] if DISTRIBUTE_MODEL else world_size
        remainder = len(cached_dataset) % example_world_size
        if remainder > 0:
            cached_dataset = cached_dataset.select(
                range(len(cached_dataset) - remainder)
            )
        cached_dataset = cached_dataset.shard(example_world_size, dp_rank)

    # We still need a model to initialize the Trainer (for hookpoint resolution
    # and width detection). Load it temporarily on CPU to avoid GPU memory usage,
    # then build the trainer with a minimal model reference.
    # However, the Trainer.__init__ needs model on GPU for resolve_widths.
    # Instead, we load the model, init the trainer, then delete the model.
    from transformers import AutoModel, AutoModelForCausalLM

    model_cls = AutoModel if args.loss_fn == "fvu" else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"},
        revision=args.revision,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else "auto",
        token=args.hf_token,
    )
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            model.config._attn_implementation = "flash_attention_2"
        else:
            model.config._attn_implementation = "sdpa"
    except ImportError:
        model.config._attn_implementation = "sdpa"
    model.config.use_cache = False

    if distributed and DISTRIBUTE_MODEL:
        from torch.distributed.tensor import distribute_module
        model = distribute_module(model, mesh)

    # Use the cached dataset's hookpoints to override config
    args.hookpoints = cached_dataset.hookpoints

    trainer = Trainer(args, cached_dataset, model, None, mesh)

    # Free the base model from GPU
    del model
    torch.cuda.empty_cache()
    print("Base model deleted — GPU memory freed for SAE training")

    if args.resume:
        trainer.load_state(f"checkpoints/{args.run_name}")
    elif args.finetune:
        for name, sae in trainer.saes.items():
            if not os.path.exists(f"{args.finetune}/{name}"):
                repo_path = snapshot_download(
                    args.finetune,
                    allow_patterns=f"{name}/*",
                )
                sae.load_state(Path(repo_path) / name)
            else:
                sae.load_state(f"{args.finetune}/{name}")

    trainer.fit_cached(cached_dataset)


if __name__ == "__main__":
    run()
