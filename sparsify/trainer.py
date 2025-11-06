import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
from fnmatch import fnmatchcase
from functools import partial
from glob import glob
from typing import Sized

import torch
import torch.distributed as dist
from datasets import Dataset as HfDataset
from natsort import natsorted
from schedulefree import ScheduleFreeWrapper, ScheduleFreeWrapperReference
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental import implicit_replication
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from .config import TrainConfig
from .data import MemmapDataset
from .fused_encoder import DeadLatentLoss

# from .nanogpt import Muon
from .muon import Muon
from .runner import CrossLayerRunner
from .sign_sgd import SignSGD
from .sparse_coder import ForwardOutput, SparseCoder
from .utils import (
    DISTRIBUTE_MODEL,
    get_layer_list,
    load_sharded,
    resolve_widths,
    save_sharded,
    set_submodule,
    unflatten_dict,
)

ScheduleFreeWrapperType = (ScheduleFreeWrapper, ScheduleFreeWrapperReference)


class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, num_training_steps):
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.warmup_steps)),
        )


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        dataset: HfDataset | MemmapDataset,
        model: PreTrainedModel,
        mesh: DeviceMesh | None = None,
    ):
        self.mesh = mesh
        # Store the whole model, including any potential causal LM wrapper
        self.model = model

        if cfg.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for base model.")

        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.base_model.named_modules():
                if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                    raw_hookpoints.append(name)

            # Natural sort to impose a consistent order
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            # If no layers are specified, train on all of them
            if not cfg.layers:
                N = model.config.num_hidden_layers
                cfg.layers = list(range(0, N))

            # Now convert layers to hookpoints
            layers_name, _ = get_layer_list(model)
            cfg.hookpoints = [f"{layers_name}.{i}" for i in cfg.layers]

        cfg.hookpoints = cfg.hookpoints[:: cfg.layer_stride]

        self.cfg = cfg
        self.dataset = dataset

        device = model.device
        with self.implicit_replication():
            input_widths = resolve_widths(
                model,
                cfg.hookpoints,
                mesh=self.mesh if DISTRIBUTE_MODEL else None,
                use_input_dim=cfg.sae.transcode,
            )
            # For transcoders, also resolve output dimensions
            if cfg.sae.transcode:
                output_widths = resolve_widths(
                    model,
                    cfg.hookpoints,
                    mesh=self.mesh if DISTRIBUTE_MODEL else None,
                    use_input_dim=False,
                )
            else:
                output_widths = (
                    input_widths  # For autoencoders, input and output are the same
                )
        unique_widths = set(input_widths.values())

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}"
            )

        # Initialize all the SAEs
        print(f"Initializing SAEs with random seed(s) {cfg.init_seeds}")
        self.saes = {}
        sources = []
        for position, hook in enumerate(
            self.cfg.hookpoints,
        ):
            for seed in cfg.init_seeds:
                torch.manual_seed(seed)

                # Add suffix to the name to disambiguate multiple seeds
                name = f"{hook}/seed{seed}" if len(cfg.init_seeds) > 1 else hook
                if cfg.cross_layer > 0:
                    n_targets = cfg.cross_layer
                    n_targets = min(n_targets, len(self.cfg.hookpoints) - position)
                    sources.append(position + n_targets)
                    while sources[0] < position:
                        sources.pop(0)
                    n_sources = len(sources)
                else:
                    n_targets = 0
                    n_sources = 0
                sae_cfg = replace(
                    cfg.sae,
                    n_targets=n_targets,
                    n_sources=n_sources,
                )
                self.saes[name] = SparseCoder(
                    input_widths[hook],
                    sae_cfg,
                    device,
                    mesh=mesh,
                    # dtype=torch.float32,
                    dtype=torch.float16,
                    d_out=output_widths[hook],
                )

        assert isinstance(dataset, Sized)
        num_batches = len(dataset) // (
            cfg.batch_size * cfg.grad_acc_steps * cfg.micro_acc_steps
        )

        match cfg.optimizer:
            case "adam" | "adam8":
                from torch.optim import Adam

                if cfg.optimizer == "adam8":
                    try:
                        from torchao.optim import AdamW8bit as Adam

                        print("Using 8-bit Adam from torchao")
                    except ImportError:
                        print(
                            "torchao 8-bit Adam not available, using torch.optim.Adam"
                        )
                        print("Run `pip install torchao` for less memory usage.")

                pgs = [
                    dict(
                        params=sae.parameters(),
                        lr=torch.tensor(
                            cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5
                        ),
                    )
                    for sae in self.saes.values()
                ]
                # For logging purposes
                lrs = [f"{lr:.2e}" for lr in sorted(set(float(pg["lr"]) for pg in pgs))]

                adam = Adam(
                    pgs,
                    betas=(cfg.b1, cfg.b2),
                    **(
                        dict(bf16_stochastic_round=True)
                        if cfg.sae.dtype == "bfloat16" and cfg.optimizer == "adam8"
                        else {}
                    ),
                )
                self.optimizers = [adam]
                self.lr_schedulers = [
                    WarmupLinearSchedule(adam, cfg.lr_warmup_steps, num_batches)
                    # get_linear_schedule_with_warmup(
                    #     adam, cfg.lr_warmup_steps, num_batches
                    # )
                ]
            case "muon":
                params = [
                    p
                    for sae in self.saes.values()
                    for _, p in sorted(sae.named_parameters())
                ]
                muon_params = [p for p in params if p.ndim >= 2]
                muon_param_set = set(muon_params)
                lrs = [f"{cfg.lr or 2e-3:.2e}"]

                self.optimizers = [
                    Muon(
                        muon_params,
                        # Muon LR is independent of the number of latents
                        lr=cfg.lr or 2e-3,
                        ddp=True,
                        group=None if self.mesh is None else self.mesh.get_group(0),
                    ),
                    torch.optim.Adam(
                        [param for param in params if param not in muon_param_set],
                        lr=cfg.lr or 2e-3,
                        betas=(cfg.b1, cfg.b2),
                    ),
                ]
                self.lr_schedulers = [
                    get_linear_schedule_with_warmup(self.optimizers[0], 0, num_batches),
                    get_linear_schedule_with_warmup(
                        self.optimizers[1], cfg.lr_warmup_steps, num_batches
                    ),
                ]
            case "signum":
                pgs = [
                    dict(
                        params=sae.parameters(),
                        lr=cfg.lr or 5e-3 / (sae.num_latents / (2**14)) ** 0.5,
                    )
                    for sae in self.saes.values()
                ]
                lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]

                if self.mesh is not None:
                    ScheduleFreeWrapperType = ScheduleFreeWrapperReference
                else:
                    ScheduleFreeWrapperType = ScheduleFreeWrapper

                opt = SignSGD(pgs)
                if not cfg.force_lr_warmup:
                    opt = ScheduleFreeWrapperType(opt, momentum=0.95)
                    opt.train()

                self.optimizers = [opt]
                if cfg.force_lr_warmup:
                    self.lr_schedulers = [
                        get_linear_schedule_with_warmup(
                            opt, cfg.lr_warmup_steps, num_batches
                        )
                    ]
                else:
                    self.lr_schedulers = []
            case other:
                raise ValueError(f"Unknown optimizer '{other}'")

        print(f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}")
        self.global_step = 0
        self.update_step = 0
        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }

        num_latents = list(self.saes.values())[0].num_latents
        self.initial_k = min(num_latents, self.cfg.sae.k * self.cfg.k_anneal_mul)
        self.final_k = self.cfg.sae.k

        self.best_loss = (
            {name: float("inf") for name in self.cfg.hookpoints}
            if self.cfg.loss_fn == "fvu"
            else float("inf")
        )

        self.model.eval()

    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = self.model.device

        # Load the train state first so we can print the step number
        train_state = torch.load(
            f"{path}/state.pt", map_location=device, weights_only=True
        )
        self.global_step = train_state["global_step"]

        for file in glob(f"{path}/rank_*_state.pt"):
            rank_state = torch.load(file, map_location=device, weights_only=True)

            for k in self.cfg.hookpoints:
                if k in rank_state["num_tokens_since_fired"]:
                    self.num_tokens_since_fired[k] = rank_state[
                        "num_tokens_since_fired"
                    ][k]

                if not isinstance(rank_state["best_loss"], dict):
                    self.best_loss = rank_state["best_loss"]
                elif k in rank_state["best_loss"]:
                    self.best_loss[k] = rank_state["best_loss"][k]  # type: ignore

        print(
            f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m"
        )

        for i, scheduler in enumerate(self.lr_schedulers):
            try:
                lr_state = torch.load(
                    f"{path}/lr_scheduler_{i}.pt",
                    map_location=device,
                    weights_only=True,
                )
            except FileNotFoundError:
                print(f"No lr_scheduler_{i}.pt found at {path}, skipping")
                continue
            scheduler.load_state_dict(lr_state)

        if self.mesh is not None:
            for sae in self.saes.values():
                for param in sae.parameters():
                    param.grad = torch.zeros_like(param)
        for i, optimizer in enumerate(self.optimizers):
            opt_state_path = f"{path}/optimizer_{i}.pt"
            if not os.path.exists(opt_state_path):
                print(f"No optimizer state found at {opt_state_path}, skipping")
                continue
            # we haven't stepped the optimizer yet, so the buffers aren't filled
            # we need to perform a fake step
            for sae in self.saes.values():
                for param in sae.parameters():
                    param.grad = torch.zeros_like(param)
            optimizer.step()
            optimizer.zero_grad()
            for sae in self.saes.values():
                for param in sae.parameters():
                    param.grad = torch.zeros_like(param)
            optimizer.step()
            optimizer.zero_grad()
            if self.mesh is None:
                opt_state = torch.load(
                    opt_state_path, map_location=device, weights_only=True
                )
                opt_state = unflatten_dict(opt_state)
            else:
                opt_state = load_sharded(
                    opt_state_path,
                    optimizer.state_dict(),
                    self.mesh,
                    load_st=False,
                )
            optimizer.load_state_dict(opt_state)

        for name, sae in self.saes.items():
            if os.path.exists(f"{path}/{name}"):
                sae.load_state(f"{path}/{name}")
            else:
                print(f"No checkpoint found for {name} at {path}, skipping")

    def get_current_k(self) -> int:
        """Get the current k value based on a linear decay schedule."""
        if self.global_step >= self.cfg.k_decay_steps:
            return self.final_k

        progress = self.global_step / self.cfg.k_decay_steps
        return round(self.initial_k * (1 - progress) + self.final_k * progress)

    def set_correct_k(self):
        if self.cfg.per_layer_k:
            assert len(self.cfg.per_layer_k) == len(
                self.saes
            ), "Must specify a k for each layer"
            for i, sae in enumerate(self.saes.values()):
                sae.cfg.k = self.cfg.per_layer_k[i]
        else:
            k = self.get_current_k()
            for sae in self.saes.values():
                sae.cfg.k = k

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        # Make sure the model is frozen
        self.model.requires_grad_(False)
        self.model.eval()

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        wandb = None
        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                if self.cfg.resume and self.cfg.run_id is not None:
                    print(f'Resuming Weights & Biases run with ID "{self.cfg.run_id}"')
                    wandb.init(
                        name=self.cfg.run_name,
                        project=os.getenv("WANDB_PROJECT", "sparsify"),
                        entity=os.getenv("WANDB_ENTITY", None),
                        id=self.cfg.run_id,
                        resume="allow",
                        config=asdict(self.cfg),
                        save_code=True,
                    )
                else:
                    wandb.init(
                        name=self.cfg.run_name,
                        project=os.getenv("WANDB_PROJECT", "sparsify"),
                        entity=os.getenv("WANDB_ENTITY", None),
                        config=asdict(self.cfg),
                        save_code=True,
                    )
            except (AttributeError, ImportError):
                print("Weights & Biases not available, skipping logging.")
                print("Run `pip install -U wandb` if you want to use it.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
        num_model_params = sum(p.numel() for p in self.model.parameters())
        global_batch_size = (
            self.cfg.batch_size
            * self.cfg.ctx_len
            * self.cfg.grad_acc_steps
            * self.cfg.micro_acc_steps
        )
        print(f"Global batch size: {global_batch_size:_}")
        print(f"Number of CLT parameters: {num_sae_params:_}")
        print(f"Number of model parameters: {num_model_params:_}")

        num_batches = len(self.dataset) // (
            self.cfg.batch_size * self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
        )
        if self.global_step > 0:
            assert hasattr(self.dataset, "select"), "Dataset must implement `select`"

            n = (
                self.global_step
                * self.cfg.batch_size
                * self.cfg.grad_acc_steps
                * self.cfg.micro_acc_steps
            )
            if self.cfg.restart_epoch:
                n = 0
                num_batches += len(self.dataset) // self.cfg.batch_size
            ds = self.dataset.select(range(n, len(self.dataset)))  # type: ignore
        else:
            ds = self.dataset

        device = self.model.device
        dl = DataLoader(
            ds,  # type: ignore
            batch_size=self.cfg.batch_size,
            # NOTE: We do not shuffle here for reproducibility; the dataset should
            # be shuffled before passing it to the trainer.
            shuffle=False,
        )
        pbar = tqdm(
            desc="Training",
            disable=not rank_zero,
            initial=self.global_step,
            total=num_batches,
            leave=False,
        )

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }

        acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
        denom = acc_steps * self.cfg.wandb_log_frequency
        num_tokens_in_step = 0

        # For logging purposes
        avg_explained_variance = defaultdict(float)
        avg_explained_variance_legacy = defaultdict(float)
        avg_unexplained_variance = defaultdict(float)
        avg_unexplained_variance_legacy = defaultdict(float)
        avg_per_token_l0 = defaultdict(float)
        avg_per_sequence_l0 = defaultdict(float)
        avg_per_batch_l0 = defaultdict(float)
        avg_per_feature_l0 = defaultdict(float)
        avg_per_token_l1 = defaultdict(float)
        avg_per_sequence_l1 = defaultdict(float)
        avg_per_batch_l1 = defaultdict(float)
        avg_per_feature_l1 = defaultdict(float)
        avg_mse_loss = defaultdict(float)
        avg_norm_mse_loss = defaultdict(float)
        avg_l2_loss = defaultdict(float)
        avg_l2_ratio = defaultdict(float)
        avg_cossim = defaultdict(float)
        avg_relative_reconstruction_bias = defaultdict(float)
        seen_tokens = 0
        fvu_losses = defaultdict(float)
        avg_ce = 0.0
        avg_kl = 0.0
        avg_acc_top1 = 0.0
        avg_losses = (
            {name: float("inf") for name in self.cfg.hookpoints}
            if self.cfg.loss_fn == "fvu"
            else float("inf")
        )

        if self.cfg.loss_fn == "ce":
            batch = next(iter(dl))
            x = self.input_ids_to_mesh(batch["input_ids"])

            with self.implicit_replication():
                clean_loss = self.model(x, labels=x).loss
            if rank_zero:
                print(f"Initial CE loss: {clean_loss.item():.4f}")

            # If doing end-to-end transcoders, then we don't actually want to run the
            # modules that we're replacing
            if self.cfg.sae.transcode and self.cfg.remove_transcoded_modules:
                for point in self.cfg.hookpoints:
                    set_submodule(self.model.base_model, point, nn.Identity())

        name_to_module = {
            name: self.model.base_model.get_submodule(name)
            for name in self.cfg.hookpoints + self.cfg.hookpoints_in
        }
        module_to_name = {v: k for k, v in name_to_module.items()}

        cached_inputs = {}
        cached_outputs = {}
        runner = CrossLayerRunner()

        def record_inputs(module: nn.Module, inputs, outputs):
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            cached_inputs[module_to_name[module]] = inputs

        def record_outputs(module: nn.Module, inputs, outputs):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            cached_outputs[module_to_name[module]] = outputs

        def hook(module: nn.Module, inputs, outputs, force_loss_fn=None):
            if force_loss_fn is not None:
                loss_fn = force_loss_fn
            else:
                loss_fn = self.cfg.loss_fn

            aux_out = None

            # Maybe unpack tuple inputs and outputs
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            if isinstance(outputs, tuple):
                outputs, *aux_out = outputs

            module_name = module_to_name[module]
            if module_name in cached_outputs:
                outputs = cached_outputs.pop(module_name)
            layer_idx = self.cfg.hookpoints.index(module_name)
            if layer_idx < len(self.cfg.hookpoints_in):
                inputs = cached_inputs.pop(self.cfg.hookpoints_in[layer_idx])

            # Name may optionally contain a suffix of the form /seedN where N is an
            # integer. We only care about the part before the slash.
            name, _, _ = module_to_name[module].partition("/")

            # Remember the original output shape since we'll need it for e2e training
            out_shape = outputs.shape

            outputs_original = outputs

            # Flatten the batch and sequence dimensions (only if they are still separate)
            # Only flatten if tensors have 3 or more dimensions (batch, seq, hidden_dim)
            if outputs.ndim >= 3:
                outputs = outputs.flatten(0, 1)
            if self.cfg.sae.transcode:
                if inputs.ndim >= 3:
                    inputs = inputs.flatten(0, 1)
            else:
                inputs = outputs

            if self.mesh is not None:
                # If tensors are already DTensors from distributed model, convert to local first
                if isinstance(inputs, DTensor):
                    # Fully replicate and then convert to local
                    inputs = inputs.redistribute(
                        self.mesh, [Replicate(), Replicate()]
                    ).to_local()
                    # Re-create as DTensor with correct placements
                    inputs = DTensor.from_local(
                        inputs, self.mesh, [Replicate(), Replicate()]
                    )
                if isinstance(outputs, DTensor):
                    # Fully replicate and then convert to local
                    outputs = outputs.redistribute(
                        self.mesh, [Replicate(), Replicate()]
                    ).to_local()
                    # Re-create as DTensor with correct placements
                    outputs = DTensor.from_local(
                        outputs, self.mesh, [Replicate(), Replicate()]
                    )

                if not DISTRIBUTE_MODEL:
                    # Convert local tensors to replicated DTensors
                    if not isinstance(inputs, DTensor):
                        inputs = DTensor.from_local(
                            inputs, self.mesh, [Replicate(), Replicate()]
                        )
                    if not isinstance(outputs, DTensor):
                        outputs = DTensor.from_local(
                            outputs, self.mesh, [Replicate(), Replicate()]
                        )
                    bos_mask_mesh = DTensor.from_local(
                        bos_mask.flatten(0, 1), self.mesh, [Replicate(), Replicate()]
                    )
                else:
                    bos_mask_mesh = DTensor.from_local(
                        bos_mask.flatten(0, 1), self.mesh, [Replicate(), Replicate()]
                    )

                inputs = inputs.redistribute(self.mesh, [Shard(0), Replicate()])
                # Check if outputs has enough dimensions for Shard(1)
                if outputs.ndim >= 2:
                    outputs = outputs.redistribute(self.mesh, [Shard(0), Shard(1)])
                else:
                    # For 1D tensors, replicate along the second mesh dimension
                    outputs = outputs.redistribute(self.mesh, [Shard(0), Replicate()])
                bos_mask_mesh = bos_mask_mesh.redistribute(
                    self.mesh, [Shard(0), Replicate()]
                )
            else:
                bos_mask_mesh = bos_mask.flatten(0, 1)

            # On the first iteration, initialize the encoder and decoder biases
            raw = self.saes[name]
            # wrapped = maybe_wrapped[name]
            wrapped = raw
            if self.global_step == 0 and not self.cfg.finetune:
                # Ensure the preactivations are centered at initialization
                # This is mathematically equivalent to Anthropic's proposal of
                # subtracting the decoder bias
                if self.cfg.sae.transcode:
                    if self.mesh is not None and self.mesh.shape[0] == 1:
                        # fix annoying SIGSEGV
                        # Need to fully replicate inputs before computing mean to ensure correct shape
                        if isinstance(inputs, DTensor):
                            inputs_replicated = inputs.redistribute(
                                self.mesh, [Replicate(), Replicate()]
                            )
                            inputs_local = inputs_replicated.to_local()
                            mean_local = inputs_local.mean(0).to(raw.dtype)
                        else:
                            mean_local = inputs.mean(0).to(raw.dtype)
                        # Ensure mean is at least 1D for linear operation
                        if mean_local.ndim == 0:
                            mean_local = mean_local.unsqueeze(0)
                        mean = mean_local  # Keep as local tensor
                    else:
                        # Ensure inputs is fully replicated before computing mean to get correct global mean
                        if isinstance(inputs, DTensor):
                            # Debug: print shapes
                            import torch.distributed as dist

                            if dist.get_rank() == 0:
                                print(
                                    f"Debug: inputs shape before redistribute: {inputs.shape}, placements: {inputs.placements}"
                                )
                            inputs_replicated = inputs.redistribute(
                                self.mesh, [Replicate(), Replicate()]
                            )
                            if dist.get_rank() == 0:
                                print(
                                    f"Debug: inputs_replicated shape: {inputs_replicated.shape}, placements: {inputs_replicated.placements}"
                                )
                            inputs_local = inputs_replicated.to_local()
                            if dist.get_rank() == 0:
                                print(
                                    f"Debug: inputs_local shape: {inputs_local.shape}"
                                )
                            mean = inputs_local.mean(0).to(raw.dtype)
                            if dist.get_rank() == 0:
                                print(f"Debug: mean shape after mean(0): {mean.shape}")
                        else:
                            mean = inputs.mean(0).to(raw.dtype)

                    # Ensure all tensors are local before linear operation
                    mean = -mean
                    weight = wrapped.encoder.weight.data
                    bias = wrapped.encoder.bias.data * 0

                    # Convert DTensors to local if needed, fully replicating first
                    if isinstance(weight, DTensor):
                        weight = weight.redistribute(
                            self.mesh, [Replicate(), Replicate()]
                        ).to_local()
                    if isinstance(bias, DTensor):
                        # For 1D bias tensor on 2D mesh, need 2 placements
                        if self.mesh.ndim == 2:
                            bias = bias.redistribute(
                                self.mesh, [Replicate(), Replicate()]
                            ).to_local()
                        else:
                            bias = bias.redistribute(
                                self.mesh, [Replicate()]
                            ).to_local()

                    mean_image = torch.nn.functional.linear(mean, weight, bias)
                    # Convert mean_image to DTensor if encoder.bias is a DTensor
                    if isinstance(raw.encoder.bias.data, DTensor):
                        # mean_image is computed with full replicated weights, so it has full size
                        # Create as replicated DTensor first, then redistribute to match encoder.bias placement
                        mean_image = DTensor.from_local(
                            mean_image, self.mesh, [Replicate(), Replicate()]
                        ).redistribute(self.mesh, raw.encoder.bias.data.placements)
                    raw.encoder.bias.data[:] = mean_image

                if self.mesh is not None and self.mesh.shape[0] == 1:
                    # Need to fully replicate outputs before computing mean to ensure correct shape
                    if isinstance(outputs, DTensor):
                        outputs_replicated = outputs.redistribute(
                            self.mesh, [Replicate(), Replicate()]
                        )
                        mean_local = outputs_replicated.to_local().mean(0).to(raw.dtype)
                    else:
                        mean_local = outputs.mean(0).to(raw.dtype)
                    # Ensure mean is at least 1D
                    if mean_local.ndim == 0:
                        mean_local = mean_local.unsqueeze(0)
                    # mean_local is the full mean computed from replicated outputs
                    # Create as replicated DTensor, then redistribute to match decoder bias placement if needed
                    if isinstance(outputs, DTensor):
                        mean = DTensor.from_local(
                            mean_local, self.mesh, [Replicate(), Replicate()]
                        )
                    else:
                        mean = mean_local
                else:
                    # Ensure outputs is fully replicated before computing mean to get correct global mean
                    if isinstance(outputs, DTensor):
                        outputs_replicated = outputs.redistribute(
                            self.mesh, [Replicate(), Replicate()]
                        )
                        mean = outputs_replicated.to_local().mean(0)
                    else:
                        mean = outputs.mean(0)
                if not hasattr(raw, "b_decs"):
                    # Ensure mean is DTensor if b_dec is DTensor
                    if isinstance(raw.b_dec.data, DTensor) and not isinstance(
                        mean, DTensor
                    ):
                        mean = DTensor.from_local(
                            mean.to(raw.dtype), self.mesh, [Replicate(), Replicate()]
                        ).redistribute(self.mesh, raw.b_dec.data.placements)
                    else:
                        mean = mean.to(raw.dtype)
                    raw.b_dec.data[:] = mean
                else:
                    # the current layer must be what handles the bias,
                    # not the contributing previous layers
                    if isinstance(raw.b_decs[0].data, DTensor) and not isinstance(
                        mean, DTensor
                    ):
                        mean = DTensor.from_local(
                            mean.to(raw.dtype), self.mesh, [Replicate(), Replicate()]
                        ).redistribute(self.mesh, raw.b_decs[0].data.placements)
                    else:
                        mean = mean.to(raw.dtype)
                    raw.b_decs[0].data[:] = mean

                if raw.cfg.normalize_io:
                    in_norm = inputs.norm(dim=-1).mean()
                    out_norm = outputs.norm(dim=-1).mean()

                    raw.in_norm.data[:] = in_norm
                    raw.out_norm.data[:] = out_norm
                    for b_dec in raw.b_decs if hasattr(raw, "b_decs") else [raw.b_dec]:
                        b_dec.data[:] = b_dec.data * (
                            (b_dec.shape[-1] ** 0.5) / out_norm
                        )
                    raw.encoder.bias.data[:] = raw.encoder.bias.data * (
                        (raw.encoder.weight.shape[-1] ** 0.5) / in_norm
                    )

            # Make sure the W_dec is still unit-norm if we're autoencoding
            if raw.cfg.normalize_decoder and not self.cfg.sae.transcode:
                raw.set_decoder_norm_to_unit_norm()

            encoding = runner.encode(
                inputs,
                sparse_coder=raw,
                dead_mask=self.num_tokens_since_fired[name]
                > self.cfg.dead_feature_threshold,
            )
            out = runner.decode(
                encoding,
                outputs,
                module_name=name,
                detach_grad=loss_fn == "fvu",
                loss_mask=(~bos_mask_mesh if loss_fn == "fvu" else None),
            )
            output = out.y_hat

            if self.cfg.loss_fn == "fvu":
                del output
            else:
                assert isinstance(output, Tensor)
            assert isinstance(out, ForwardOutput)

            if self.cfg.loss_fn == "kl-fvu":
                fvu_losses[name] = float(out.unexplained_variance.detach().mean())

            # Update the did_fire mask
            latent_indices = encoding.latent_indices.flatten()

            if isinstance(latent_indices, DTensor):
                latent_indices = latent_indices.to_local()
            did_fire[name][latent_indices] = True
            self.maybe_all_reduce(did_fire[name], "max")

            if loss_fn in ("ce", "kl"):
                # reshard outputs
                if self.mesh is not None:
                    output = output.redistribute(
                        self.mesh, [Shard(0), Shard(0)]
                    ).to_local()
                output = output.reshape(out_shape).type_as(outputs)

                if self.cfg.filter_bos:
                    output = torch.where(
                        bos_mask[..., None],
                        outputs_original,
                        output,
                    )

                # Replace the normal output with the SAE output
                return (output, *aux_out) if aux_out is not None else output
            else:
                avg_explained_variance[name] += float(
                    out.explained_variance.detach() / denom
                )
                avg_explained_variance_legacy[name] += float(
                    out.explained_variance_legacy.detach() / denom
                )
                avg_unexplained_variance[name] += float(
                    out.unexplained_variance.detach() / denom
                )
                avg_unexplained_variance_legacy[name] += float(
                    out.unexplained_variance_legacy.detach() / denom
                )
                avg_per_token_l0[name] += float(out.per_token_l0 / denom)
                avg_per_sequence_l0[name] += float(out.per_sequence_l0 / denom)
                avg_per_batch_l0[name] += float(out.per_batch_l0.sum() / denom)
                avg_per_feature_l0[name] += float(out.per_feature_l0.mean() / denom)
                avg_per_token_l1[name] += float(out.per_token_l1 / denom)
                avg_per_sequence_l1[name] += float(out.per_sequence_l1 / denom)
                avg_per_batch_l1[name] += float(out.per_batch_l1.sum() / denom)
                avg_per_feature_l1[name] += float(out.per_feature_l1.mean() / denom)
                avg_mse_loss[name] += float(out.mse_loss / denom)
                avg_norm_mse_loss[name] += float(out.norm_mse_loss / denom)
                avg_l2_loss[name] += float(out.l2_loss / denom)
                avg_l2_ratio[name] += float(out.l2_ratio / denom)
                avg_cossim[name] += float(out.cossim / denom)
                avg_relative_reconstruction_bias[name] += float(
                    out.relative_reconstruction_bias / denom
                )

                prev_modules = [mod for mod in runner.outputs.keys() if mod != name]
                prev_modules = [self.saes[mod] for mod in prev_modules]

                dead_latent_loss = 0.0
                if self.cfg.dead_latent_penalty > 0.0:
                    if isinstance(raw.W_dec, DTensor):
                        norms = raw.W_dec.pow(2).sum(dim=-1).add(1e-10).sqrt()
                    else:
                        norms = raw.W_dec.norm(dim=-1)
                    dead_latent_loss = DeadLatentLoss.apply(
                        inputs, raw.encoder.weight, raw.encoder.bias, norms
                    )
                    active_correction = (
                        encoding.latent_acts.to_local()
                        * norms.to_local()[encoding.latent_indices.to_local()]
                    ).sum()
                    if self.mesh is not None:
                        active_correction = DTensor.from_local(
                            active_correction, self.mesh, [Replicate(), Replicate()]
                        )
                    dead_latent_loss = dead_latent_loss + active_correction

                loss = (
                    out.unexplained_variance
                    + self.cfg.dead_latent_penalty * dead_latent_loss
                ) / acc_steps

                # Do a "local" backward pass if we're not training end-to-end
                loss.backward()
            del loss

            runner.restore()

        for batch in dl:
            x = self.input_ids_to_mesh(batch["input_ids"])
            if self.model.config.bos_token_id is not None:
                bos_mask = x == self.model.config.bos_token_id
            else:
                bos_mask = torch.zeros_like(x, dtype=torch.bool)
            if not self.cfg.filter_bos:
                bos_mask[:] = False
            if self.cfg.remove_first_token:
                # bos_mask[:, 0] = True
                # Convert DTensor to local tensor if needed
                if isinstance(bos_mask, DTensor):
                    local_mask = bos_mask.to_local()
                    local_mask[:, 0] = True
                    bos_mask = DTensor.from_local(
                        local_mask,
                        device_mesh=bos_mask.device_mesh,
                        placements=bos_mask.placements,
                    )
                else:
                    bos_mask[:, 0] = True

            runner.reset()

            # Bookkeeping for dead feature detection
            N = x.numel()
            num_tokens_in_step += N

            # Compute clean logits if using KL loss
            with self.implicit_replication():
                handles = (
                    [
                        mod.register_forward_hook(
                            partial(hook, force_loss_fn="fvu")
                            if name in self.cfg.hookpoints
                            else record_inputs
                        )
                        for name, mod in name_to_module.items()
                    ]
                    if self.cfg.loss_fn == "kl-fvu"
                    else []
                )
                clean_logits = (
                    self.model(x).logits
                    if self.cfg.loss_fn in ("kl", "kl-fvu")
                    else None
                )
                for handle in handles:
                    handle.remove()
                clean_probs = (
                    clean_logits.softmax(dim=-1)
                    if self.cfg.loss_fn in ("kl", "kl-fvu")
                    else None
                )

            # Forward pass on the model to get the next batch of activations
            handles = [
                mod.register_forward_hook(
                    partial(
                        hook,
                        force_loss_fn=None if self.cfg.loss_fn != "kl-fvu" else "kl",
                    )
                    if name in self.cfg.hookpoints
                    else record_inputs
                )
                for name, mod in name_to_module.items()
            ]
            try:
                with self.implicit_replication():
                    match self.cfg.loss_fn:
                        case "ce":
                            ce = self.model(x, labels=x).loss
                            ce.div(acc_steps).backward()

                            avg_ce += float(ce.detach() / denom)

                            avg_losses = avg_ce
                        case "kl" | "kl-fvu":
                            dirty_lps = self.model(x).logits.log_softmax(dim=-1)
                            kl = torch.sum(
                                clean_probs
                                * (clean_logits.log_softmax(dim=-1) - dirty_lps),
                                dim=-1,
                            ).mean()
                            acc_top1 = (
                                (
                                    clean_logits.argmax(dim=-1)
                                    == dirty_lps.argmax(dim=-1)
                                )
                                .float()
                                .mean()
                            )
                            loss = kl
                            if self.cfg.loss_fn == "kl-fvu":
                                if not isinstance(kl, DTensor):
                                    kl = DTensor.from_local(
                                        kl.unsqueeze(0),
                                        self.mesh,
                                        [Shard(0), Replicate()],
                                    ).mean()
                                fvu_loss = sum(fvu_losses.values()) / len(fvu_losses)
                                kl_coeff = self.cfg.kl_coeff
                                if kl_coeff == 0.0:
                                    kl_coeff = (fvu_loss / kl).detach()
                                loss = kl * kl_coeff + fvu_loss
                            loss.div(acc_steps).backward()

                            avg_kl += float(kl / denom)
                            avg_acc_top1 += float(acc_top1 / denom)
                            avg_losses = avg_kl
                            fvu_losses.clear()
                        case "fvu":
                            self.model(x)
                            avg_losses = dict(avg_explained_variance)
                        case other:
                            raise ValueError(f"Unknown loss function '{other}'")
            finally:
                for handle in handles:
                    handle.remove()

            # Check if we need to actually do a training step
            step, substep = divmod(
                self.global_step + 1, self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
            )
            if substep == 0:
                if self.cfg.sae.normalize_decoder and not self.cfg.sae.transcode:
                    for sae in self.saes.values():
                        sae.remove_gradient_parallel_to_decoder_directions()

                for name, optimizer in zip(self.saes.keys(), self.optimizers):
                    optimizer.step()
                    optimizer.zero_grad()

                for scheduler in self.lr_schedulers:
                    scheduler.step()

                self.set_correct_k()

                ###############
                with torch.no_grad():
                    # Update the dead feature mask
                    for name, counts in self.num_tokens_since_fired.items():
                        counts += num_tokens_in_step
                        counts[did_fire[name]] = 0

                    # Reset stats for this step
                    num_tokens_in_step = 0
                    for mask in did_fire.values():
                        mask.zero_()

                if (step + 1) % self.cfg.save_every == 0:
                    print(f"Saving checkpoint at step {step + 1}")
                    self.save()

                    if self.cfg.save_best:
                        self.save_best(avg_losses)

                if (
                    self.cfg.log_to_wandb
                    and (step + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    info = {}
                    if self.cfg.loss_fn == "ce":
                        info["ce_loss"] = avg_ce
                    elif self.cfg.loss_fn in ("kl", "kl-fvu"):
                        info["kl_loss"] = avg_kl
                        info["acc_top1"] = avg_acc_top1

                    for name in self.saes:
                        mask = (
                            self.num_tokens_since_fired[name]
                            > self.cfg.dead_feature_threshold
                        )

                        ratio = mask.mean(dtype=torch.float32).item()
                        info.update({f"dead_pct/{name}": ratio})
                        if "fvu" in self.cfg.loss_fn:
                            info[f"explained_variance/{name}"] = avg_explained_variance[
                                name
                            ]
                            info[f"explained_variance_legacy/{name}"] = (
                                avg_explained_variance_legacy[name]
                            )
                            info[f"unexplained_variance/{name}"] = (
                                avg_unexplained_variance[name]
                            )
                            info[f"unexplained_variance_legacy/{name}"] = (
                                avg_unexplained_variance_legacy[name]
                            )

                        info[f"mse/{name}"] = avg_mse_loss[name]
                        info[f"norm_mse/{name}"] = avg_norm_mse_loss[name]

                        info[f"l0(per_token)/{name}"] = avg_per_token_l0[name]
                        info[f"l0(per_sequence)/{name}"] = avg_per_sequence_l0[name]
                        info[f"l0(per_batch)/{name}"] = avg_per_batch_l0[name]
                        info[f"l0(per_feature)/{name}"] = avg_per_feature_l0[name]
                        info[f"l1(per_token)/{name}"] = avg_per_token_l1[name]
                        info[f"l1(per_sequence)/{name}"] = avg_per_sequence_l1[name]
                        info[f"l1(per_batch)/{name}"] = avg_per_batch_l1[name]
                        info[f"l1(per_feature)/{name}"] = avg_per_feature_l1[name]
                        info[f"l2/{name}"] = avg_l2_loss[name]
                        info[f"l2_ratio/{name}"] = avg_l2_ratio[name]
                        info[f"cossim/{name}"] = avg_cossim[name]
                        info[f"relative_reconstruction_bias/{name}"] = (
                            avg_relative_reconstruction_bias[name]
                        )

                    if rank_zero:
                        info["train/k"] = self.get_current_k()
                        info["train/lr"] = self.optimizers[0].param_groups[0]["lr"]
                        info["train/global_step"] = step
                        info["train/epoch"] = (step * self.cfg.batch_size) / len(
                            self.dataset
                        )
                        info["train/seen_tokens"] = seen_tokens = (
                            seen_tokens
                            + self.cfg.batch_size
                            * self.cfg.ctx_len
                            * self.cfg.grad_acc_steps
                            * self.cfg.micro_acc_steps
                        )
                        for name in self.cfg.hookpoints:
                            info[f"loss/{name}"] = (
                                avg_losses[name]
                                if isinstance(avg_losses, dict)
                                else avg_losses
                            )

                        if wandb is not None:
                            wandb.log(info, step=step)

                avg_explained_variance.clear()
                avg_explained_variance_legacy.clear()
                avg_unexplained_variance.clear()
                avg_unexplained_variance_legacy.clear()
                avg_per_token_l0.clear()
                avg_per_sequence_l0.clear()
                avg_per_batch_l0.clear()
                avg_per_feature_l0.clear()
                avg_per_token_l1.clear()
                avg_per_sequence_l1.clear()
                avg_per_batch_l1.clear()
                avg_per_feature_l1.clear()
                avg_mse_loss.clear()
                avg_norm_mse_loss.clear()
                avg_l2_loss.clear()
                avg_l2_ratio.clear()
                avg_cossim.clear()
                avg_relative_reconstruction_bias.clear()
                avg_ce = 0.0
                avg_kl = 0.0
                avg_acc_top1 = 0.0

                pbar.update()
                self.update_step += 1

            self.global_step += 1

            if self.global_step >= self.cfg.max_steps:
                print("Early stop with reaching maximum training steps.")
                break

        self.save()
        if self.cfg.save_best:
            self.save_best(avg_losses)

        pbar.close()

    def input_ids_to_mesh(self, x: Tensor) -> Tensor:
        if self.mesh is not None and DISTRIBUTE_MODEL:
            x = DTensor.from_local(x, self.mesh, [Shard(0), Replicate()])
        else:
            x = x.to(self.model.device)
        return x

    def maybe_all_reduce(self, x: Tensor, op: str = "mean", axis: int = 0) -> Tensor:
        if not dist.is_initialized() or self.mesh is None:
            return x

        if op == "sum":
            dist_op = dist.ReduceOp.SUM
        elif op == "mean":
            dist_op = dist.ReduceOp.SUM
        elif op == "max":
            dist_op = dist.ReduceOp.MAX
        else:
            raise ValueError(f"Unknown reduction op '{op}'")
        dist.all_reduce(
            x,
            op=dist_op,
            group=(
                self.mesh.get_group("dp")
                if axis == 0
                else (self.mesh.get_group("tp") if axis == 1 else None)
            ),
        )

        if op == "mean":
            x /= self.mesh.shape[0]

        return x

    @contextmanager
    def implicit_replication(self):
        if not DISTRIBUTE_MODEL or self.mesh is None:
            yield
        else:
            # with context_parallel(self.mesh), implicit_replication():
            with implicit_replication():
                yield

    def _checkpoint(self, saes: dict[str, SparseCoder], path: str):
        """Save SAEs and training state to disk."""
        print("Saving checkpoint")

        for optimizer in self.optimizers:
            if isinstance(optimizer, ScheduleFreeWrapperType):
                optimizer.eval()

        for name, sae in saes.items():
            assert isinstance(sae, SparseCoder)

            sae.save_to_disk(f"{path}/{name}")

        for i, optimizer in enumerate(self.optimizers):
            optimizer_state_dict = optimizer.state_dict()
            if isinstance(optimizer, Muon):
                for param_group in optimizer_state_dict["param_groups"]:
                    if "update_buffer" in param_group:
                        del param_group["update_buffer"]
                    if "update_buffer_views" in param_group:
                        del param_group["update_buffer_views"]
            save_sharded(
                optimizer_state_dict,
                f"{path}/optimizer_{i}.pt",
                self.mesh,
                save_st=False,
                unflatten=True,
            )

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        if rank_zero:
            for i, scheduler in enumerate(self.lr_schedulers):
                torch.save(scheduler.state_dict(), f"{path}/lr_scheduler_{i}.pt")

            torch.save(
                {"global_step": self.update_step},
                f"{path}/state.pt",
            )

            self.cfg.save_json(f"{path}/config.json")

        for optimizer in self.optimizers:
            if isinstance(optimizer, ScheduleFreeWrapperType):
                optimizer.train()

        rank = 0 if rank_zero else dist.get_rank()
        torch.save(
            {
                "num_tokens_since_fired": self.num_tokens_since_fired,
                "best_loss": self.best_loss,
            },
            f"{path}/rank_{rank}_state.pt",
        )
        if dist.is_initialized():
            dist.barrier()

    def save(self):
        """Save the SAEs and training state to disk."""
        path = f'{self.cfg.save_dir}/{self.cfg.run_name or "unnamed"}'

        self._checkpoint(self.saes, path)

    def save_best(self, avg_loss: float | dict[str, float]):
        """Save individual sparse coders to disk if they have the lowest loss."""
        base_path = f'{self.cfg.save_dir}/{self.cfg.run_name or "unnamed"}/best'
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if isinstance(avg_loss, dict):
            for name in self.saes:
                if avg_loss[name] < self.best_loss[name]:  # type: ignore
                    self.best_loss[name] = avg_loss[name]  # type: ignore

                    if rank_zero or self.cfg.distribute_modules:
                        self._checkpoint({name: self.saes[name]}, f"{base_path}/{name}")
        else:
            if avg_loss < self.best_loss:  # type: ignore
                self.best_loss = avg_loss  # type: ignore

                if rank_zero or self.cfg.distribute_modules:
                    self._checkpoint(self.saes, base_path)

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()


# Support old name for compatibility
SaeTrainer = Trainer
