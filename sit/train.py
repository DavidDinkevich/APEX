# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import sys

from apex_loss import apex_training_loss
from apex_sample import initial_latents, prefix_latents, stack_batch_as_rows, staggered_euler
from apex_schedule import group_channel_sizes
from datasets.img_latent_dataset import ImgLatentDataset
from models import SiT_models
import wandb_utils

SAMPLE_CLASS_LABELS = (207, 360, 387, 974, 88, 979, 417, 279)
SAMPLE_CFG_SCALE = 4.0
SAMPLE_STEPS = 50


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def parse_level_channels(value):
    channels = tuple(int(part) for part in value.split(","))
    assert len(channels) >= 1, value
    return channels


def should_log_generations(step):
    return step == 100 or step % 500 == 0


def load_decoder(config_path, ckpt_path, device):
    config_path = os.path.abspath(config_path)
    ckpt_path = os.path.abspath(ckpt_path)
    assert os.path.isfile(config_path), config_path
    assert os.path.isfile(ckpt_path), ckpt_path
    vae_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vae"))
    sys.path.insert(0, vae_root)
    cwd = os.getcwd()
    os.chdir(vae_root)
    try:
        from omegaconf import OmegaConf
        from sgm.util import instantiate_from_config

        config = OmegaConf.load(config_path)
        decoder = instantiate_from_config(config.model)
        decoder.apply_ckpt(ckpt_path)
    finally:
        os.chdir(cwd)
    decoder.to(device).eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder


@torch.no_grad()
def log_level_generations(ema, decoder, latent_mean, latent_std, args, device, step):
    n = len(SAMPLE_CLASS_LABELS)
    z = initial_latents(n, args.level_channels, args.latent_size, args.latent_size, device)
    y = torch.tensor(SAMPLE_CLASS_LABELS, device=device)
    z = torch.cat([z, z], 0)
    y_null = torch.full((n,), args.num_classes, device=device, dtype=y.dtype)
    y = torch.cat([y, y_null], 0)
    samples = staggered_euler(
        ema.forward_with_cfg,
        z,
        dict(y=y, cfg_scale=SAMPLE_CFG_SCALE),
        level_channels=args.level_channels,
        gap=args.gap,
        num_steps=SAMPLE_STEPS,
    )
    samples, _ = samples.chunk(2, dim=0)
    mean = latent_mean.to(device=device, dtype=samples.dtype)
    std = latent_std.to(device=device, dtype=samples.dtype)
    decoder.to(device)
    try:
        panels = [decoder.decode(prefix * std + mean) for prefix in prefix_latents(samples, args.level_channels)]
        strip = torch.cat(panels, dim=-1).cpu()
    finally:
        decoder.to("cpu")
        torch.cuda.empty_cache()
    wandb_utils.log_image(stack_batch_as_rows(strip), step)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if args.wandb:
        assert args.vae_config is not None, "wandb sampling requires --vae-config"
        assert args.vae_ckpt is not None, "wandb sampling requires --vae-ckpt"
        assert os.path.isfile(args.vae_config), args.vae_config
        assert os.path.isfile(args.vae_ckpt), args.vae_ckpt

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        if args.experiment_name is not None:
            experiment_name = args.experiment_name
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
            experiment_name = f"{experiment_index:03d}-{model_string_name}-apex"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    in_channels = args.level_channels[-1]
    model = SiT_models[args.model](
        input_size=args.latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        level_channels=args.level_channels,
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[device])
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    train_steps = 0
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        assert "train_steps" in checkpoint, "checkpoint is missing train_steps"
        train_steps = int(checkpoint["train_steps"])
    else:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights

    dataset = ImgLatentDataset(args.data_path)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} latents ({args.data_path})")
    latent_mean = dataset._latent_mean
    latent_std = dataset._latent_std
    decoder = None
    if args.wandb and rank == 0:
        decoder = load_decoder(args.vae_config, args.vae_ckpt, torch.device("cpu"))

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    num_levels = len(group_channel_sizes(args.level_channels))
    running_level_losses = torch.zeros(num_levels, device=device)
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            assert x.shape[1:] == (in_channels, args.latent_size, args.latent_size), tuple(x.shape)
            loss, level_losses = apex_training_loss(
                model,
                x,
                y,
                level_channels=args.level_channels,
                gap=args.gap,
                p_l0=args.p_l0,
            )
            assert level_losses.shape == (num_levels,), (level_losses.shape, num_levels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            running_level_losses += level_losses
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_level = running_level_losses / log_steps
                dist.all_reduce(avg_level, op=dist.ReduceOp.SUM)
                avg_level = avg_level / dist.get_world_size()
                level_text = " ".join(
                    f"level_{level}={value:.4f}" for level, value in enumerate(avg_level.tolist())
                )
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, {level_text}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if args.wandb:
                    stats = {"train loss": avg_loss, "train steps/sec": steps_per_sec}
                    for level, value in enumerate(avg_level.tolist()):
                        stats[f"learning_level/level_{level}_loss"] = value
                    wandb_utils.log(stats, step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                running_level_losses.zero_()
                log_steps = 0
                start_time = time()

            if args.wandb and rank == 0 and should_log_generations(train_steps):
                logger.info(f"(step={train_steps:07d}) Logging level generations")
                log_level_generations(ema, decoder, latent_mean, latent_std, args, device, train_steps)

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/1")
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--level-channels", type=parse_level_channels, default="8,32,128")
    parser.add_argument("--gap", type=float, default=0.25)
    parser.add_argument("--p-l0", type=float, default=0.9)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--vae-config", type=str, default=None)
    parser.add_argument("--vae-ckpt", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    args = parser.parse_args()
    assert args.gap > 0.0, args.gap
    assert 0.0 < args.p_l0 <= 1.0, args.p_l0
    main(args)
