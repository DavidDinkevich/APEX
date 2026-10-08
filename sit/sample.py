# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample latents from an APEX-trained SiT with staggered Euler.
"""
import argparse

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from apex_sample import initial_latents, staggered_euler
from models import SiT_models
from train import parse_level_channels


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels = args.level_channels[-1]
    model = SiT_models[args.model](
        input_size=args.latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        level_channels=args.level_channels,
    ).to(device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    class_labels = args.class_labels
    n = len(class_labels)
    z = initial_latents(n, args.level_channels, args.latent_size, args.latent_size, device)
    y = torch.tensor(class_labels, device=device)
    use_cfg = args.cfg_scale > 1.0
    if use_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([args.num_classes] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_fn = model.forward_with_cfg
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    else:
        model_fn = model.forward
        model_kwargs = dict(y=y)

    samples = staggered_euler(
        model_fn,
        z,
        model_kwargs,
        level_channels=args.level_channels,
        gap=args.gap,
        num_steps=args.num_sampling_steps,
    )
    if use_cfg:
        samples, _ = samples.chunk(2, dim=0)
    torch.save(samples.cpu(), args.output)
    print(f"Wrote {tuple(samples.shape)} latents to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="sample_latents.pt")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/1")
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--level-channels", type=parse_level_channels, default="8,32,128")
    parser.add_argument("--gap", type=float, default=0.25)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--class-labels",
        type=int,
        nargs="+",
        default=[207, 360, 387, 974, 88, 979, 417, 279],
    )
    args = parser.parse_args()
    assert args.gap > 0.0, args.gap
    main(args)
