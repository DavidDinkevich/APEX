"""Write one ImgLatentDataset shard of random latents."""
import argparse
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.img_latent_dataset import ImgLatentDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    shard = os.path.join(args.output, "latents_rank00_shard000.safetensors")
    if not os.path.isfile(shard):
        generator = torch.Generator().manual_seed(0)
        count = 4096
        latents = torch.randn(count, 128, 8, 8, generator=generator)
        latents_flip = torch.randn(count, 128, 8, 8, generator=generator)
        labels = torch.randint(0, 1000, (count,), generator=generator)
        save_file(
            {"latents": latents, "latents_flip": latents_flip, "labels": labels},
            shard,
        )
        print(f"Wrote {shard}")
    else:
        print(f"Reuse {shard}")
    if not os.path.isfile(os.path.join(args.output, "latents_stats.pt")):
        ImgLatentDataset(args.output)
        print(f"Wrote latents_stats.pt in {args.output}")
    else:
        print(f"Reuse {args.output}/latents_stats.pt")


if __name__ == "__main__":
    main()
