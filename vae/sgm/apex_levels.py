from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def assert_level_config(
    level_channels: Sequence[int],
    level_target_sizes: Sequence[int],
    latent_channels: int,
) -> None:
    assert len(level_channels) >= 1
    assert len(level_channels) == len(level_target_sizes)
    channels = tuple(int(c) for c in level_channels)
    sizes = tuple(int(s) for s in level_target_sizes)
    assert all(c > 0 for c in channels)
    assert all(s > 0 for s in sizes)
    assert channels == tuple(sorted(channels))
    assert len(set(channels)) == len(channels)
    assert sizes == tuple(sorted(sizes))
    assert len(set(sizes)) == len(sizes)
    assert channels[-1] == latent_channels, (
        f"last level_channels entry must equal latent_channels={latent_channels}, got {channels[-1]}"
    )


def keep_prefix_channels(z: torch.Tensor, num_channels: int) -> torch.Tensor:
    assert z.ndim == 4
    assert 0 < num_channels <= z.shape[1]
    out = z.clone()
    out[:, num_channels:] = 0
    return out


def box_downsample_to(x: torch.Tensor, target: int) -> torch.Tensor:
    assert x.ndim == 4
    _, _, height, width = x.shape
    assert height == width, f"BOX expects square NCHW, got {height}x{width}"
    spatial = height
    assert spatial >= target
    while spatial > target:
        assert spatial % 2 == 0, f"BOX requires even spatial size before pool, got {spatial} -> {target}"
        x = F.avg_pool2d(x, kernel_size=2)
        spatial = x.shape[-1]
    assert spatial == target
    return x


def sample_level(num_levels: int, generator: Optional[torch.Generator] = None) -> int:
    assert num_levels >= 1
    idx = torch.randint(num_levels, (1,), generator=generator)
    return int(idx.item())
