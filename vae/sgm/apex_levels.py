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


def level_log_name(channels: int, target: int) -> str:
    assert channels > 0
    assert target > 0
    return f"c{channels}_{target}"


def level_recon_strip(panels: Sequence[torch.Tensor]) -> torch.Tensor:
    assert len(panels) >= 1
    first = panels[0]
    assert all(panel.shape[:-1] == first.shape[:-1] for panel in panels)
    return torch.cat(list(panels), dim=-1)


def level_recon_sheet(panels: Sequence[torch.Tensor]) -> torch.Tensor:
    strip = level_recon_strip(panels)
    assert strip.ndim == 4
    assert strip.shape[0] >= 1
    rows = torch.cat([strip[index] for index in range(strip.shape[0])], dim=-2)
    return rows.unsqueeze(0)


def level_recon_panel(reconstruction: torch.Tensor, image: torch.Tensor, target: int) -> torch.Tensor:
    assert reconstruction.shape == image.shape
    _, _, height, width = image.shape
    assert height == width
    recon = box_downsample_to(reconstruction, target)
    gt = box_downsample_to(image, target)
    if target < height:
        recon = F.interpolate(recon, size=(height, height), mode="nearest")
        gt = F.interpolate(gt, size=(height, height), mode="nearest")
    return torch.cat([recon, gt], dim=-1)
