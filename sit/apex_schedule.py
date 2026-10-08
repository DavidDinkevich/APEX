from typing import Optional, Sequence, Tuple

import torch


def prefix_channel_counts(level_channels: Sequence[int]) -> Tuple[int, ...]:
    prefixes = tuple(int(c) for c in level_channels)
    assert len(prefixes) >= 1
    assert prefixes == tuple(sorted(prefixes))
    assert len(set(prefixes)) == len(prefixes)
    assert all(c > 0 for c in prefixes)
    return prefixes


def group_channel_sizes(level_channels: Sequence[int]) -> Tuple[int, ...]:
    prefixes = prefix_channel_counts(level_channels)
    sizes = (prefixes[0],) + tuple(right - left for left, right in zip(prefixes, prefixes[1:]))
    assert all(s > 0 for s in sizes)
    assert sum(sizes) == prefixes[-1]
    return sizes


def sample_clock(
    batch: int,
    num_levels: int,
    gap: float,
    p_l0: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    assert num_levels >= 2
    assert 0.0 < gap <= 1.0
    assert 0.0 < p_l0 <= 1.0
    max_offset = (num_levels - 1) * gap
    l0_active = torch.rand(batch, device=device, generator=generator) < p_l0
    c_active = torch.rand(batch, device=device, generator=generator)
    c_clean = torch.rand(batch, device=device, generator=generator) * max_offset - max_offset
    return torch.where(l0_active, c_active, c_clean)


def level_times_from_clock(clock: torch.Tensor, num_levels: int, gap: float) -> torch.Tensor:
    assert clock.ndim == 1
    assert num_levels >= 1
    assert 0.0 < gap <= 1.0 or num_levels == 1
    levels = torch.arange(num_levels, device=clock.device, dtype=clock.dtype)
    return (clock[:, None] + levels * gap).clamp(0.0, 1.0)


def sample_level_times(
    batch: int,
    num_levels: int,
    gap: float,
    p_l0: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    clock = sample_clock(batch, num_levels, gap, p_l0, device, generator)
    return level_times_from_clock(clock, num_levels, gap)


def hold_zero_mask(level_times: torch.Tensor) -> torch.Tensor:
    """Hold ℓ≥1 iff t_ℓ == 1 and t_{ℓ-1} > 0. L0 is never held."""
    assert level_times.ndim >= 1
    hold = torch.zeros_like(level_times, dtype=torch.bool)
    if level_times.shape[-1] == 1:
        return hold
    hold[..., 1:] = (level_times[..., 1:] == 1.0) & (level_times[..., :-1] > 0.0)
    return hold


def train_mask(level_times: torch.Tensor) -> torch.Tensor:
    return (~hold_zero_mask(level_times)).to(dtype=level_times.dtype)


def clock_grid(num_levels: int, gap: float, num_steps: int) -> torch.Tensor:
    assert num_levels >= 2
    assert 0.0 < gap <= 1.0
    assert num_steps >= 1
    max_offset = (num_levels - 1) * gap
    return torch.linspace(1.0, -max_offset, num_steps + 1)


def apply_held_zero(latents: torch.Tensor, level_times: torch.Tensor, level_channels: Sequence[int]) -> torch.Tensor:
    """latents: (N, C, H, W). level_times: (N, L) or (L,)."""
    sizes = group_channel_sizes(level_channels)
    assert latents.ndim == 4
    assert latents.shape[1] == sum(sizes)
    times = level_times
    if times.ndim == 1:
        times = times.expand(latents.shape[0], -1)
    assert times.shape == (latents.shape[0], len(sizes))
    hold = hold_zero_mask(times)
    out = latents.clone()
    start = 0
    for level, width in enumerate(sizes):
        end = start + width
        if level > 0:
            mask = hold[:, level].view(-1, 1, 1, 1)
            out[:, start:end] = torch.where(mask, torch.zeros_like(out[:, start:end]), out[:, start:end])
        start = end
    return out
