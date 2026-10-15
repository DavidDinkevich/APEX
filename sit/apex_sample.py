from typing import Sequence

import torch

from apex_schedule import (
    apply_held_zero,
    clock_grid,
    group_channel_sizes,
    hold_zero_mask,
    level_times_from_clock,
    prefix_channel_counts,
)


def _dt_to_channels(level_dt: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
    pieces = [level_dt[i : i + 1].expand(width) for i, width in enumerate(sizes)]
    return torch.cat(pieces, dim=0)


def resample_leaving_hold(
    latents: torch.Tensor,
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    level_channels: Sequence[int],
    generator=None,
) -> torch.Tensor:
    sizes = group_channel_sizes(level_channels)
    hold_cur = hold_zero_mask(t_cur)
    hold_next = hold_zero_mask(t_next)
    out = latents.clone()
    start = sizes[0]
    for level in range(1, len(sizes)):
        width = sizes[level]
        end = start + width
        if bool(hold_cur[level] and (not hold_next[level])):
            out[:, start:end] = torch.randn(
                out[:, start:end].shape,
                device=out.device,
                dtype=out.dtype,
                generator=generator,
            )
        start = end
    return out


def staggered_euler(
    model_fn,
    latents: torch.Tensor,
    model_kwargs: dict,
    level_channels: Sequence[int],
    gap: float,
    num_steps: int,
) -> torch.Tensor:
    sizes = group_channel_sizes(level_channels)
    assert latents.ndim == 4
    assert latents.shape[1] == sum(sizes)
    batch = latents.shape[0]
    grid = clock_grid(len(sizes), gap, num_steps).to(device=latents.device, dtype=latents.dtype)
    x = latents
    for step in range(num_steps):
        t_cur = level_times_from_clock(grid[step : step + 1], len(sizes), gap)[0]
        t_next = level_times_from_clock(grid[step + 1 : step + 2], len(sizes), gap)[0]
        x = resample_leaving_hold(x, t_cur, t_next, level_channels)
        x = apply_held_zero(x, t_cur.expand(batch, -1), level_channels)
        t0 = t_cur[0].expand(batch)
        velocity = model_fn(x, t0, **model_kwargs)
        dt = _dt_to_channels(t_cur - t_next, sizes).view(1, -1, 1, 1)
        x = x - dt * velocity
        x = apply_held_zero(x, t_next.expand(batch, -1), level_channels)
    return x


def stack_batch_as_rows(strip: torch.Tensor) -> torch.Tensor:
    assert strip.ndim == 4
    assert strip.shape[0] >= 1
    rows = torch.cat([strip[index] for index in range(strip.shape[0])], dim=-2)
    return rows.unsqueeze(0)


def prefix_latents(latents: torch.Tensor, level_channels: Sequence[int]) -> list[torch.Tensor]:
    prefixes = prefix_channel_counts(level_channels)
    assert latents.ndim == 4
    assert latents.shape[1] == prefixes[-1], (latents.shape, prefixes)
    panels = []
    for keep in prefixes:
        panel = latents.clone()
        panel[:, keep:] = 0
        panels.append(panel)
    return panels


def initial_latents(batch: int, level_channels: Sequence[int], height: int, width: int, device, dtype=torch.float32):
    sizes = group_channel_sizes(level_channels)
    z = torch.zeros(batch, sum(sizes), height, width, device=device, dtype=dtype)
    z[:, : sizes[0]] = torch.randn(batch, sizes[0], height, width, device=device, dtype=dtype)
    return z
