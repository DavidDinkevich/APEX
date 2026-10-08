from typing import Sequence

import torch

from apex_schedule import (
    apply_held_zero,
    group_channel_sizes,
    sample_level_times,
    train_mask,
)


def _times_to_channels(level_times: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
    pieces = [level_times[:, i : i + 1].expand(-1, width) for i, width in enumerate(sizes)]
    return torch.cat(pieces, dim=1)


def apex_training_loss(
    model,
    latents: torch.Tensor,
    labels: torch.Tensor,
    level_channels: Sequence[int],
    gap: float,
    p_l0: float,
) -> torch.Tensor:
    assert latents.ndim == 4
    sizes = group_channel_sizes(level_channels)
    assert latents.shape[1] == sum(sizes)
    batch = latents.shape[0]
    level_times = sample_level_times(batch, len(sizes), gap, p_l0, latents.device)
    noise = torch.randn_like(latents)
    t_channels = _times_to_channels(level_times, sizes).view(batch, latents.shape[1], 1, 1)
    xt = (1.0 - t_channels) * latents + t_channels * noise
    xt = apply_held_zero(xt, level_times, level_channels)
    pred = model(xt, level_times[:, 0], labels)
    target = noise - latents
    mask = train_mask(level_times)
    start = 0
    group_mse = []
    for i, width in enumerate(sizes):
        end = start + width
        mse = (pred[:, start:end] - target[:, start:end]).square().reshape(batch, -1).mean(dim=1)
        group_mse.append(mse)
        start = end
    group_mse = torch.stack(group_mse, dim=1)
    mass = mask.sum(dim=1).clamp(min=1.0)
    return (group_mse * mask).sum(dim=1).div(mass).mean()
