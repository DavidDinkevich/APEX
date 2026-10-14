import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import apex_loss
from apex_loss import apex_training_loss
from apex_schedule import group_channel_sizes, train_mask


class _ZeroVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t, y):
        return self.scale * x


def test_per_level_loss_is_masked_group_mean():
    level_times = torch.tensor([[0.2, 0.45, 1.0], [0.0, 0.25, 0.5]])
    original = apex_loss.sample_level_times
    apex_loss.sample_level_times = lambda *args, **kwargs: level_times
    try:
        torch.manual_seed(0)
        latents = torch.ones(2, 8, 2, 2)
        labels = torch.zeros(2, dtype=torch.long)
        loss, level_losses = apex_training_loss(_ZeroVelocity(), latents, labels, (2, 4, 8), 0.25, 0.9)
        torch.manual_seed(0)
        noise = torch.randn_like(latents)
    finally:
        apex_loss.sample_level_times = original

    sizes = group_channel_sizes((2, 4, 8))
    target = noise - latents
    pred = torch.zeros_like(latents)
    mask = train_mask(level_times)
    start = 0
    group_mse = []
    for width in sizes:
        end = start + width
        mse = (pred[:, start:end] - target[:, start:end]).square().reshape(latents.shape[0], -1).mean(dim=1)
        group_mse.append(mse)
        start = end
    group_mse = torch.stack(group_mse, dim=1)
    mass = mask.sum(dim=1).clamp(min=1.0)
    expected_loss = (group_mse * mask).sum(dim=1).div(mass).mean()
    expected_levels = (group_mse * mask).mean(dim=0)

    assert mask[0, 2].item() == 0.0
    assert mask[1, 2].item() == 1.0
    assert loss.requires_grad
    assert not level_losses.requires_grad
    assert level_losses.shape == (3,)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(level_losses, expected_levels)
    torch.testing.assert_close(level_losses[2], group_mse[1, 2] / 2)


if __name__ == "__main__":
    test_per_level_loss_is_masked_group_mean()
    print("ok")
