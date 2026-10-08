import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex_schedule import (
    apply_held_zero,
    clock_grid,
    group_channel_sizes,
    hold_zero_mask,
    level_times_from_clock,
    train_mask,
)


def test_group_sizes_from_prefixes():
    assert group_channel_sizes([8, 32, 128]) == (8, 24, 96)


def test_l0_is_never_held():
    times = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.75, 1.0]])
    hold = hold_zero_mask(times)
    assert torch.all(~hold[:, 0])


def test_hold_when_t_is_one_and_predecessor_positive():
    times = torch.tensor([[0.5, 1.0, 1.0]])
    hold = hold_zero_mask(times)
    assert torch.equal(hold, torch.tensor([[False, True, True]]))


def test_handoff_t_equals_one_with_clean_predecessor_is_not_held():
    times = torch.tensor([[0.0, 1.0, 1.0]])
    hold = hold_zero_mask(times)
    assert torch.equal(hold, torch.tensor([[False, False, True]]))


def test_train_mask_drops_held_groups():
    times = torch.tensor([[0.5, 1.0]])
    assert torch.equal(train_mask(times), torch.tensor([[1.0, 0.0]]))


def test_clock_plus_gap_makes_later_levels_noisier():
    clock = torch.tensor([0.5])
    times = level_times_from_clock(clock, num_levels=3, gap=0.25)
    torch.testing.assert_close(times, torch.tensor([[0.5, 0.75, 1.0]]))


def test_clock_grid_starts_at_noise_and_ends_clean():
    grid = clock_grid(num_levels=3, gap=0.25, num_steps=10)
    assert grid.shape == (11,)
    assert torch.isclose(grid[0], torch.tensor(1.0))
    assert torch.isclose(grid[-1], torch.tensor(-0.5))
    times_start = level_times_from_clock(grid[:1], 3, 0.25)
    times_end = level_times_from_clock(grid[-1:], 3, 0.25)
    torch.testing.assert_close(times_start, torch.ones(1, 3))
    torch.testing.assert_close(times_end, torch.zeros(1, 3))


def test_apply_held_zero_zeros_held_groups_only():
    latents = torch.ones(2, 4, 1, 1)
    times = torch.tensor([[0.5, 1.0], [0.0, 1.0]])
    out = apply_held_zero(latents, times, level_channels=(2, 4))
    torch.testing.assert_close(out[0, :2], torch.ones(2, 1, 1))
    torch.testing.assert_close(out[0, 2:], torch.zeros(2, 1, 1))
    torch.testing.assert_close(out[1], torch.ones(4, 1, 1))


if __name__ == "__main__":
    test_group_sizes_from_prefixes()
    test_l0_is_never_held()
    test_hold_when_t_is_one_and_predecessor_positive()
    test_handoff_t_equals_one_with_clean_predecessor_is_not_held()
    test_train_mask_drops_held_groups()
    test_clock_plus_gap_makes_later_levels_noisier()
    test_clock_grid_starts_at_noise_and_ends_clean()
    test_apply_held_zero_zeros_held_groups_only()
    print("ok")
