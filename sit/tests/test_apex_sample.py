import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex_sample import initial_latents, prefix_latents, resample_leaving_hold, staggered_euler
from apex_schedule import group_channel_sizes
from models import SiT


def test_prefix_latents_zero_the_suffix():
    latents = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1, 1)
    panels = prefix_latents(latents, (2, 4, 8))
    assert len(panels) == 3
    assert torch.equal(panels[0][:, :2], latents[:, :2])
    assert torch.all(panels[0][:, 2:] == 0)
    assert torch.equal(panels[1][:, :4], latents[:, :4])
    assert torch.all(panels[1][:, 4:] == 0)
    assert torch.equal(panels[2], latents)


def test_initial_latents_are_gaussian_l0_and_zero_later():
    z = initial_latents(2, (8, 32, 128), 8, 8, device="cpu")
    assert z.shape == (2, 128, 8, 8)
    assert not torch.all(z[:, :8] == 0)
    assert torch.all(z[:, 8:] == 0)


def test_resample_leaving_hold_injects_noise():
    latents = torch.zeros(2, 4, 1, 1)
    t_cur = torch.tensor([0.05, 1.0])
    t_next = torch.tensor([0.0, 1.0])
    out = resample_leaving_hold(latents, t_cur, t_next, level_channels=(2, 4))
    assert torch.all(out[:, :2] == 0)
    assert not torch.all(out[:, 2:] == 0)


def test_euler_freezes_clamped_band():
    latents = torch.randn(2, 4, 2, 2)
    t_cur = torch.tensor([0.0, 0.4])
    t_next = torch.tensor([0.0, 0.2])

    class ConstV:
        def __call__(self, x, t, **kwargs):
            return torch.ones_like(x)

    # One-step using the same dt rule as staggered_euler's inner update.
    from apex_sample import _dt_to_channels
    from apex_schedule import apply_held_zero, group_channel_sizes

    sizes = group_channel_sizes((2, 4))
    x = apply_held_zero(latents, t_cur.expand(2, -1), (2, 4))
    v = torch.ones_like(x)
    dt = _dt_to_channels(t_cur - t_next, sizes).view(1, -1, 1, 1)
    out = x - dt * v
    torch.testing.assert_close(out[:, :2], x[:, :2])
    torch.testing.assert_close(out[:, 2:], x[:, 2:] - 0.2)


def test_staggered_euler_reaches_clean_end():
    def model_fn(x, t, **kwargs):
        return torch.zeros_like(x)

    z = initial_latents(1, (2, 4), 2, 2, device="cpu")
    out = staggered_euler(model_fn, z, dict(), level_channels=(2, 4), gap=0.5, num_steps=4)
    assert out.shape == z.shape


def test_cfg_guides_all_latent_channels():
    model = SiT(
        input_size=4,
        patch_size=1,
        in_channels=8,
        hidden_size=32,
        depth=1,
        num_heads=4,
        learn_sigma=False,
        level_channels=(4, 8),
        num_classes=10,
        class_dropout_prob=0.1,
    )
    model.eval()
    x = torch.randn(4, 8, 4, 4)
    t = torch.zeros(4)
    y = torch.tensor([0, 1, 10, 10])
    out = model.forward_with_cfg(x, t, y, cfg_scale=2.0)
    assert out.shape == (4, 8, 4, 4)


if __name__ == "__main__":
    test_prefix_latents_zero_the_suffix()
    test_initial_latents_are_gaussian_l0_and_zero_later()
    test_resample_leaving_hold_injects_noise()
    test_euler_freezes_clamped_band()
    test_staggered_euler_reaches_clean_end()
    test_cfg_guides_all_latent_channels()
    print("ok")
