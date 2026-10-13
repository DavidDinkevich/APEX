import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "apex_levels",
    Path(__file__).resolve().parents[1] / "sgm" / "apex_levels.py",
)
_apex_levels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_apex_levels)
level_log_name = _apex_levels.level_log_name
level_recon_panel = _apex_levels.level_recon_panel
level_recon_sheet = _apex_levels.level_recon_sheet
level_recon_strip = _apex_levels.level_recon_strip


def test_level_log_name():
    assert level_log_name(8, 64) == "c8_64"
    assert level_log_name(32, 128) == "c32_128"
    assert level_log_name(128, 256) == "c128_256"


def test_level_panel_is_recon_beside_input_at_target_resolution():
    image = torch.zeros(2, 3, 8, 8)
    image[..., :, 4:] = 1
    recon = torch.ones(2, 3, 8, 8)
    panel = level_recon_panel(recon, image, target=4)
    assert panel.shape == (2, 3, 8, 16)
    assert torch.equal(panel[..., :8], torch.ones_like(panel[..., :8]))
    gt = panel[..., 8:]
    assert torch.equal(gt[..., :4], torch.zeros_like(gt[..., :4]))
    assert torch.equal(gt[..., 4:], torch.ones_like(gt[..., 4:]))


def test_levels_sit_in_one_strip_low_to_high():
    image = torch.zeros(1, 3, 4, 4)
    coarse = level_recon_panel(torch.zeros_like(image), image, target=2)
    fine = level_recon_panel(torch.ones_like(image), image, target=4)
    strip = level_recon_strip([coarse, fine])
    assert strip.shape == (1, 3, 4, 16)
    assert torch.equal(strip[..., :8], coarse)
    assert torch.equal(strip[..., 8:], fine)


def test_samples_are_stacked_as_rows():
    image = torch.zeros(2, 3, 4, 4)
    coarse = level_recon_panel(torch.zeros_like(image), image, target=2)
    fine = level_recon_panel(torch.ones_like(image), image, target=4)
    sheet = level_recon_sheet([coarse, fine])
    strip = level_recon_strip([coarse, fine])
    assert sheet.shape == (1, 3, 8, 16)
    assert torch.equal(sheet[0, :, :4], strip[0])
    assert torch.equal(sheet[0, :, 4:], strip[1])


def test_native_target_skips_upsample():
    image = torch.arange(4, dtype=torch.float32).view(1, 1, 2, 2)
    recon = -image
    panel = level_recon_panel(recon, image, target=2)
    assert panel.shape == (1, 1, 2, 4)
    assert torch.equal(panel[..., :2], recon)
    assert torch.equal(panel[..., 2:], image)
