import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import SiT


def test_zero_later_init_zeros_later_columns_and_keeps_bias_zero():
    level_channels = (8, 32, 128)
    model = SiT(
        input_size=8,
        patch_size=1,
        in_channels=128,
        hidden_size=64,
        depth=1,
        num_heads=4,
        learn_sigma=False,
        level_channels=level_channels,
    )
    weight = model.x_embedder.proj.weight.data
    c0 = level_channels[0]
    assert weight.shape[:2] == (64, 128)
    assert torch.all(weight[:, c0:] == 0)
    assert not torch.all(weight[:, :c0] == 0)
    assert torch.all(model.x_embedder.proj.bias.data == 0)


def test_stock_init_does_not_zero_later_columns():
    torch.manual_seed(0)
    model = SiT(
        input_size=8,
        patch_size=1,
        in_channels=128,
        hidden_size=64,
        depth=1,
        num_heads=4,
        learn_sigma=False,
    )
    weight = model.x_embedder.proj.weight.data
    assert not torch.all(weight[:, 8:] == 0)


if __name__ == "__main__":
    test_zero_later_init_zeros_later_columns_and_keeps_bias_zero()
    test_stock_init_does_not_zero_later_columns()
    print("ok")
