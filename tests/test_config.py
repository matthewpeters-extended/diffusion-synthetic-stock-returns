"""Guards the split from SCOPE.md so a later edit can't silently create leakage."""

from __future__ import annotations

from src import config


def test_split_is_ordered_and_disjoint():
    assert config.DATA_START == config.TRAIN_START < config.TRAIN_END
    assert config.TRAIN_END < config.TEST_START < config.TEST_END == config.DATA_END


def test_window_constants():
    assert config.EVAL_STRIDE == config.WINDOW
    assert config.ACF_LAGS < config.WINDOW


def test_torch_is_cpu_importable():
    import torch

    assert torch.zeros(2).sum().item() == 0.0
