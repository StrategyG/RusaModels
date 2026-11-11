from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from rusamodels.carbon_cpu.model import CarbonConfig, CarbonTiny, count_parameters  # noqa: E402


def test_carbon_tiny_forward_and_loss() -> None:
    config = CarbonConfig(
        vocab_size=258,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=128,
        max_seq_len=32,
    )
    model = CarbonTiny(config)
    inputs = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    assert count_parameters(model) > 0


def test_config_rejects_invalid_gqa_ratio() -> None:
    with pytest.raises(ValueError, match="n_heads must be divisible"):
        CarbonConfig(d_model=96, n_heads=6, n_kv_heads=4)
