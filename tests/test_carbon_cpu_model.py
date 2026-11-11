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


def test_reference_carbon_tiny_parameter_count() -> None:
    model = CarbonTiny(CarbonConfig())
    assert count_parameters(model) == 3_214_080


def test_config_rejects_invalid_gqa_ratio() -> None:
    with pytest.raises(ValueError, match="n_heads must be divisible"):
        CarbonConfig(d_model=96, n_heads=6, n_kv_heads=4)


def test_config_rejects_zero_kv_heads_without_zero_division() -> None:
    with pytest.raises(ValueError, match="n_heads and n_kv_heads must be positive"):
        CarbonConfig(n_kv_heads=0)


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown CarbonConfig fields"):
        CarbonConfig.from_dict({"n_layers": 4, "num_layers": 8})


def test_forward_rejects_target_shape_mismatch() -> None:
    model = CarbonTiny(CarbonConfig(n_layers=1, d_model=32, n_heads=4, n_kv_heads=2, intermediate_size=64))
    inputs = torch.ones((1, 8), dtype=torch.long)
    targets = torch.ones((1, 7), dtype=torch.long)
    with pytest.raises(ValueError, match="same shape"):
        model(inputs, targets)


def test_generate_restores_training_mode_and_validates_arguments() -> None:
    model = CarbonTiny(CarbonConfig(n_layers=1, d_model=32, n_heads=4, n_kv_heads=2, intermediate_size=64))
    model.train()
    inputs = torch.tensor([[65, 66, 67]], dtype=torch.long)
    output = model.generate(inputs, max_new_tokens=1, top_k=0)
    assert output.shape == (1, 4)
    assert model.training
    with pytest.raises(ValueError, match="top_k"):
        model.generate(inputs, max_new_tokens=1, top_k=-1)
