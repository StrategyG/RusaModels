from pathlib import Path

import json

import pytest

from rusamodels.architecture.config import (
    ArchitectureConfigError,
    DecoderArchitectureConfig,
    load_architecture_blueprint,
)
from rusamodels.architecture.model_factory import build_model
from rusamodels.architecture.validation import validate_architecture_config


ROOT = Path(__file__).resolve().parents[1]


def test_research_100m_blueprint_matches_training_engine_target() -> None:
    blueprint = load_architecture_blueprint(
        ROOT / "configs/architecture/rusaai-research-100m.json"
    )
    summary = validate_architecture_config(blueprint.architecture)

    assert blueprint.model_id == "rusaai-research-100m"
    assert blueprint.status == "design_ready_training_not_started"
    assert summary["head_dim"] == 64
    assert summary["kv_group_size"] == 3
    assert summary["parameter_count"] == 100_092_672


def test_architecture_rejects_invalid_gqa_and_head_dimension() -> None:
    with pytest.raises(ArchitectureConfigError, match="query_heads must be divisible"):
        DecoderArchitectureConfig(
            vocab_size=32000,
            layers=12,
            hidden_size=768,
            query_heads=12,
            kv_heads=5,
            intermediate_size=2048,
            context_length=4096,
        )
    with pytest.raises(ArchitectureConfigError, match="head_dim must be even"):
        DecoderArchitectureConfig(
            vocab_size=32000,
            layers=2,
            hidden_size=25,
            query_heads=5,
            kv_heads=1,
            intermediate_size=64,
            context_length=128,
        )


def test_architecture_rejects_unknown_fields_and_wrong_parameter_count() -> None:
    base = {
        "vocab_size": 258,
        "layers": 1,
        "hidden_size": 32,
        "query_heads": 4,
        "kv_heads": 2,
        "intermediate_size": 64,
        "context_length": 32,
        "eos_token_id": 256,
    }
    with pytest.raises(ArchitectureConfigError, match="unsupported architecture fields"):
        DecoderArchitectureConfig.from_dict({**base, "num_experts": 8})
    with pytest.raises(ArchitectureConfigError, match="does not match"):
        DecoderArchitectureConfig.from_dict({**base, "expected_parameters": 1})


def test_architecture_rejects_unsupported_model_features() -> None:
    values = {
        "vocab_size": 258,
        "layers": 1,
        "hidden_size": 32,
        "query_heads": 4,
        "kv_heads": 2,
        "intermediate_size": 64,
        "context_length": 32,
        "eos_token_id": 256,
        "attention_bias": True,
    }
    with pytest.raises(ArchitectureConfigError, match="attention_bias=false"):
        DecoderArchitectureConfig.from_dict(values)


def test_blueprint_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        '{"schema_version":"1.0.0","model_id":"a","model_id":"b"}',
        encoding="utf-8",
    )
    with pytest.raises(ArchitectureConfigError, match="duplicate JSON object key"):
        load_architecture_blueprint(path)


def test_blueprint_rejects_symlink(tmp_path: Path) -> None:
    source = ROOT / "configs/architecture/rusaai-research-100m.json"
    link = tmp_path / "blueprint-link.json"
    try:
        link.symlink_to(source)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not supported in this environment")
    with pytest.raises(ArchitectureConfigError, match="must not be a symlink"):
        load_architecture_blueprint(link)


def test_model_factory_enforces_parameter_limit_before_importing_model() -> None:
    config = DecoderArchitectureConfig(
        vocab_size=32000,
        layers=12,
        hidden_size=768,
        query_heads=12,
        kv_heads=4,
        intermediate_size=2048,
        context_length=4096,
        eos_token_id=2,
    )
    with pytest.raises(ValueError, match="exceeding the construction limit"):
        build_model(config, max_parameters=100_000_000)


def test_model_factory_requires_eos_token_before_construction() -> None:
    config = DecoderArchitectureConfig(
        vocab_size=258,
        layers=1,
        hidden_size=32,
        query_heads=4,
        kv_heads=2,
        intermediate_size=64,
        context_length=32,
    )
    with pytest.raises(ValueError, match="eos_token_id is required"):
        config.to_carbon_config_dict()


def test_carbon_model_factory_constructs_expected_shapes() -> None:
    torch = pytest.importorskip("torch")
    config = DecoderArchitectureConfig(
        vocab_size=300,
        layers=2,
        hidden_size=64,
        query_heads=4,
        kv_heads=2,
        intermediate_size=128,
        context_length=64,
        eos_token_id=2,
        rms_norm_eps=1e-5,
    )
    model = build_model(config, max_parameters=100_000)
    inputs = torch.randint(0, config.vocab_size, (2, 16), dtype=torch.long)
    logits, loss = model(inputs, inputs)
    assert tuple(logits.shape) == (2, 16, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    assert model.config.eos_token_id == 2
    assert model.final_norm.eps == pytest.approx(1e-5)


def test_blueprint_requires_explicit_parameter_count(tmp_path: Path) -> None:
    source = ROOT / "configs/architecture/rusaai-research-100m.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    del payload["architecture"]["expected_parameters"]
    path = tmp_path / "missing-count.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArchitectureConfigError, match="missing required versioned fields"):
        load_architecture_blueprint(path)


def test_architecture_rejects_eos_outside_vocabulary() -> None:
    with pytest.raises(ArchitectureConfigError, match="eos_token_id must be inside"):
        DecoderArchitectureConfig(
            vocab_size=258,
            layers=1,
            hidden_size=32,
            query_heads=4,
            kv_heads=2,
            intermediate_size=64,
            context_length=32,
            eos_token_id=258,
        )
