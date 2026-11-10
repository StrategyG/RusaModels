import json

import pytest

from rusamodels.quantization.inspect import inspect_model_directory
from rusamodels.quantization.preflight import ResourceLimitError, preflight_conversion


def test_inspect_supported_huggingface_model_directory(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "num_parameters": 125000000,
    }))
    (tmp_path / "model.safetensors").write_bytes(b"small-test-weight")
    result = inspect_model_directory(tmp_path)
    assert result.model_type == "qwen2"
    assert result.architectures == ("Qwen2ForCausalLM",)
    assert result.parameter_count == 125000000
    assert result.total_weight_bytes == len(b"small-test-weight")


def test_inspect_rejects_missing_config(tmp_path):
    with pytest.raises(ValueError, match="config.json"):
        inspect_model_directory(tmp_path)


def test_preflight_rejects_ram_exhaustion(tmp_path):
    with pytest.raises(ResourceLimitError, match="RAM"):
        preflight_conversion(
            2 * 1024**3, tmp_path, target_format="safetensors",
            ram_bytes=2 * 1024**3, disk_bytes=20 * 1024**3,
        )


def test_preflight_accepts_small_compatible_checkpoint(tmp_path):
    result = preflight_conversion(
        100, tmp_path, target_format="gguf",
        ram_bytes=4 * 1024**3, disk_bytes=4 * 1024**3,
    )
    assert result.source_bytes == 100
    assert result.estimated_peak_ram_bytes < result.available_ram_bytes
