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



def test_cgroup_v2_memory_limit_accounts_for_current_usage(tmp_path):
    from rusamodels.quantization.preflight import _cgroup_available_ram_bytes

    (tmp_path / "memory.max").write_text(str(8 * 1024**3))
    (tmp_path / "memory.current").write_text(str(3 * 1024**3))
    assert _cgroup_available_ram_bytes(tmp_path) == 5 * 1024**3


def test_cgroup_v1_memory_limit_accounts_for_current_usage(tmp_path):
    from rusamodels.quantization.preflight import _cgroup_available_ram_bytes

    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "memory.limit_in_bytes").write_text(str(16 * 1024**3))
    (memory / "memory.usage_in_bytes").write_text(str(4 * 1024**3))
    assert _cgroup_available_ram_bytes(tmp_path) == 12 * 1024**3


def test_available_ram_uses_lower_container_or_host_value(monkeypatch):
    from rusamodels.quantization import preflight

    monkeypatch.setattr(preflight, "_host_available_ram_bytes", lambda: 12 * 1024**3)
    monkeypatch.setattr(preflight, "_cgroup_available_ram_bytes", lambda: 4 * 1024**3)
    assert preflight.available_ram_bytes() == 4 * 1024**3



def test_conversion_subprocess_honors_preexisting_cancel_request(tmp_path):
    from threading import Event

    from rusamodels.quantization.engine import _run

    cancelled = Event()
    cancelled.set()
    with pytest.raises(InterruptedError, match="cancelled"):
        _run(["python", "-c", "print('should not run')"], tmp_path, 10, cancelled)


@pytest.mark.parametrize("filename", ["pytorch_model.bin", "checkpoint.pt"])
def test_inspect_rejects_pickle_based_weights(tmp_path, filename):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
    }))
    (tmp_path / filename).write_bytes(b"pickle-based-test-weight")

    with pytest.raises(ValueError, match="Safetensors"):
        inspect_model_directory(tmp_path)


def test_inspect_prefers_safetensors_when_legacy_weights_are_also_present(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
    }))
    (tmp_path / "model.safetensors").write_bytes(b"safe-weight")
    (tmp_path / "pytorch_model.bin").write_bytes(b"ignored-pickle-weight")

    result = inspect_model_directory(tmp_path)

    assert [path.name for path in result.weight_files] == ["model.safetensors"]
    assert result.total_weight_bytes == len(b"safe-weight")
