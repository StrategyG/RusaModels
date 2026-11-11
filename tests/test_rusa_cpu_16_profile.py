from pathlib import Path
import json

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_rusa_cpu_16_profile_matches_rusaai_architecture_v2():
    profile = yaml.safe_load(
        (ROOT / "configs" / "runtime" / "rusa-cpu-16.yaml").read_text(encoding="utf-8")
    )
    model = json.loads(
        (ROOT / "models" / "rusaai-edge" / "manifests" / "0.1.0-design.json").read_text(
            encoding="utf-8"
        )
    )

    assert profile["profile"]["id"] == "rusa-cpu-16"
    assert profile["profile"]["version"] == "2.0.0"
    assert profile["profile"]["source_architecture"] == "RusaAi Architecture v2.0"
    assert profile["profile"]["status"] == "architecture_target_not_yet_deployable"
    assert profile["platform"]["operating_system"] == "Windows 11 64-bit"
    assert profile["platform"]["physical_memory_gb"] == 16
    assert profile["platform"]["cpu"]["sku_detection"] == "required"

    assert profile["model"]["id"] == "rusaai-edge-1.54b"
    assert profile["model"]["target_parameters"] == 1541277696
    assert profile["model"]["checkpoint"]["selection_requirement"] == "non_qwen_open_weight"
    assert profile["model"]["checkpoint"]["selection_status"] == "NOT_SELECTED"
    assert profile["model"]["tokenizer"]["status"] == "NOT_SELECTED"
    assert profile["model"]["artifact"]["status"] == "NOT_BUILT"
    assert profile["model"]["benchmark"]["status"] == "NOT_TESTED"

    assert model["model_id"] == "rusaai-edge-1.54b"
    assert model["status"] == "DESIGN_ONLY"
    assert model["checkpoint"]["selection_status"] == "NOT_SELECTED"
    assert model["artifact"]["status"] == "NOT_BUILT"
    assert model["architecture"]["target_parameters"] == 1541277696
    assert model["architecture"]["layers"] == 32
    assert model["architecture"]["hidden_size"] == 2048
    assert model["architecture"]["query_heads"] == 16
    assert model["architecture"]["key_value_heads"] == 4
    assert model["architecture"]["feed_forward"] == "swiglu"
    assert model["architecture"]["feed_forward_intermediate"] == 5632


def test_v2_runtime_parameters_and_control_plane_ownership():
    profile = yaml.safe_load(
        (ROOT / "configs" / "runtime" / "rusa-cpu-16.yaml").read_text(encoding="utf-8")
    )

    assert profile["runtime"]["context"]["initial_tokens"] == 4096
    assert profile["runtime"]["context"]["target_tokens"] == 8192
    assert profile["runtime"]["generation_threads_initial"] == 4
    assert profile["runtime"]["batch_threads_initial"] == 4
    assert profile["runtime"]["http_threads_initial"] == 2
    assert profile["runtime"]["batch_size_initial"] == 128
    assert profile["runtime"]["micro_batch_size_initial"] == 64
    assert profile["runtime"]["parallel_slots"] == 1
    assert profile["runtime"]["gpu_layers"] == 0
    assert profile["orchestration"]["owner"] == "Rusaai"
    assert profile["orchestration"]["active_generation_limit"] == 1
    assert profile["orchestration"]["pending_inference_queue_limit"] == 3
    assert profile["orchestration"]["queue_owner"] == "Rusaai"
    assert profile["resource_protection"]["minimum_free_memory_gb"] == 2.0
    assert profile["memory_estimates"]["kv_cache_q8_0_mib"]["context_4096"] == 128
    assert profile["memory_estimates"]["kv_cache_q8_0_mib"]["context_8192"] == 256


def test_native_windows_launcher_fails_closed_until_model_is_ready():
    start = (ROOT / "scripts" / "start_rusa_cpu_16.ps1").read_text(encoding="utf-8")
    verify = (ROOT / "scripts" / "verify_rusa_cpu_16.ps1").read_text(encoding="utf-8")
    env = (ROOT / ".env.rusa-cpu-16.example").read_text(encoding="utf-8")

    assert "RUSAMODELS_INFERENCE_MODEL_ID=rusaai-edge-1.54b" in env
    assert "RUSAMODELS_CONTEXT_SIZE=4096" in env
    assert 'selection_status -ne "SELECTED"' in start
    assert 'compatibility_status -ne "VERIFIED"' in start
    assert 'tokenizer.status -ne "VALIDATED"' in start
    assert 'artifact.status -ne "BUILT"' in start
    assert 'status -ne "READY_FOR_DEVICE_VALIDATION"' in start
    assert "tokenizer.revision" in start
    assert "artifact.filename" in start
    assert "Qwen/" in start
    assert '"--alias", $modelId' in start
    assert '"--ctx-size", "$contextSize"' in start
    assert '"--batch-size", "128"' in start
    assert '"--ubatch-size", "64"' in start
    assert '"--n-gpu-layers", "0"' in start
    assert '"--reasoning-budget"' not in start
    assert "rusaai-edge-1.54b" in verify
    assert "tokens_predicted_seconds_total" in verify
    assert "requests_processing" in verify
    assert "requests_deferred" in verify
    assert "minimumFreeMemoryGiB" in verify
    assert "IsProcessorFeaturePresent" in start
    assert "RUSA-CPU-16 services are ready" in start


def test_qwen3_v1_profile_is_preserved_as_historical_reference():
    legacy = yaml.safe_load(
        (
            ROOT / "configs" / "runtime" / "archive"
            / "rusa-cpu-16-v1.0-qwen3.yaml"
        ).read_text(encoding="utf-8")
    )
    legacy_docs = (
        ROOT / "docs" / "archive" / "RUSA-CPU-16-WINDOWS-v1.0-Qwen3.md"
    ).read_text(encoding="utf-8")

    assert legacy["model"]["repository"] == "Qwen/Qwen3-4B-GGUF"
    assert "Qwen3-4B" in legacy_docs
    assert "v1.0" in legacy_docs


def test_rusamodels_edge_test_configuration_uses_v2_model_identity():
    config_test = (ROOT / "tests" / "test_config.py").read_text(encoding="utf-8")
    assert '"inference_model_id": "rusaai-edge-1.54b"' in config_test
