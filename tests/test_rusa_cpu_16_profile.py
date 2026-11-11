from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_rusa_cpu_16_profile_matches_locked_windows_baseline():
    profile = yaml.safe_load(
        (ROOT / "configs" / "runtime" / "rusa-cpu-16.yaml").read_text(encoding="utf-8")
    )

    assert profile["profile"]["id"] == "rusa-cpu-16"
    assert profile["profile"]["version"] == "1.0.0"
    assert profile["platform"]["operating_system"] == "Windows 11 64-bit"
    assert profile["platform"]["cpu"]["family"] == "Core i5"
    assert profile["platform"]["memory_gib"] == 16
    assert profile["model"]["repository"] == "Qwen/Qwen3-4B-GGUF"
    assert profile["model"]["quantization"] == "Q4_K_M"
    assert profile["model"]["artifact_filename"] == "Qwen3-4B-Q4_K_M.gguf"
    assert profile["model"]["repository_revision"] == "a9a60d009fa7ff9606305047c2bf77ac25dbec49"
    assert profile["model"]["artifact_sha256"] == "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
    assert profile["model"]["artifact_size_bytes"] == 2497280256
    assert profile["runtime"]["context_size"] == 8192
    assert profile["runtime"]["generation_threads"]["initial"] == 4
    assert profile["runtime"]["http_threads"] == 2
    assert profile["runtime"]["parallel_slots"] == 1
    assert profile["runtime"]["gpu_layers"] == 0
    assert profile["context_budget_tokens"]["total"] == 8192
    assert profile["context_budget_tokens"]["retrieved_project_context"] == 2500
    assert profile["memory_retrieval"]["chunking"]["initial_retrieval_chunks"] == {"minimum": 3, "maximum": 6}
    assert profile["orchestration"]["resource_limits"]["light_file_indexing_workers"] == 2
    assert profile["orchestration"]["active_generation_limit"] == 1
    assert profile["orchestration"]["pending_inference_queue_limit"] == 3
    assert profile["orchestration"]["owner"] == "Rusaai"
    assert profile["memory_budget_gib"]["reserved_free_memory"] == 2.0
    assert profile["services"]["external_inference_fallback"] is False
    assert profile["services"]["separate_embedding_runtime"] == "disabled"


def test_native_windows_scripts_bind_only_loopback_and_use_expected_ports():
    start = (ROOT / "scripts" / "start_rusa_cpu_16.ps1").read_text(encoding="utf-8")
    verify = (ROOT / "scripts" / "verify_rusa_cpu_16.ps1").read_text(encoding="utf-8")

    for expected in (
        '"--host", "127.0.0.1"',
        '"--port", "8081"',
        '"--ctx-size", "8192"',
        '"--parallel", "1"',
        '"--n-gpu-layers", "0"',
        '"--threads-http", "2"',
        '"--cache-type-k", "q8_0"',
        '"--cache-type-v", "q8_0"',
        '"--no-webui"',
    ):
        assert expected in start
    assert "127.0.0.1:8090" in start
    assert "127.0.0.1:8081" in verify
    assert "SequentialRequests" in verify
    assert "tokens_predicted_seconds_total" in verify
    assert "firstTokenSeconds" in verify
    assert "minimumFreeMemoryGiB" in verify
    assert "IsProcessorFeaturePresent" in start
    assert "--help" in start
    assert "--version" in start
    assert "kvCacheQ8Supported" in start
    assert "Raw Rusaai tokens must not be stored" in start
    assert 'SetEnvironmentVariable("RUSAMODELS_API_TOKEN", $null, "Process")' in start
    assert 'SetEnvironmentVariable("RUSAMODELS_SERVICE_TOKEN", $null, "Process")' in start
    assert '$parsedApiBaseUrl.Host -ne "127.0.0.1"' in verify
    assert "$parsedApiBaseUrl.Port -ne 8090" in verify


def test_codespaces_profile_is_documented_as_separate_from_native_windows():
    docs = (ROOT / "docs" / "CPU_INFERENCE.md").read_text(encoding="utf-8")
    assert "not the locked native Windows baseline" in docs


def test_windows_edge_example_uses_official_qwen3_q4_k_m_filename():
    env = (ROOT / ".env.rusa-cpu-16.example").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "RUSA-CPU-16-WINDOWS.md").read_text(encoding="utf-8")

    assert "Qwen3-4B-Q4_K_M.gguf" in env
    assert "Qwen3-4B-Q4_K_M.gguf" in docs
    assert "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5" in env
    assert "a9a60d009fa7ff9606305047c2bf77ac25dbec49" in docs
    assert ".local\\artifacts\\models" in env
    assert ".local\\artifacts\\models" in docs



def test_raw_rusaai_tokens_are_kept_out_of_rusamodels_env_and_child_processes():
    docs = (ROOT / "docs" / "RUSA-CPU-16-WINDOWS.md").read_text(encoding="utf-8")
    env = (ROOT / ".env.rusa-cpu-16.example").read_text(encoding="utf-8")

    assert "$env:RUSAMODELS_API_TOKEN = $inferenceToken" not in docs.split("## 5. Start and stop")[0]
    assert "$env:RUSAMODELS_SERVICE_TOKEN = $serviceToken" not in docs
    assert "RUSAMODELS_API_TOKEN=" not in env
    assert "RUSAMODELS_SERVICE_TOKEN=" not in env
