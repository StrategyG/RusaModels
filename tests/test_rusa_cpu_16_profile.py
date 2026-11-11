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
    assert profile["runtime"]["context_size"] == 8192
    assert profile["runtime"]["generation_threads"]["initial"] == 4
    assert profile["runtime"]["http_threads"] == 2
    assert profile["runtime"]["parallel_slots"] == 1
    assert profile["runtime"]["gpu_layers"] == 0
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


def test_codespaces_profile_is_documented_as_separate_from_native_windows():
    docs = (ROOT / "docs" / "CPU_INFERENCE.md").read_text(encoding="utf-8")
    assert "not the locked native Windows baseline" in docs
