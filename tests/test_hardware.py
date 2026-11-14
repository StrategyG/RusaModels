from rusamodels.hardware import GIB, GPUDevice, HardwareInventory, detect_nvidia_gpus, recommend_runtime_profile


def inventory(total, available, *, cores=8, gpus=()):
    return HardwareInventory(
        operating_system="windows", operating_system_version="test", architecture="amd64",
        cpu_model="test cpu", logical_cores=cores, total_memory_bytes=total,
        available_memory_bytes=available, free_disk_bytes=100 * GIB, gpus=gpus,
    )


def test_8gb_profile_is_bounded_and_cpu_first():
    result = recommend_runtime_profile(inventory(8 * GIB, 6 * GIB, cores=12))
    assert result["profile"] == "local-8gb"
    assert result["inference_device"] == "cpu"
    assert result["gpu_offload_layers"] == 0
    assert result["max_concurrent_generations"] == 1
    assert result["max_waiting_requests"] == 1
    assert result["cpu_threads"] == 8
    assert result["status"] == "RECOMMENDATION_ONLY"


def test_16gb_profile_has_bounded_queue():
    result = recommend_runtime_profile(inventory(16 * GIB, 12 * GIB))
    assert result["profile"] == "local-16gb"
    assert result["max_concurrent_generations"] == 1
    assert result["max_waiting_requests"] == 3


def test_unknown_memory_fails_conservatively():
    result = recommend_runtime_profile(inventory(None, None))
    assert result["profile"] == "unknown"
    assert result["context_tokens"] == 2048
    assert result["max_waiting_requests"] == 1
    assert result["estimated_safe_model_memory_bytes"] is None


def test_gpu_detection_is_only_a_candidate():
    gpu = GPUDevice("NVIDIA", "Test GPU", 8 * GIB, "test-driver")
    result = recommend_runtime_profile(inventory(32 * GIB, 24 * GIB, gpus=(gpu,)))
    assert result["gpu_acceleration_candidate"] is True
    assert result["inference_device"] == "cpu"
    assert result["gpu_offload_layers"] == 0


def test_nvidia_smi_output_parser(monkeypatch):
    import rusamodels.hardware as hardware
    monkeypatch.setattr(
        hardware, "_run",
        lambda command, timeout=3.0: "NVIDIA RTX Test, 8192, 555.1\nNVIDIA A10, 23028, 550.2",
    )
    devices = detect_nvidia_gpus()
    assert len(devices) == 2
    assert devices[0].name == "NVIDIA RTX Test"
    assert devices[0].memory_total_bytes == 8192 * 1024 * 1024
