"""Cross-platform hardware inventory and conservative local runtime hints."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GIB = 1024 ** 3


@dataclass(frozen=True)
class GPUDevice:
    vendor: str
    name: str
    memory_total_bytes: int | None = None
    driver_version: str | None = None
    detection_source: str = "nvidia-smi"


@dataclass(frozen=True)
class HardwareInventory:
    operating_system: str
    operating_system_version: str
    architecture: str
    cpu_model: str
    logical_cores: int
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    free_disk_bytes: int | None
    gpus: tuple[GPUDevice, ...]


def _run(command: list[str], timeout: float = 3.0) -> str | None:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _linux_memory() -> tuple[int | None, int | None]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return (
            int(os.sysconf("SC_PHYS_PAGES")) * page_size,
            int(os.sysconf("SC_AVPHYS_PAGES")) * page_size,
        )
    except (AttributeError, OSError, ValueError):
        return None, None


def _windows_memory() -> tuple[int | None, int | None]:
    if sys.platform != "win32":
        return None, None
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _mac_memory() -> tuple[int | None, int | None]:
    raw = _run(["sysctl", "-n", "hw.memsize"])
    if not raw or not raw.isdigit():
        return None, None
    try:
        available = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        available = None
    return int(raw), available


def detect_memory() -> tuple[int | None, int | None]:
    if sys.platform == "win32":
        return _windows_memory()
    if sys.platform == "darwin":
        return _mac_memory()
    if sys.platform.startswith("linux"):
        return _linux_memory()
    return None, None


def detect_nvidia_gpus() -> tuple[GPUDevice, ...]:
    output = _run([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return ()
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0]:
            continue
        try:
            memory_bytes = int(parts[1]) * 1024 * 1024
        except ValueError:
            memory_bytes = None
        devices.append(GPUDevice("NVIDIA", parts[0], memory_bytes, parts[2] or None))
    return tuple(devices)


def detect_hardware(storage_path: str | Path | None = None) -> HardwareInventory:
    total_memory, available_memory = detect_memory()
    try:
        free_disk = int(shutil.disk_usage(Path(storage_path or Path.home()).expanduser()).free)
    except OSError:
        free_disk = None
    cpu_model = platform.processor() or platform.uname().processor or platform.machine()
    return HardwareInventory(
        operating_system=platform.system().lower(),
        operating_system_version=platform.version(),
        architecture=platform.machine().lower(),
        cpu_model=cpu_model or "unknown",
        logical_cores=max(1, os.cpu_count() or 1),
        total_memory_bytes=total_memory,
        available_memory_bytes=available_memory,
        free_disk_bytes=free_disk,
        gpus=detect_nvidia_gpus(),
    )


def recommend_runtime_profile(inventory: HardwareInventory) -> dict[str, Any]:
    """Return a conservative starting point, never an automatic fit guarantee."""
    total, available = inventory.total_memory_bytes, inventory.available_memory_bytes
    if total is None:
        tier, safe_budget, waiting, context = "unknown", None, 1, 2048
    else:
        tier = (
            "local-32gb" if total >= 32 * GIB else
            "local-16gb" if total >= 16 * GIB else
            "local-8gb" if total >= 8 * GIB else "minimal"
        )
        observed = available if available is not None else total
        safe_budget = max(0, observed - max(2 * GIB, int(total * 0.20)))
        waiting = 3 if total >= 16 * GIB else 1
        context = 4096 if safe_budget >= 3 * GIB else 2048
    return {
        "status": "RECOMMENDATION_ONLY",
        "profile": tier,
        "inference_device": "cpu",
        "gpu_acceleration_candidate": bool(inventory.gpus),
        "gpu_offload_layers": 0,
        "cpu_threads": max(1, min(inventory.logical_cores, 8)),
        "context_tokens": context,
        "max_concurrent_generations": 1,
        "max_waiting_requests": waiting,
        "estimated_safe_model_memory_bytes": safe_budget,
        "requires_model_specific_benchmark": True,
        "notes": [
            "GPU presence is detected, not proof that the runtime/build supports it.",
            "Memory budget is an estimate, not a reservation.",
            "Verify model license, immutable revision, tokenizer, manifest and artifact SHA-256.",
            "Benchmark load, latency, thermal behavior and soak stability on the target device.",
        ],
    }


def inventory_payload(inventory: HardwareInventory) -> dict[str, Any]:
    payload = asdict(inventory)
    payload["gpus"] = [asdict(gpu) for gpu in inventory.gpus]
    payload["recommended_profile"] = recommend_runtime_profile(inventory)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-path", help="Path whose free space should be measured")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    payload = inventory_payload(detect_hardware(args.storage_path))
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        profile = payload["recommended_profile"]
        print(f"OS: {payload['operating_system']} {payload['operating_system_version']}")
        print(f"Architecture: {payload['architecture']}")
        print(f"CPU: {payload['cpu_model']} ({payload['logical_cores']} logical cores)")
        print(f"RAM: {payload['total_memory_bytes'] or 'unknown'} bytes")
        print(f"Free disk: {payload['free_disk_bytes'] or 'unknown'} bytes")
        print(f"Detected GPUs: {len(payload['gpus'])}")
        print(f"Suggested profile: {profile['profile']} (recommendation only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
