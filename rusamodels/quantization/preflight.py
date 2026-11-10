"""Conservative container-aware CPU memory and temporary-disk preflight checks."""
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


class ResourceLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreflightResult:
    source_bytes: int
    estimated_peak_ram_bytes: int
    available_ram_bytes: int
    estimated_disk_bytes: int
    available_disk_bytes: int


def _host_available_ram_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return None


def _cgroup_available_ram_bytes(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    # cgroup v2 reports the hard memory limit and current usage separately.
    v2_limit = root / "memory.max"
    v2_current = root / "memory.current"
    if v2_limit.is_file():
        try:
            raw_limit = v2_limit.read_text(encoding="ascii").strip()
            if raw_limit != "max":
                limit = int(raw_limit)
                current = int(v2_current.read_text(encoding="ascii").strip()) if v2_current.is_file() else 0
                if 0 < limit < 2**60:
                    return max(0, limit - current)
        except (OSError, ValueError):
            pass

    # cgroup v1 uses memory.limit_in_bytes and memory.usage_in_bytes.
    v1_root = root / "memory"
    v1_limit = v1_root / "memory.limit_in_bytes"
    v1_current = v1_root / "memory.usage_in_bytes"
    if v1_limit.is_file():
        try:
            limit = int(v1_limit.read_text(encoding="ascii").strip())
            current = int(v1_current.read_text(encoding="ascii").strip()) if v1_current.is_file() else 0
            if 0 < limit < 2**60:
                return max(0, limit - current)
        except (OSError, ValueError):
            pass
    return None


def available_ram_bytes() -> int:
    host_available = _host_available_ram_bytes()
    cgroup_available = _cgroup_available_ram_bytes()
    if host_available is None and cgroup_available is None:
        raise ResourceLimitError("Cannot determine available system or container RAM")
    if host_available is None:
        return cgroup_available
    if cgroup_available is None:
        return host_available
    return min(host_available, cgroup_available)


def preflight_conversion(
    source_bytes: int,
    work_dir: Path,
    *,
    target_format: str,
    ram_bytes: int | None = None,
    disk_bytes: int | None = None,
) -> PreflightResult:
    if source_bytes <= 0:
        raise ValueError("Source model has no weight bytes")
    if target_format not in {"safetensors", "gguf"}:
        raise ValueError("Unsupported target format")
    ram_available = ram_bytes if ram_bytes is not None else available_ram_bytes()
    disk_available = disk_bytes if disk_bytes is not None else shutil.disk_usage(work_dir).free
    # CPU conversion holds model tensors plus temporary serialization buffers in memory.
    ram_factor = 2.5 if target_format == "safetensors" else 1.8
    estimated_ram = int(source_bytes * ram_factor) + 1024**3
    estimated_disk = int(source_bytes * 2.2)
    result = PreflightResult(
        source_bytes=source_bytes,
        estimated_peak_ram_bytes=estimated_ram,
        available_ram_bytes=ram_available,
        estimated_disk_bytes=estimated_disk,
        available_disk_bytes=disk_available,
    )
    if estimated_ram > ram_available:
        raise ResourceLimitError(
            f"Estimated peak RAM {estimated_ram} exceeds available RAM {ram_available}"
        )
    if estimated_disk > disk_available:
        raise ResourceLimitError(
            f"Estimated temporary disk {estimated_disk} exceeds available disk {disk_available}"
        )
    return result
