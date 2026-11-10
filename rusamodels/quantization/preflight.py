"""Conservative CPU memory and temporary-disk preflight checks."""
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


def available_ram_bytes() -> int:
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError as exc:
        raise ResourceLimitError("Cannot determine available system RAM") from exc


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
