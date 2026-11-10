"""Isolated conversion and GGUF quantization adapters.

External llama.cpp tools are optional and must be installed explicitly. No shell is used.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rusamodels.quantization.inspect import inspect_model_directory

GGUF_MODEL_TYPES = {
    "llama", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
    "deepseek_v2", "deepseek_v3", "deepseek_v2_moe", "deepseek_v3_moe",
}
QUANTIZATION_TYPES = {
    "Q8": "Q8_0",
    "Q6": "Q6_K",
    "Q5": "Q5_K_M",
    "Q4": "Q4_K_M",
}
TORCH_DTYPES = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}


def _safe_environment(work_dir: Path) -> dict[str, str]:
    allowed = {"PATH", "HOME", "LD_LIBRARY_PATH", "SYSTEMROOT"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["HOME"] = str(work_dir)
    env["TMPDIR"] = str(work_dir)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return env


def _run(command: list[str], work_dir: Path, timeout_seconds: int) -> None:
    result = subprocess.run(
        command,
        cwd=work_dir,
        env=_safe_environment(work_dir),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        shell=False,
    )
    if result.returncode:
        raise RuntimeError(f"Conversion subprocess failed with exit code {result.returncode}")


class ConversionEngine:
    def __init__(self, timeout_seconds: int = 6 * 60 * 60):
        self.timeout_seconds = timeout_seconds

    def convert(
        self,
        source_dir: Path,
        output_dir: Path,
        *,
        target_format: str,
        target_precision: str,
        quantization: str | None = None,
    ) -> list[Path]:
        if target_precision not in TORCH_DTYPES:
            raise ValueError("target_precision must be F32, F16 or BF16")
        if quantization is not None and quantization not in QUANTIZATION_TYPES:
            raise ValueError("quantization must be Q8, Q6, Q5 or Q4")
        if quantization and target_format != "gguf":
            raise ValueError("Integer quantization is supported only for GGUF output")
        output_dir.mkdir(parents=True, exist_ok=True)
        gguf_sources = list(source_dir.glob("*.gguf"))
        if gguf_sources and target_format == "gguf" and quantization:
            return [self._quantize_existing_gguf(gguf_sources[0], output_dir, quantization)]
        if target_format == "safetensors":
            if quantization:
                raise ValueError("Integer quantization is not a safetensors precision")
            return self._convert_transformers(source_dir, output_dir, target_precision)
        if target_format == "gguf":
            return [self._convert_gguf(source_dir, output_dir, target_precision, quantization)]
        raise ValueError("target_format must be safetensors or gguf")

    def _convert_transformers(self, source_dir: Path, output_dir: Path, precision: str) -> list[Path]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        inspect_model_directory(source_dir)
        dtype = getattr(torch, TORCH_DTYPES[precision])
        model = AutoModelForCausalLM.from_pretrained(
            source_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            local_files_only=True,
        )
        model.save_pretrained(output_dir, safe_serialization=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source_dir, trust_remote_code=False, local_files_only=True
            )
            tokenizer.save_pretrained(output_dir)
        except (OSError, ValueError):
            pass
        del model
        return sorted(path for path in output_dir.rglob("*") if path.is_file())

    def _convert_gguf(
        self, source_dir: Path, output_dir: Path, precision: str, quantization: str | None
    ) -> Path:
        inspection = inspect_model_directory(source_dir)
        allowed_types = set(GGUF_MODEL_TYPES)
        extra = os.getenv("RUSAMODELS_GGUF_MODEL_TYPES", "")
        allowed_types.update(item.strip() for item in extra.split(",") if item.strip())
        if inspection.model_type not in allowed_types:
            raise RuntimeError(f"GGUF conversion is not enabled for model_type={inspection.model_type}")
        script_value = os.getenv("LLAMA_CPP_CONVERT_SCRIPT")
        if not script_value or not Path(script_value).is_file():
            raise RuntimeError("LLAMA_CPP_CONVERT_SCRIPT is not configured or missing")
        script_path = str(Path(script_value).resolve())
        converter_output = output_dir / "model-f16.gguf"
        converter_precision = "f16" if quantization else precision.lower()
        _run(
            [sys.executable, script_path, str(source_dir), "--outfile", str(converter_output),
             "--outtype", converter_precision],
            output_dir,
            self.timeout_seconds,
        )
        if not quantization:
            return converter_output
        return self._quantize_existing_gguf(converter_output, output_dir, quantization)

    def _quantize_existing_gguf(self, source: Path, output_dir: Path, quantization: str) -> Path:
        binary_value = os.getenv("LLAMA_QUANTIZE_BIN", "llama-quantize")
        binary = shutil.which(binary_value) or (
            str(Path(binary_value).resolve()) if Path(binary_value).is_file() else None
        )
        if not binary:
            raise RuntimeError("llama-quantize executable is not installed")
        output = output_dir / f"model-{quantization.lower()}.gguf"
        _run(
            [str(binary), str(source), str(output), QUANTIZATION_TYPES[quantization]],
            output_dir,
            self.timeout_seconds,
        )
        return output
