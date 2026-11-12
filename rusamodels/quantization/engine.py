"""Isolated conversion and GGUF quantization adapters.

External llama.cpp tools are optional and must be installed explicitly. No shell is used.
"""
import os
import shutil
import subprocess
import sys
import time
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


def _run(command: list[str], work_dir: Path, timeout_seconds: int, cancel_event=None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Conversion was cancelled")
    process = subprocess.Popen(
        command,
        cwd=work_dir,
        env=_safe_environment(work_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        start_new_session=(os.name == "posix"),
    )
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise InterruptedError("Conversion was cancelled")
            return_code = process.poll()
            if return_code is not None:
                if return_code:
                    raise RuntimeError(
                        f"Conversion subprocess failed with exit code {return_code}"
                    )
                return
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                raise TimeoutError("Conversion subprocess exceeded its time limit")
            time.sleep(0.25)
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise


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
        cancel_event=None,
    ) -> list[Path]:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Conversion was cancelled")
        if target_precision not in TORCH_DTYPES:
            raise ValueError("target_precision must be F32, F16 or BF16")
        if quantization is not None and quantization not in QUANTIZATION_TYPES:
            raise ValueError("quantization must be Q8, Q6, Q5 or Q4")
        if quantization and target_format != "gguf":
            raise ValueError("Integer quantization is supported only for GGUF output")
        output_dir.mkdir(parents=True, exist_ok=True)
        gguf_sources = list(source_dir.glob("*.gguf"))
        if gguf_sources and target_format == "gguf" and quantization:
            return [
                self._quantize_existing_gguf(
                    gguf_sources[0], output_dir, quantization, cancel_event
                )
            ]
        if target_format == "safetensors":
            if quantization:
                raise ValueError("Integer quantization is not a safetensors precision")
            return self._convert_transformers(
                source_dir, output_dir, target_precision, cancel_event
            )
        if target_format == "gguf":
            return [
                self._convert_gguf(
                    source_dir, output_dir, target_precision, quantization, cancel_event
                )
            ]
        raise ValueError("target_format must be safetensors or gguf")

    def _convert_transformers(
        self, source_dir: Path, output_dir: Path, precision: str, cancel_event=None
    ) -> list[Path]:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Conversion was cancelled")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        inspect_model_directory(source_dir)
        dtype = getattr(torch, TORCH_DTYPES[precision])
        # source_dir is a locally downloaded snapshot pinned to its immutable Hub commit.
        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            source_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            use_safetensors=True,
            local_files_only=True,
        )
        model.save_pretrained(output_dir, safe_serialization=True)
        if cancel_event is not None and cancel_event.is_set():
            del model
            raise InterruptedError("Conversion was cancelled")
        try:
            # source_dir is local and local_files_only prevents any Hub fallback.
            tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                source_dir, trust_remote_code=False, local_files_only=True
            )
            tokenizer.save_pretrained(output_dir)
        except (OSError, ValueError):
            pass
        del model
        return sorted(path for path in output_dir.rglob("*") if path.is_file())

    def _convert_gguf(
        self,
        source_dir: Path,
        output_dir: Path,
        precision: str,
        quantization: str | None,
        cancel_event=None,
    ) -> Path:
        inspection = inspect_model_directory(source_dir)
        allowed_types = set(GGUF_MODEL_TYPES)
        extra = os.getenv("RUSAMODELS_GGUF_MODEL_TYPES", "")
        allowed_types.update(item.strip() for item in extra.split(",") if item.strip())
        if inspection.model_type not in allowed_types:
            raise RuntimeError(
                f"GGUF conversion is not enabled for model_type={inspection.model_type}"
            )
        script_value = os.getenv("LLAMA_CPP_CONVERT_SCRIPT")
        if not script_value or not Path(script_value).is_file():
            raise RuntimeError("LLAMA_CPP_CONVERT_SCRIPT is not configured or missing")
        script_path = str(Path(script_value).resolve())
        converter_output = output_dir / "model-f16.gguf"
        converter_precision = "f16" if quantization else precision.lower()
        _run(
            [
                sys.executable, script_path, str(source_dir), "--outfile",
                str(converter_output), "--outtype", converter_precision,
            ],
            output_dir,
            self.timeout_seconds,
            cancel_event,
        )
        if not quantization:
            return converter_output
        return self._quantize_existing_gguf(
            converter_output, output_dir, quantization, cancel_event
        )

    def _quantize_existing_gguf(
        self, source: Path, output_dir: Path, quantization: str, cancel_event=None
    ) -> Path:
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
            cancel_event,
        )
        return output
