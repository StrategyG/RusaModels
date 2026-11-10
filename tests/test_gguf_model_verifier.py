import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from rusamodels.gguf import verify_gguf_file


def test_verifies_pinned_gguf_file(tmp_path):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"gguf-fixture")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()

    verified = verify_gguf_file(model, expected)

    assert verified.path == model
    assert verified.size_bytes == len(b"gguf-fixture")
    assert verified.sha256 == expected


def test_rejects_wrong_gguf_checksum(tmp_path):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"gguf-fixture")

    with pytest.raises(ValueError, match="does not match"):
        verify_gguf_file(model, "0" * 64)


def test_rejects_non_absolute_model_path(tmp_path, monkeypatch):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"gguf-fixture")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="absolute"):
        verify_gguf_file("tiny.gguf", hashlib.sha256(b"gguf-fixture").hexdigest())


def test_rejects_non_gguf_file(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"gguf-fixture")

    with pytest.raises(ValueError, match=".gguf"):
        verify_gguf_file(model, hashlib.sha256(b"gguf-fixture").hexdigest())


def test_rejects_symlinked_model(tmp_path):
    model = tmp_path / "tiny.gguf"
    target = tmp_path / "actual.gguf"
    target.write_bytes(b"gguf-fixture")
    try:
        model.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are unavailable on this platform")

    with pytest.raises(OSError):
        verify_gguf_file(model, hashlib.sha256(b"gguf-fixture").hexdigest())


def test_rejects_invalid_expected_checksum(tmp_path):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"gguf-fixture")

    with pytest.raises(ValueError, match="64 hexadecimal"):
        verify_gguf_file(model, "not-a-checksum")


def test_verifier_cli_loads_model_settings_from_dotenv(tmp_path, monkeypatch):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"gguf-fixture")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"RUSAMODELS_GGUF_MODEL_PATH={model}\nRUSAMODELS_GGUF_MODEL_SHA256={digest}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("RUSA_GGUF_MODEL_PATH", raising=False)
    monkeypatch.delenv("RUSA_GGUF_MODEL_SHA256", raising=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_gguf_model.py"

    result = subprocess.run(
        [sys.executable, str(script), "--env-file", str(env_file)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"sha256={digest}" in result.stdout