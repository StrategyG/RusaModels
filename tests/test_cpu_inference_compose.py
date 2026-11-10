from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_codespaces_cpu_profile_runs_private_llm_and_embedding_services():
    compose = (ROOT / ".devcontainer" / "docker-compose.cpu-inference.yml").read_text(encoding="utf-8")
    assert "RUSAMODELS_INFERENCE_BASE_URL: http://llama-cpp:8080" in compose
    assert "RUSAMODELS_EMBEDDING_BASE_URL: http://embedding-server:8081" in compose
    assert "${RUSAMODELS_GGUF_MODEL_PATH:?Set absolute RUSAMODELS_GGUF_MODEL_PATH}" in compose
    assert "${RUSAMODELS_EMBEDDING_MODEL_PATH:?Set absolute RUSAMODELS_EMBEDDING_MODEL_PATH}" in compose
    assert "--n-gpu-layers" in compose
    assert '      - "0"' in compose
    assert "ports:" not in compose.split("  llama-cpp:", 1)[1].split("  embedding-server:", 1)[0]
    devcontainer = (ROOT / ".devcontainer" / "model-server-cpu" / "devcontainer.json").read_text(encoding="utf-8")
    assert '"memory": 16' in devcontainer
    assert '"cpus": 4' in devcontainer
