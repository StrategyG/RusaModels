from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_production_compose():
    return yaml.load(
        (ROOT / "docker-compose.production.yml").read_text(encoding="utf-8"),
        Loader=UniqueKeyLoader,
    )


def test_production_compose_has_no_local_database_or_object_store():
    compose = _load_production_compose()
    services = compose["services"]

    assert set(services) == {"migrate", "model-server", "quant-server", "quant-worker"}
    assert "postgres" not in services
    assert "minio" not in services
    assert all("env_file" not in service for service in services.values())


def test_model_server_receives_only_read_scoped_object_storage_credentials():
    compose = _load_production_compose()
    env = compose["services"]["model-server"]["environment"]

    assert "S3_READ_ACCESS_KEY_ID" in env
    assert "S3_READ_SECRET_ACCESS_KEY" in env
    assert "S3_WRITE_ACCESS_KEY_ID" not in env
    assert "S3_WRITE_SECRET_ACCESS_KEY" not in env
    assert "HF_TOKEN" not in env


def test_quant_services_receive_separate_read_and_write_credentials():
    compose = _load_production_compose()
    for service_name in ("quant-server", "quant-worker"):
        env = compose["services"][service_name]["environment"]
        assert "S3_READ_ACCESS_KEY_ID" in env
        assert "S3_WRITE_ACCESS_KEY_ID" in env
        assert "S3_READ_SECRET_ACCESS_KEY" in env
        assert "S3_WRITE_SECRET_ACCESS_KEY" in env


def test_production_containers_are_hardened_and_worker_has_no_published_port():
    compose = _load_production_compose()
    for name in ("model-server", "quant-server", "quant-worker"):
        service = compose["services"][name]
        assert service["read_only"] is True
        assert "ALL" in service["cap_drop"]
        assert "no-new-privileges:true" in service["security_opt"]

    assert "ports" not in compose["services"]["quant-worker"]
    assert compose["services"]["model-server"]["ports"] == ["127.0.0.1:8000:8000"]
    assert compose["services"]["quant-server"]["ports"] == ["127.0.0.1:8001:8000"]


def test_production_images_are_digest_pinned_by_operator_input():
    compose = (ROOT / "docker-compose.production.yml").read_text(encoding="utf-8")
    assert compose.count("${RUSAMODELS_IMAGE:?Set RUSAMODELS_IMAGE to an immutable GHCR digest}") == 3
    assert "${RUSAMODELS_WORKER_IMAGE:?Set RUSAMODELS_WORKER_IMAGE to a quantization-enabled immutable GHCR digest}" in compose
    assert "latest" not in compose
