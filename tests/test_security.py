import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rusamodels.config import Settings

from rusamodels.security import (
    ROLE_PERMISSIONS,
    parse_principals,
    sha256_hex,
    validate_production_service_credentials,
)


def test_role_boundaries():
    assert "models:read" in ROLE_PERMISSIONS["viewer"]
    assert "models:write" not in ROLE_PERMISSIONS["viewer"]
    assert "models:write" in ROLE_PERMISSIONS["manager"]
    assert "admin:destructive" not in ROLE_PERMISSIONS["manager"]
    assert "admin:destructive" in ROLE_PERMISSIONS["administrator"]


def test_hashed_principal_registry():
    settings = Settings(environment="test", api_principals_json=json.dumps({
        sha256_hex("secret"): {"subject": "ci", "role": "viewer"}}))
    assert parse_principals(settings.api_principals_json)[sha256_hex("secret")].subject == "ci"


def test_plaintext_token_registry_rejected():
    with pytest.raises(RuntimeError, match="SHA-256"):
        parse_principals(json.dumps({"plaintext": {"subject": "ci", "role": "viewer"}}))


def test_production_requires_postgres_and_s3():
    with pytest.raises(ValidationError):
        Settings(environment="production", database_url="sqlite:///prod.db", storage_backend="local")


def test_inference_role_is_limited_to_inference_capability():
    assert ROLE_PERMISSIONS["inference"] == frozenset({"inference:invoke"})
    principals = parse_principals(json.dumps({
        sha256_hex("inference-token"): {
            "subject": "rusa-ai-inference",
            "role": "inference",
        }
    }))
    assert principals[sha256_hex("inference-token")].role == "inference"


def test_inference_role_is_not_a_model_manager_or_administrator():
    permissions = ROLE_PERMISSIONS["inference"]
    assert "models:write" not in permissions
    assert "quantization:run" not in permissions
    assert "admin:destructive" not in permissions
    assert "audit:read" not in permissions



def test_production_service_credentials_require_valid_principal_registry():
    settings = SimpleNamespace(
        service_token_sha256="a" * 64,
        api_principals_json=json.dumps({
            sha256_hex("rusa-inference-secret"): {
                "subject": "rusa-ai-inference",
                "role": "inference",
            }
        }),
    )
    validate_production_service_credentials(settings, require_inference=True)


def test_model_server_fails_closed_without_dedicated_inference_principal():
    settings = SimpleNamespace(
        service_token_sha256="a" * 64,
        api_principals_json=json.dumps({
            sha256_hex("manager-secret"): {
                "subject": "operator",
                "role": "manager",
            }
        }),
    )
    with pytest.raises(RuntimeError, match="role=inference"):
        validate_production_service_credentials(settings, require_inference=True)


def test_production_service_rejects_invalid_service_token_digest():
    settings = SimpleNamespace(
        service_token_sha256="not-a-digest",
        api_principals_json="{}",
    )
    with pytest.raises(RuntimeError, match="SHA-256 digest"):
        validate_production_service_credentials(settings)
