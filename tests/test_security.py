import json
import pytest
from pydantic import ValidationError
from rusamodels.config import Settings
from rusamodels.security import ROLE_PERMISSIONS, parse_principals, sha256_hex


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
