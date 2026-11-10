"""Token authentication, RBAC and append-only audit writes."""
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from rusamodels.models import AuditEvent

Role = Literal["viewer", "manager", "administrator"]
ROLE_PERMISSIONS = {
    "viewer": frozenset({"platform:read", "models:read", "artifacts:read"}),
    "manager": frozenset({"platform:read", "models:read", "artifacts:read", "models:write", "quantization:run"}),
    "administrator": frozenset({"platform:read", "models:read", "artifacts:read", "models:write", "quantization:run", "admin:destructive", "audit:read", "users:manage"}),
}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    subject: str
    role: Role


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def parse_principals(raw: str) -> dict[str, Principal]:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("RUSAMODELS_API_PRINCIPALS must be valid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError("RUSAMODELS_API_PRINCIPALS must be an object")
    result = {}
    for digest, item in data.items():
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise RuntimeError("API principal keys must be lowercase SHA-256 digests")
        if not isinstance(item, dict) or not isinstance(item.get("subject"), str):
            raise RuntimeError("Each API principal needs a subject and role")
        if item.get("role") not in ROLE_PERMISSIONS:
            raise RuntimeError("Role must be viewer, manager or administrator")
        result[digest] = Principal(item["subject"], item["role"])
    return result


def authenticate(request: Request, credentials: HTTPAuthorizationCredentials | None) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Bearer API token required", headers={"WWW-Authenticate": "Bearer"})
    supplied = sha256_hex(credentials.credentials)
    for stored, principal in parse_principals(request.app.state.settings.api_principals_json).items():
        if hmac.compare_digest(supplied, stored):
            return principal
    raise HTTPException(401, "Invalid API token", headers={"WWW-Authenticate": "Bearer"})


def get_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    principal = authenticate(request, credentials)
    request.state.principal = principal
    return principal


def require_roles(*roles: Role):
    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(403, "Insufficient permissions")
        return principal
    return dependency


def record_audit(db: Session, request: Request, principal: Principal, action: str, resource: str,
                 outcome: str = "success", metadata: dict | None = None) -> None:
    db.add(AuditEvent(
        actor_subject=principal.subject, actor_role=principal.role, action=action,
        resource=resource, outcome=outcome, request_id=request.state.request_id,
        event_metadata=metadata or {},
    ))
    db.commit()


def require_service_token(
    request: Request,
    token: str | None = Header(default=None, alias="X-Rusa-Service-Token"),
) -> str:
    expected = request.app.state.settings.service_token_sha256
    if not expected:
        raise HTTPException(503, "Service authentication is not configured")
    if not token or not hmac.compare_digest(sha256_hex(token), expected):
        raise HTTPException(401, "Invalid service credential")
    return "authenticated-service"
