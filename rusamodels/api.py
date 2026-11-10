"""Shared API factory for independent Model Server and Quant Server."""
import re
import uuid
from collections.abc import Generator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rusamodels.config import Settings, get_settings
from rusamodels.database import build_engine, build_session_factory
from rusamodels.models import AuditEvent, ModelCatalog, ModelVersion
from rusamodels.schemas import ModelCreate, ModelOut, ModelPatch, VersionCreate, VersionOut
from rusamodels.security import Principal, authenticate, record_audit, require_service_token
from rusamodels.storage import create_storage

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_REQUIRED_TABLES = {"model_catalog", "model_versions", "artifacts", "audit_events"}
_bearer = HTTPBearer(auto_error=False)


def create_service_app(service_name: str, settings: Settings | None = None, *, engine=None, storage=None) -> FastAPI:
    runtime = settings or get_settings()
    db_engine = engine or build_engine(runtime.database_url)
    sessions = build_session_factory(db_engine)
    artifact_storage = storage or create_storage(runtime)
    app = FastAPI(title=f"RusaModels {service_name}", version="0.1.0",
                  description="Independent RUSA-AI model infrastructure control plane.")
    app.state.settings = runtime
    app.state.engine = db_engine
    app.state.sessions = sessions
    app.state.storage = artifact_storage
    app.state.service_name = service_name

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = supplied if _REQUEST_ID.fullmatch(supplied) else str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def get_db() -> Generator[Session, None, None]:
        db = sessions()
        try:
            yield db
        finally:
            db.close()

    def current_principal(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> Principal:
        principal = authenticate(request, credentials)
        request.state.principal = principal
        return principal

    def roles(*allowed: str):
        def dependency(
            request: Request,
            principal: Principal = Depends(current_principal),
            db: Session = Depends(get_db),
        ) -> Principal:
            if principal.role not in allowed:
                try:
                    record_audit(db, request, principal, "authorization.denied",
                                 request.url.path, "denied", {"required_roles": list(allowed)})
                except Exception as exc:
                    raise HTTPException(503, "Authorization audit is unavailable") from exc
                raise HTTPException(403, "Insufficient permissions")
            return principal
        return dependency

    def audit(db: Session, request: Request, principal: Principal, action: str, resource: str,
              metadata: dict | None = None):
        record_audit(db, request, principal, action, resource, metadata=metadata)

    @app.get("/health/live", tags=["health"])
    def live():
        return {"status": "alive", "service": service_name}

    @app.get("/health/ready", tags=["health"])
    def ready():
        try:
            with db_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            missing = _REQUIRED_TABLES - set(inspect(db_engine).get_table_names())
            if missing:
                raise RuntimeError("Database migrations are not applied")
            artifact_storage.check_ready()
        except Exception as exc:
            raise HTTPException(503, "Platform dependencies are not ready") from exc
        return {"status": "ready", "service": service_name,
                "database": "ready", "artifact_storage": "ready"}

    @app.get("/v1/platform/identity", tags=["platform"])
    def identity(request: Request, principal: Principal = Depends(roles("viewer", "manager", "administrator")),
                 db: Session = Depends(get_db)):
        audit(db, request, principal, "platform.identity.read", "/v1/platform/identity")
        return {"subject": principal.subject, "role": principal.role, "service": service_name}

    @app.get("/v1/models", response_model=list[ModelOut], tags=["models"])
    def list_models(request: Request, q: str | None = Query(default=None, max_length=255),
                    limit: int = Query(default=50, ge=1, le=200),
                    principal: Principal = Depends(roles("viewer", "manager", "administrator")),
                    db: Session = Depends(get_db)):
        query = select(ModelCatalog).where(ModelCatalog.status == "active").order_by(ModelCatalog.created_at.desc()).limit(limit)
        if q:
            query = query.where(ModelCatalog.name.ilike(f"%{q}%"))
        items = db.scalars(query).all()
        audit(db, request, principal, "models.list", "/v1/models", {"count": len(items)})
        return items

    @app.post("/v1/models", response_model=ModelOut, status_code=201, tags=["models"])
    def create_model(payload: ModelCreate, request: Request,
                     principal: Principal = Depends(roles("manager", "administrator")),
                     db: Session = Depends(get_db)):
        item = ModelCatalog(name=payload.name.strip(), publisher=payload.publisher,
                            architecture=payload.architecture, license_id=payload.license_id,
                            source_uri=str(payload.source_uri) if payload.source_uri else None,
                            created_by=principal.subject)
        db.add(item)
        db.commit()
        db.refresh(item)
        audit(db, request, principal, "models.create", f"model:{item.id}")
        return item

    @app.get("/v1/models/{model_id}", response_model=ModelOut, tags=["models"])
    def get_model(model_id: str, request: Request,
                  principal: Principal = Depends(roles("viewer", "manager", "administrator")),
                  db: Session = Depends(get_db)):
        item = db.get(ModelCatalog, model_id)
        if item is None or item.status != "active":
            raise HTTPException(404, "Model not found")
        audit(db, request, principal, "models.read", f"model:{model_id}")
        return item

    @app.patch("/v1/models/{model_id}", response_model=ModelOut, tags=["models"])
    def patch_model(model_id: str, payload: ModelPatch, request: Request,
                    principal: Principal = Depends(roles("manager", "administrator")),
                    db: Session = Depends(get_db)):
        item = db.get(ModelCatalog, model_id)
        if item is None or item.status != "active":
            raise HTTPException(404, "Model not found")
        for key, value in payload.model_dump(exclude_unset=True).items():
            setattr(item, key, value)
        db.commit()
        db.refresh(item)
        audit(db, request, principal, "models.update", f"model:{model_id}")
        return item

    @app.delete("/v1/models/{model_id}", status_code=204, tags=["models"])
    def archive_model(model_id: str, request: Request,
                      principal: Principal = Depends(roles("administrator")),
                      db: Session = Depends(get_db)):
        item = db.get(ModelCatalog, model_id)
        if item is None or item.status != "active":
            raise HTTPException(404, "Model not found")
        item.status = "archived"
        db.commit()
        audit(db, request, principal, "models.archive", f"model:{model_id}")
        return None

    @app.get("/v1/models/{model_id}/versions", response_model=list[VersionOut], tags=["versions"])
    def list_versions(model_id: str, request: Request,
                      principal: Principal = Depends(roles("viewer", "manager", "administrator")),
                      db: Session = Depends(get_db)):
        if db.get(ModelCatalog, model_id) is None:
            raise HTTPException(404, "Model not found")
        items = db.scalars(select(ModelVersion).where(ModelVersion.model_id == model_id)
                           .order_by(ModelVersion.created_at.desc())).all()
        audit(db, request, principal, "versions.list", f"model:{model_id}", {"count": len(items)})
        return items

    @app.post("/v1/models/{model_id}/versions", response_model=VersionOut, status_code=201, tags=["versions"])
    def create_version(model_id: str, payload: VersionCreate, request: Request,
                       principal: Principal = Depends(roles("manager", "administrator")),
                       db: Session = Depends(get_db)):
        model = db.get(ModelCatalog, model_id)
        if model is None or model.status != "active":
            raise HTTPException(404, "Model not found")
        item = ModelVersion(model_id=model_id, version=payload.version,
                            source_revision=payload.source_revision, precision=payload.precision,
                            manifest_sha256=payload.manifest_sha256.lower() if payload.manifest_sha256 else None,
                            created_by=principal.subject)
        db.add(item)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(409, "Model version already exists") from exc
        db.refresh(item)
        audit(db, request, principal, "versions.create", f"model-version:{item.id}")
        return item

    @app.get("/v1/platform/audit-events", tags=["platform"])
    def list_audit_events(request: Request, limit: int = Query(default=50, ge=1, le=200),
                          principal: Principal = Depends(roles("administrator")),
                          db: Session = Depends(get_db)) -> dict[str, Any]:
        rows = db.scalars(select(AuditEvent).order_by(AuditEvent.occurred_at.desc()).limit(limit)).all()
        audit(db, request, principal, "audit.events.read", "/v1/platform/audit-events", {"limit": limit})
        return {"items": [{"id": x.id, "actor_subject": x.actor_subject, "actor_role": x.actor_role,
                           "action": x.action, "resource": x.resource,
                           "occurred_at": x.occurred_at.isoformat() if x.occurred_at else None,
                           "outcome": x.outcome, "request_id": x.request_id,
                           "metadata": x.event_metadata} for x in rows], "count": len(rows)}

    @app.get("/internal/v1/platform/dependencies", tags=["internal"])
    def internal_dependencies(_service: str = Depends(require_service_token)):
        try:
            with db_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            artifact_storage.check_ready()
        except Exception as exc:
            raise HTTPException(503, "Shared dependencies are not ready") from exc
        return {"status": "ready", "service": service_name}

    return app
