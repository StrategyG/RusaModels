"""Shared API factory for Model Server and Quant Server."""
import hashlib
import re
import uuid
from datetime import UTC, datetime
from collections.abc import Generator
from typing import Any
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rusamodels.config import Settings, get_settings
from rusamodels.database import build_engine, build_session_factory
from rusamodels.models import (
    Artifact, AuditEvent, ModelCatalog, ModelJob, ModelVersion, UploadPart, UploadSession
)
from rusamodels.schemas import (
    ArtifactOut,
    ConversionCreate,
    HubDownloadCreate,
    JobOut,
    ModelCreate,
    ModelOut,
    ModelPatch,
    UploadCreate,
    UploadOut,
    VersionCreate,
    VersionOut,
)
from rusamodels.hub import search_models
from rusamodels.security import Principal, authenticate, record_audit, require_service_token
from rusamodels.storage import create_storage

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_REQUIRED_TABLES = {
    "model_catalog", "model_versions", "artifacts", "upload_sessions", "upload_parts",
    "model_jobs", "audit_events"
}
_PART_SIZE_LIMIT = 64 * 1024 * 1024
_S3_MIN_PART_SIZE = 5 * 1024 * 1024
_bearer = HTTPBearer(auto_error=False)


def create_service_app(
    service_name: str,
    settings: Settings | None = None,
    *,
    engine=None,
    storage=None,
) -> FastAPI:
    runtime = settings or get_settings()
    db_engine = engine or build_engine(runtime.database_url)
    sessions = build_session_factory(db_engine)
    artifact_storage = storage or create_storage(runtime)
    app = FastAPI(
        title=f"RusaModels {service_name}",
        version="0.2.0",
        description="Independent RUSA-AI model infrastructure control plane.",
    )
    app.state.settings = runtime
    app.state.engine = db_engine
    app.state.sessions = sessions
    app.state.storage = artifact_storage
    app.state.service_name = service_name

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied if _REQUEST_ID.fullmatch(supplied) else str(uuid.uuid4())
        )
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
                    record_audit(
                        db, request, principal, "authorization.denied", request.url.path,
                        "denied", {"required_roles": list(allowed)}
                    )
                except Exception as exc:
                    raise HTTPException(503, "Authorization audit is unavailable") from exc
                raise HTTPException(403, "Insufficient permissions")
            return principal
        return dependency

    def audit(
        db: Session, request: Request, principal: Principal, action: str,
        resource: str, metadata: dict | None = None, outcome: str = "success",
    ):
        record_audit(db, request, principal, action, resource, outcome, metadata)

    def get_upload(db: Session, upload_id: str) -> UploadSession:
        upload = db.get(UploadSession, upload_id)
        if upload is None:
            raise HTTPException(404, "Upload session not found")
        return upload

    def upload_out(upload: UploadSession) -> UploadOut:
        return UploadOut(
            id=upload.id, version_id=upload.version_id,
            filename=upload.original_filename, expected_size=upload.expected_size,
            expected_sha256=upload.expected_sha256, status=upload.status,
            part_size_limit=_PART_SIZE_LIMIT,
        )

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
        return {
            "status": "ready", "service": service_name,
            "database": "ready", "artifact_storage": "ready",
        }

    @app.get("/v1/platform/identity", tags=["platform"])
    def identity(
        request: Request,
        principal: Principal = Depends(roles("viewer", "manager", "administrator")),
        db: Session = Depends(get_db),
    ):
        audit(db, request, principal, "platform.identity.read", "/v1/platform/identity")
        return {"subject": principal.subject, "role": principal.role, "service": service_name}

    @app.get("/v1/models", response_model=list[ModelOut], tags=["models"])
    def list_models(
        request: Request,
        q: str | None = Query(default=None, max_length=255),
        limit: int = Query(default=50, ge=1, le=200),
        principal: Principal = Depends(roles("viewer", "manager", "administrator")),
        db: Session = Depends(get_db),
    ):
        query = (
            select(ModelCatalog).where(ModelCatalog.status == "active")
            .order_by(ModelCatalog.created_at.desc()).limit(limit)
        )
        if q:
            query = query.where(ModelCatalog.name.ilike(f"%{q}%"))
        items = db.scalars(query).all()
        audit(db, request, principal, "models.list", "/v1/models", {"count": len(items)})
        return items

    @app.post("/v1/models", response_model=ModelOut, status_code=201, tags=["models"])
    def create_model(
        payload: ModelCreate,
        request: Request,
        principal: Principal = Depends(roles("manager", "administrator")),
        db: Session = Depends(get_db),
    ):
        item = ModelCatalog(
            name=payload.name.strip(), publisher=payload.publisher,
            architecture=payload.architecture, license_id=payload.license_id,
            source_uri=str(payload.source_uri) if payload.source_uri else None,
            created_by=principal.subject,
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        audit(db, request, principal, "models.create", f"model:{item.id}")
        return item

    @app.get("/v1/models/{model_id}", response_model=ModelOut, tags=["models"])
    def get_model(
        model_id: str,
        request: Request,
        principal: Principal = Depends(roles("viewer", "manager", "administrator")),
        db: Session = Depends(get_db),
    ):
        item = db.get(ModelCatalog, model_id)
        if item is None or item.status != "active":
            raise HTTPException(404, "Model not found")
        audit(db, request, principal, "models.read", f"model:{model_id}")
        return item

    @app.patch("/v1/models/{model_id}", response_model=ModelOut, tags=["models"])
    def patch_model(
        model_id: str,
        payload: ModelPatch,
        request: Request,
        principal: Principal = Depends(roles("manager", "administrator")),
        db: Session = Depends(get_db),
    ):
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
    def archive_model(
        model_id: str,
        request: Request,
        principal: Principal = Depends(roles("administrator")),
        db: Session = Depends(get_db),
    ):
        item = db.get(ModelCatalog, model_id)
        if item is None or item.status != "active":
            raise HTTPException(404, "Model not found")
        item.status = "archived"
        db.commit()
        audit(db, request, principal, "models.archive", f"model:{model_id}")
        return None

    @app.get("/v1/models/{model_id}/versions", response_model=list[VersionOut], tags=["versions"])
    def list_versions(
        model_id: str,
        request: Request,
        principal: Principal = Depends(roles("viewer", "manager", "administrator")),
        db: Session = Depends(get_db),
    ):
        if db.get(ModelCatalog, model_id) is None:
            raise HTTPException(404, "Model not found")
        items = db.scalars(
            select(ModelVersion).where(ModelVersion.model_id == model_id)
            .order_by(ModelVersion.created_at.desc())
        ).all()
        audit(db, request, principal, "versions.list", f"model:{model_id}", {"count": len(items)})
        return items

    @app.post("/v1/models/{model_id}/versions", response_model=VersionOut, status_code=201, tags=["versions"])
    def create_version(
        model_id: str,
        payload: VersionCreate,
        request: Request,
        principal: Principal = Depends(roles("manager", "administrator")),
        db: Session = Depends(get_db),
    ):
        model = db.get(ModelCatalog, model_id)
        if model is None or model.status != "active":
            raise HTTPException(404, "Model not found")
        item = ModelVersion(
            model_id=model_id, version=payload.version,
            source_revision=payload.source_revision, precision=payload.precision,
            manifest_sha256=payload.manifest_sha256.lower() if payload.manifest_sha256 else None,
            created_by=principal.subject,
        )
        db.add(item)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(409, "Model version already exists") from exc
        db.refresh(item)
        audit(db, request, principal, "versions.create", f"model-version:{item.id}")
        return item

    @app.post("/v1/models/{model_id}/versions/{version_id}/uploads", response_model=UploadOut, status_code=201, tags=["artifacts"])
    def create_upload(
        model_id: str,
        version_id: str,
        payload: UploadCreate,
        request: Request,
        principal: Principal = Depends(roles("manager", "administrator")),
        db: Session = Depends(get_db),
    ):

        if service_name != "quant-server":
            raise HTTPException(404, "Artifact writes are available only on Quant Server")
        if payload.expected_size > runtime.max_upload_bytes:
            raise HTTPException(413, "Upload exceeds configured MAX_UPLOAD_BYTES")
        version = db.get(ModelVersion, version_id)
        if version is None or version.model_id != model_id:
            raise HTTPException(404, "Model version not found")
        filename = payload.filename.replace("\\", "/").split("/")[-1]
        filename = _SAFE_FILENAME.sub("_", filename).strip("._")[:255]
        if not filename:
            raise HTTPException(422, "Filename is invalid")
        key = f"models/{model_id}/{version_id}/{uuid.uuid4().hex}/{filename}"
        media_type = payload.media_type or "application/octet-stream"
        try:
            storage_upload_id = artifact_storage.create_multipart(
                key, media_type, {"expected-sha256": payload.expected_sha256.lower()}
            )
        except Exception as exc:
            raise HTTPException(503, "Artifact storage could not start multipart upload") from exc
        item = UploadSession(
            version_id=version_id, storage_key=key, storage_upload_id=storage_upload_id,
            original_filename=filename, media_type=media_type,
            expected_size=payload.expected_size, expected_sha256=payload.expected_sha256.lower(),
            created_by=principal.subject,
        )
        db.add(item)
        try:
            db.commit()
        except Exception:
            db.rollback()
            artifact_storage.abort_multipart(key, storage_upload_id)
            raise
        db.refresh(item)
        audit(db, request, principal, "artifacts.upload_started", f"upload:{item.id}",
              {"version_id": version_id, "expected_size": payload.expected_size})
        return upload_out(item)

    @app.put("/v1/uploads/{upload_id}/parts/{part_number}", tags=["artifacts"])
    async def put_upload_part(
        upload_id: str,
        part_number: int,
        request: Request,
        principal: Principal = Depends(roles("manager", "administrator")),
        db: Session = Depends(get_db),
    ):

        if service_name != "quant-server":
            raise HTTPException(404, "Artifact writes are available only on Quant Server")
        if part_number < 1 or part_number > 10000:
            raise HTTPException(422, "part_number must be between 1 and 10000")
        upload = get_upload(db, upload_id)
        if upload.status != "open":
            raise HTTPException(409, "Upload session is not open")
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > _PART_SIZE_LIMIT:
            raise HTTPException(413, "Upload part exceeds 64 MiB")
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > _PART_SIZE_LIMIT:
                raise HTTPException(413, "Upload part exceeds 64 MiB")
        if not payload:
            raise HTTPException(400, "Upload part must not be empty")
        data = bytes(payload)
        previous_parts = db.scalars(
            select(UploadPart).where(UploadPart.upload_id == upload_id)
        ).all()
        already_uploaded = sum(
            part.size_bytes for part in previous_parts if part.part_number != part_number
        )
        if already_uploaded + len(data) > upload.expected_size:
            raise HTTPException(413, "Uploaded parts exceed expected_size")
        digest = hashlib.sha256(data).hexdigest()
        try:
            etag = artifact_storage.upload_part(
                upload.storage_key, upload.storage_upload_id, part_number, data
            )
        except Exception as exc:
            raise HTTPException(502, "Artifact storage rejected upload part") from exc
        existing = db.scalar(
            select(UploadPart).where(
                UploadPart.upload_id == upload_id, UploadPart.part_number == part_number
            )
        )
        if existing:
            existing.etag = etag
            existing.size_bytes = len(data)
            existing.sha256 = digest
        else:
            db.add(UploadPart(
                upload_id=upload_id, part_number=part_number, etag=etag,
                size_bytes=len(data), sha256=digest,
            ))
        db.commit()
        audit(db, request, principal, "artifacts.upload_part", f"upload:{upload_id}",
              {"part_number": part_number, "size_bytes": len(data), "sha256": digest})
        return {"upload_id": upload_id, "part_number": part_number,
                "size_bytes": len(data), "sha256": digest, "etag": etag}

    @app.get("/v1/uploads/{upload_id}", response_model=UploadOut, tags=["artifacts"])
    def get_upload_status(
        upload_id: str,
        request: Request,
        principal: Principal = Depends(roles("manager", "administrator")),
        db: Session = Depends(get_db),
    ):
        upload = get_upload(db, upload_id)
        audit(db, request, principal, "artifacts.upload_status", f"upload:{upload_id}")
        return upload_out(upload)

    @app.post("/v1/uploads/{upload_id}/complete", response_model=ArtifactOut, tags=["artifacts"])
    def complete_upload(
        upload_id: str,
        request: Request,
        principal: Principal = Depends(roles("manager", "administrator")),
        db: Session = Depends(get_db),
    ):

        if service_name != "quant-server":
            raise HTTPException(404, "Artifact writes are available only on Quant Server")
        upload = get_upload(db, upload_id)
        if upload.status != "open":
            raise HTTPException(409, "Upload session is not open")
        parts = db.scalars(
            select(UploadPart).where(UploadPart.upload_id == upload_id)
            .order_by(UploadPart.part_number.asc())
        ).all()
        if not parts or [part.part_number for part in parts] != list(range(1, len(parts) + 1)):
            raise HTTPException(422, "Upload parts must be contiguous starting at 1")
        if sum(part.size_bytes for part in parts) != upload.expected_size:
            raise HTTPException(422, "Uploaded part sizes do not match expected_size")
        if artifact_storage.backend == "s3" and any(
            part.size_bytes < _S3_MIN_PART_SIZE for part in parts[:-1]
        ):
            raise HTTPException(422, "Non-final S3 multipart parts must be at least 5 MiB")
        completed_parts = [
            {"PartNumber": part.part_number, "ETag": part.etag} for part in parts
        ]
        try:
            artifact_storage.complete_multipart(
                upload.storage_key, upload.storage_upload_id, completed_parts
            )
        except Exception as exc:
            raise HTTPException(502, "Object storage could not complete multipart upload") from exc
        digest = hashlib.sha256()
        total_size = 0
        for chunk in artifact_storage.read_stream(upload.storage_key):
            digest.update(chunk)
            total_size += len(chunk)
        actual_sha256 = digest.hexdigest()
        if total_size != upload.expected_size or actual_sha256 != upload.expected_sha256:
            artifact_storage.delete(upload.storage_key)
            upload.status = "failed"
            db.commit()
            audit(db, request, principal, "artifacts.upload_rejected", f"upload:{upload_id}",
                  {"actual_size": total_size, "actual_sha256": actual_sha256}, "failed")
            raise HTTPException(422, "Completed artifact failed size or SHA-256 verification")
        artifact = Artifact(
            version_id=upload.version_id, storage_key=upload.storage_key,
            original_filename=upload.original_filename, media_type=upload.media_type,
            size_bytes=total_size, sha256=actual_sha256, immutable=True,
        )
        upload.status = "completed"
        db.add(artifact)
        try:
            db.commit()
        except Exception:
            db.rollback()
            artifact_storage.delete(upload.storage_key)
            raise
        db.refresh(artifact)
        audit(db, request, principal, "artifacts.upload_completed", f"artifact:{artifact.id}",
              {"size_bytes": total_size, "sha256": actual_sha256})
        return artifact

    @app.get("/v1/models/{model_id}/versions/{version_id}/artifacts", response_model=list[ArtifactOut], tags=["artifacts"])
    def list_artifacts(
        model_id: str,
        version_id: str,
        request: Request,
        principal: Principal = Depends(roles("viewer", "manager", "administrator")),
        db: Session = Depends(get_db),
    ):
        version = db.get(ModelVersion, version_id)
        if version is None or version.model_id != model_id:
            raise HTTPException(404, "Model version not found")
        items = db.scalars(
            select(Artifact).where(Artifact.version_id == version_id)
            .order_by(Artifact.created_at.desc())
        ).all()
        audit(db, request, principal, "artifacts.list", f"model-version:{version_id}",
              {"count": len(items)})
        return items

    @app.get("/v1/artifacts/{artifact_id}/download", tags=["artifacts"])
    def download_artifact(
        artifact_id: str,
        request: Request,
        principal: Principal = Depends(roles("viewer", "manager", "administrator")),
        db: Session = Depends(get_db),
    ):
        artifact = db.get(Artifact, artifact_id)
        if artifact is None or not artifact.immutable:
            raise HTTPException(404, "Artifact not found")
        audit(db, request, principal, "artifacts.download", f"artifact:{artifact_id}",
              {"size_bytes": artifact.size_bytes, "sha256": artifact.sha256})
        filename = quote(artifact.original_filename or "artifact.bin")
        headers = {
            "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
            "Content-Length": str(artifact.size_bytes),
            "X-Checksum-SHA256": artifact.sha256,
        }
        return StreamingResponse(
            artifact_storage.read_stream(artifact.storage_key),
            media_type=artifact.media_type or "application/octet-stream",
            headers=headers,
        )

    @app.delete("/v1/uploads/{upload_id}", status_code=204, tags=["artifacts"])
    def abort_upload(
        upload_id: str,
        request: Request,
        principal: Principal = Depends(roles("administrator")),
        db: Session = Depends(get_db),
    ):

        if service_name != "quant-server":
            raise HTTPException(404, "Artifact writes are available only on Quant Server")
        upload = get_upload(db, upload_id)
        if upload.status != "open":
            raise HTTPException(409, "Only open uploads can be aborted")
        artifact_storage.abort_multipart(upload.storage_key, upload.storage_upload_id)
        upload.status = "aborted"
        db.commit()
        audit(db, request, principal, "artifacts.upload_aborted", f"upload:{upload_id}")
        return None


    if service_name == "model-server":
        @app.get("/v1/hub/search", tags=["model-hub"])
        def hub_search(
            request: Request,
            q: str = Query(min_length=1, max_length=128),
            limit: int = Query(default=20, ge=1, le=50),
            principal: Principal = Depends(roles("viewer", "manager", "administrator")),
            db: Session = Depends(get_db),
        ):
            try:
                items = search_models(q, limit)
            except Exception as exc:
                raise HTTPException(502, "Hugging Face Hub search is unavailable") from exc
            audit(db, request, principal, "hub.search", "/v1/hub/search",
                  {"query": q, "result_count": len(items)})
            return {"items": items, "count": len(items)}

    if service_name == "quant-server":
        @app.post("/v1/hub/downloads", tags=["model-hub"], status_code=202)
        def create_hub_download(
            payload: HubDownloadCreate,
            request: Request,
            principal: Principal = Depends(roles("manager", "administrator")),
            db: Session = Depends(get_db),
        ):
            if not payload.license_accepted:
                raise HTTPException(400, "Explicit upstream license acceptance is required")
            model = ModelCatalog(
                name=payload.model_name.strip(),
                publisher=payload.repo_id.split("/", maxsplit=1)[0],
                architecture=payload.architecture,
                license_id=payload.license_id.strip().lower(),
                source_uri=f"https://huggingface.co/{payload.repo_id}",
                created_by=principal.subject,
            )
            db.add(model)
            db.flush()
            version = ModelVersion(
                model_id=model.id, version=payload.version,
                source_revision=payload.revision, precision="SOURCE",
                status="queued", created_by=principal.subject,
            )
            db.add(version)
            db.flush()
            job = ModelJob(
                model_id=model.id, output_version_id=version.id, job_type="download",
                parameters={
                    "repo_id": payload.repo_id,
                    "revision": payload.revision,
                    "license_accepted": payload.license_accepted,
                },
                status="queued", progress_percent=0,
                created_by=principal.subject, created_by_role=principal.role,
            )
            db.add(job)
            try:
                db.commit()
            except IntegrityError as exc:
                db.rollback()
                raise HTTPException(409, "Requested model version already exists") from exc
            db.refresh(job)
            audit(db, request, principal, "hub.download_queued", f"job:{job.id}",
                  {
                      "repo_id": payload.repo_id,
                      "revision": payload.revision,
                      "license_id": payload.license_id,
                      "license_accepted": payload.license_accepted,
                  })
            return {
                "job": JobOut.model_validate(job),
                "model_id": model.id,
                "version_id": version.id,
            }

        @app.post(
            "/v1/models/{model_id}/versions/{version_id}/conversion-jobs",
            response_model=JobOut,
            status_code=202,
            tags=["quantization"],
        )
        def create_conversion_job(
            model_id: str,
            version_id: str,
            payload: ConversionCreate,
            request: Request,
            principal: Principal = Depends(roles("manager", "administrator")),
            db: Session = Depends(get_db),
        ):
            source = db.get(ModelVersion, version_id)
            if source is None or source.model_id != model_id:
                raise HTTPException(404, "Source model version not found")
            artifact_exists = db.scalar(
                select(Artifact.id).where(Artifact.version_id == version_id).limit(1)
            )
            if artifact_exists is None:
                raise HTTPException(409, "Source model version has no registered artifacts")
            if payload.quantization and payload.target_format != "gguf":
                raise HTTPException(422, "Integer quantization requires GGUF output")
            precision = (
                f"GGUF_{payload.quantization}" if payload.quantization
                else f"GGUF_{payload.target_precision}" if payload.target_format == "gguf"
                else payload.target_precision
            )
            output_version = ModelVersion(
                model_id=model_id, version=payload.output_version,
                source_revision=source.source_revision, precision=precision,
                status="queued", created_by=principal.subject,
            )
            db.add(output_version)
            try:
                db.flush()
                job = ModelJob(
                    model_id=model_id, input_version_id=version_id,
                    output_version_id=output_version.id, job_type="convert",
                    parameters={
                        "target_format": payload.target_format,
                        "target_precision": payload.target_precision,
                        "quantization": payload.quantization,
                    },
                    status="queued", progress_percent=0,
                    created_by=principal.subject, created_by_role=principal.role,
                )
                db.add(job)
                db.commit()
            except IntegrityError as exc:
                db.rollback()
                raise HTTPException(409, "Output model version already exists") from exc
            db.refresh(job)
            audit(db, request, principal, "quantization.job_queued", f"job:{job.id}",
                  {"source_version_id": version_id, "output_version_id": output_version.id,
                   "target_format": payload.target_format, "precision": precision})
            return job

    @app.get("/v1/jobs/{job_id}", response_model=JobOut, tags=["jobs"])
    def get_job(
        job_id: str,
        request: Request,
        principal: Principal = Depends(roles("viewer", "manager", "administrator")),
        db: Session = Depends(get_db),
    ):
        job = db.get(ModelJob, job_id)
        if job is None:
            raise HTTPException(404, "Model job not found")
        audit(db, request, principal, "model_job.read", f"job:{job_id}")
        return job

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobOut, tags=["jobs"])
    def cancel_job(
        job_id: str,
        request: Request,
        principal: Principal = Depends(roles("administrator")),
        db: Session = Depends(get_db),
    ):
        job = db.get(ModelJob, job_id)
        if job is None:
            raise HTTPException(404, "Model job not found")
        if job.status == "cancelled":
            return job
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = datetime.now(UTC)
            job.next_attempt_at = None
            db.commit()
            audit(db, request, principal, "model_job.cancelled", f"job:{job_id}")
            return job
        if job.status == "running":
            if not job.cancel_requested:
                job.cancel_requested = True
                db.commit()
                audit(
                    db, request, principal, "model_job.cancel_requested",
                    f"job:{job_id}", {"lease_epoch": job.lease_epoch},
                )
            return job
        raise HTTPException(409, "Terminal model jobs cannot be cancelled")

    @app.get("/v1/platform/audit-events", tags=["platform"])
    def list_audit_events(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        principal: Principal = Depends(roles("administrator")),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        rows = db.scalars(
            select(AuditEvent).order_by(AuditEvent.occurred_at.desc()).limit(limit)
        ).all()
        audit(db, request, principal, "audit.events.read", "/v1/platform/audit-events",
              {"limit": limit})
        return {
            "items": [
                {
                    "id": row.id, "actor_subject": row.actor_subject, "actor_role": row.actor_role,
                    "action": row.action, "resource": row.resource,
                    "occurred_at": row.occurred_at.isoformat() if row.occurred_at else None,
                    "outcome": row.outcome, "request_id": row.request_id,
                    "metadata": row.event_metadata,
                }
                for row in rows
            ],
            "count": len(rows),
        }

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
