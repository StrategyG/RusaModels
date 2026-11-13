"""Background worker for Hugging Face downloads and model conversion jobs."""
import argparse
import hashlib
import json
import random
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from httpx import ConnectError, TimeoutException
from sqlalchemy import and_, or_, select, update

from rusamodels.config import get_settings
from rusamodels.database import build_engine, build_session_factory
from rusamodels.models import Artifact, AuditEvent, ModelCatalog, ModelJob, ModelVersion
from rusamodels.quantization.engine import ConversionEngine
from rusamodels.quantization.preflight import preflight_conversion
from rusamodels.storage import create_storage

_PART_BYTES = 64 * 1024 * 1024


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _safe_filename(path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name).strip("._")
    return (name or "artifact.bin")[:255]


class JobCancelled(Exception):
    """Raised when an administrator requests cooperative job cancellation."""


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Model job cancellation was requested")


def _upload_tree(
    storage,
    files: list[Path],
    model_id: str,
    version_id: str,
    cancel_event: threading.Event | None = None,
) -> tuple[list[Artifact], list[str]]:
    artifacts = []
    keys = []
    try:
        for file_path in files:
            _check_cancelled(cancel_event)
            if not file_path.is_file() or file_path.is_symlink():
                continue
            digest, size = _sha256_file(file_path)
            if size <= 0:
                continue
            filename = _safe_filename(file_path)
            key = f"models/{model_id}/{version_id}/{uuid.uuid4().hex}/{filename}"
            upload_id = storage.create_multipart(
                key, "application/octet-stream", {"sha256": digest}
            )
            parts = []
            try:
                with file_path.open("rb") as source:
                    number = 1
                    while chunk := source.read(_PART_BYTES):
                        _check_cancelled(cancel_event)
                        etag = storage.upload_part(key, upload_id, number, chunk)
                        parts.append({"PartNumber": number, "ETag": etag})
                        number += 1
                storage.complete_multipart(key, upload_id, parts)
            except Exception:
                storage.abort_multipart(key, upload_id)
                raise
            verify = hashlib.sha256()
            verified_size = 0
            for chunk in storage.read_stream(key):
                verify.update(chunk)
                verified_size += len(chunk)
            if verified_size != size or verify.hexdigest() != digest:
                storage.delete(key)
                raise RuntimeError("Uploaded object failed SHA-256 verification")
            keys.append(key)
            artifacts.append(Artifact(
                version_id=version_id,
                storage_key=key,
                original_filename=filename,
                media_type="application/octet-stream",
                size_bytes=size,
                sha256=digest,
                immutable=True,
            ))
        if not artifacts:
            raise RuntimeError("No non-empty model artifacts were produced")
        return artifacts, keys
    except Exception:
        for key in keys:
            try:
                storage.delete(key)
            except Exception:
                pass
        raise


def _audit(
    db, job: ModelJob, action: str, outcome: str, *, status: str | None = None
) -> None:
    db.add(AuditEvent(
        actor_subject=job.created_by,
        actor_role=job.created_by_role,
        action=action,
        resource=f"job:{job.id}",
        outcome=outcome,
        request_id=f"job-{job.id}",
        event_metadata={
            "job_type": job.job_type,
            "status": status or job.status,
            "attempt": job.attempt_count,
        },
    ))


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        return status in {408, 429} or status >= 500
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", 0) or 0
        return status in {408, 429} or status >= 500
    return isinstance(
        exc, (TimeoutError, ConnectionError, ConnectError, TimeoutException, BotoCoreError)
    )


def _heartbeat_loop(
    job_id: str,
    sessions,
    worker_id: str,
    lease_epoch: int,
    stop_event: threading.Event,
    lease_lost: threading.Event,
    cancel_event: threading.Event,
    lease_seconds: int,
) -> None:
    interval = max(1, lease_seconds // 3)
    while not stop_event.wait(interval):
        db = sessions()
        try:
            now = datetime.now(UTC)
            result = db.execute(
                update(ModelJob)
                .where(
                    ModelJob.id == job_id,
                    ModelJob.status == "running",
                    ModelJob.lease_owner == worker_id,
                    ModelJob.lease_epoch == lease_epoch,
                )
                .values(
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    last_heartbeat_at=now,
                )
            )
            db.commit()
            if result.rowcount != 1:
                lease_lost.set()
                return
            if db.scalar(select(ModelJob.cancel_requested).where(ModelJob.id == job_id)):
                cancel_event.set()
        except Exception:
            db.rollback()
            lease_lost.set()
            return
        finally:
            db.close()


def _download_job(
    db, job: ModelJob, settings, storage, work_dir: Path,
    cancel_event: threading.Event | None = None,
) -> list[Artifact]:
    _check_cancelled(cancel_event)
    model = db.get(ModelCatalog, job.model_id)
    version = db.get(ModelVersion, job.output_version_id)
    if model is None or version is None:
        raise RuntimeError("Download job references missing model metadata")
    if not model.license_id or not job.parameters.get("license_accepted"):
        raise RuntimeError("Model license must be identified and explicitly accepted")
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    api = HfApi(token=token)
    repo_id = job.parameters["repo_id"]
    revision = job.parameters.get("resolved_revision") or job.parameters["revision"]
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    _check_cancelled(cancel_event)
    card = getattr(info, "card_data", None)
    upstream_license = getattr(card, "license", None) if card else None
    if not upstream_license:
        raise RuntimeError("Upstream model card does not declare a license; manual review is required")
    if model.license_id.lower() != str(upstream_license).lower():
        raise RuntimeError("Submitted license identifier does not match the upstream model card")
    resolved_revision = getattr(info, "sha", None)
    if not isinstance(resolved_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", resolved_revision):
        raise RuntimeError("Hub did not return a valid immutable commit revision")
    version.source_revision = resolved_revision.lower()
    job.parameters = {
        **job.parameters,
        "resolved_revision": resolved_revision.lower(),
        "upstream_license": str(upstream_license).lower(),
    }
    db.flush()
    if job.lease_owner:
        persisted = db.execute(
            update(ModelJob)
            .where(
                ModelJob.id == job.id,
                ModelJob.status == "running",
                ModelJob.lease_owner == job.lease_owner,
                ModelJob.lease_epoch == job.lease_epoch,
            )
            .values(parameters=job.parameters)
        )
        if persisted.rowcount != 1:
            db.rollback()
            raise RuntimeError("Worker lease was lost before persisting Hub revision")
        db.commit()
    sibling_sizes = [
        int(item.size) for item in (getattr(info, "siblings", None) or [])
        if getattr(item, "size", None) is not None
    ]
    known_total = sum(sibling_sizes)
    if known_total and known_total > settings.max_upload_bytes:
        raise RuntimeError("Model repository exceeds configured MAX_UPLOAD_BYTES")
    free_disk = shutil.disk_usage(work_dir).free
    if known_total and int(known_total * 1.25) > free_disk:
        raise RuntimeError("Insufficient temporary disk space for model download")
    local_dir = work_dir / "snapshot"
    snapshot_download(
        repo_id=repo_id,
        revision=resolved_revision,
        token=token,
        local_dir=str(local_dir),
    )
    _check_cancelled(cancel_event)
    files = [
        path for path in local_dir.rglob("*")
        if path.is_file() and not path.is_symlink() and ".cache" not in path.parts
    ]
    actual_total = sum(path.stat().st_size for path in files)
    if actual_total > settings.max_upload_bytes:
        raise RuntimeError("Downloaded model exceeds configured MAX_UPLOAD_BYTES")
    if actual_total > shutil.disk_usage(work_dir).total:
        raise RuntimeError("Downloaded model exceeds available temporary disk")
    artifacts, _keys = _upload_tree(
        storage, files, job.model_id, version.id, cancel_event
    )
    return artifacts


def _conversion_job(
    db, job: ModelJob, settings, storage, work_dir: Path,
    cancel_event: threading.Event | None = None,
) -> list[Artifact]:
    _check_cancelled(cancel_event)
    source_version = db.get(ModelVersion, job.input_version_id)
    output_version = db.get(ModelVersion, job.output_version_id)
    if source_version is None or output_version is None:
        raise RuntimeError("Conversion job references missing model versions")
    source_artifacts = db.scalars(
        select(Artifact).where(Artifact.version_id == source_version.id)
    ).all()
    if not source_artifacts:
        raise RuntimeError("Source model version has no artifacts")
    source_bytes = sum(item.size_bytes for item in source_artifacts)
    if source_bytes > settings.max_upload_bytes:
        raise RuntimeError("Source model exceeds configured MAX_UPLOAD_BYTES")
    source_dir = work_dir / "source"
    output_dir = work_dir / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    for artifact in source_artifacts:
        _check_cancelled(cancel_event)
        filename = _safe_filename(Path(artifact.original_filename or artifact.id))
        target = source_dir / filename
        if target.exists():
            target = source_dir / f"{artifact.id[:8]}-{filename}"
        digest = hashlib.sha256()
        with target.open("xb") as output:
            for chunk in storage.read_stream(artifact.storage_key):
                _check_cancelled(cancel_event)
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != artifact.sha256 or target.stat().st_size != artifact.size_bytes:
            raise RuntimeError("Source artifact failed checksum verification")
    params = job.parameters
    preflight_conversion(
        source_bytes,
        work_dir,
        target_format=params["target_format"],
    )
    _check_cancelled(cancel_event)
    engine = ConversionEngine()
    outputs = engine.convert(
        source_dir,
        output_dir,
        target_format=params["target_format"],
        target_precision=params["target_precision"],
        quantization=params.get("quantization"),
        cancel_event=cancel_event,
    )
    _check_cancelled(cancel_event)
    return _upload_tree(
        storage, outputs, job.model_id, output_version.id, cancel_event
    )[0]


def process_job(
    job_id: str,
    sessions,
    settings,
    storage,
    worker_id: str,
    lease_seconds: int = 30,
) -> None:
    db = sessions()
    uploaded_keys: list[str] = []
    stop_event = threading.Event()
    lease_lost = threading.Event()
    cancel_event = threading.Event()
    heartbeat = None
    lease_epoch = None
    try:
        job = db.get(ModelJob, job_id)
        if job is None or job.status != "running" or job.lease_owner != worker_id:
            return
        lease_epoch = job.lease_epoch
        heartbeat = threading.Thread(
            target=_heartbeat_loop,
            args=(
                job_id, sessions, worker_id, lease_epoch, stop_event,
                lease_lost, cancel_event, lease_seconds,
            ),
            name=f"job-heartbeat-{job_id[:8]}",
            daemon=True,
        )
        heartbeat.start()
        with tempfile.TemporaryDirectory(prefix="rusamodels-job-") as temporary:
            work_dir = Path(temporary)
            if job.job_type == "download":
                artifacts = _download_job(
                    db, job, settings, storage, work_dir, cancel_event
                )
            elif job.job_type == "convert":
                artifacts = _conversion_job(
                    db, job, settings, storage, work_dir, cancel_event
                )
            else:
                raise RuntimeError("Unsupported model job type")
            uploaded_keys = [item.storage_key for item in artifacts]
            _check_cancelled(cancel_event)
            if lease_lost.is_set():
                db.rollback()
                for key in uploaded_keys:
                    storage.delete(key)
                return
            output_version = db.get(ModelVersion, job.output_version_id)
            manifest = [
                {"filename": item.original_filename, "size_bytes": item.size_bytes, "sha256": item.sha256}
                for item in sorted(artifacts, key=lambda value: value.original_filename or "")
            ]
            manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            output_version.artifact_index_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            output_version.status = "ready"
            result = db.execute(
                update(ModelJob)
                .where(
                    ModelJob.id == job_id,
                    ModelJob.status == "running",
                    ModelJob.lease_owner == worker_id,
                    ModelJob.lease_epoch == lease_epoch,
                    ModelJob.cancel_requested.is_(False),
                )
                .values(
                    status="succeeded",
                    progress_percent=100,
                    finished_at=datetime.now(UTC),
                    lease_owner=None,
                    lease_expires_at=None,
                )
            )
            if result.rowcount != 1:
                db.rollback()
                current = db.get(ModelJob, job_id)
                if (
                    current is not None
                    and current.status == "running"
                    and current.lease_owner == worker_id
                    and current.lease_epoch == lease_epoch
                    and current.cancel_requested
                ):
                    raise JobCancelled("Model job cancellation was requested")
                for key in uploaded_keys:
                    storage.delete(key)
                return
            for artifact in artifacts:
                artifact.created_by = job.created_by
                artifact.source_job_id = job.id
                artifact.provenance = {
                    "origin": "model_job",
                    "job_id": job.id,
                    "job_type": job.job_type,
                    "source_version_id": job.input_version_id,
                    "actor_subject": job.created_by,
                }
                artifact.retention_class = "standard"
                artifact.verified_at = datetime.now(UTC)
            db.add_all(artifacts)
            _audit(db, job, "model_job.succeeded", "success", status="succeeded")
            try:
                db.commit()
            except Exception:
                db.rollback()
                for key in uploaded_keys:
                    storage.delete(key)
                raise
    except Exception as exc:
        db.rollback()
        job = db.get(ModelJob, job_id)
        if (
            job is not None
            and job.status == "running"
            and job.lease_owner == worker_id
            and job.lease_epoch == lease_epoch
        ):
            job.error_code = type(exc).__name__[:128]
            job.lease_owner = None
            job.lease_expires_at = None
            if isinstance(exc, JobCancelled) or cancel_event.is_set() or job.cancel_requested:
                for key in uploaded_keys:
                    try:
                        storage.delete(key)
                    except Exception:
                        pass
                job.status = "cancelled"
                job.finished_at = datetime.now(UTC)
                job.next_attempt_at = None
                status = "cancelled"
                if job.output_version_id:
                    version = db.get(ModelVersion, job.output_version_id)
                    if version is not None:
                        version.status = "cancelled"
                _audit(db, job, "model_job.cancelled", "success", status=status)
                db.commit()
                return
            retry = _is_retryable(exc) and job.attempt_count < job.max_attempts
            if retry:
                backoff_cap = min(300, 5 * (2 ** max(0, job.attempt_count - 1)))
                job.status = "queued"
                job.next_attempt_at = datetime.now(UTC) + timedelta(
                    seconds=random.uniform(0, backoff_cap)
                )
                job.finished_at = None
                status = "queued"
                outcome = "retry_scheduled"
                if job.output_version_id:
                    version = db.get(ModelVersion, job.output_version_id)
                    if version is not None:
                        version.status = "queued"
            else:
                job.status = "failed"
                job.finished_at = datetime.now(UTC)
                job.next_attempt_at = None
                status = "failed"
                outcome = "failed"
                if job.output_version_id:
                    version = db.get(ModelVersion, job.output_version_id)
                    if version is not None:
                        version.status = "failed"
            _audit(db, job, "model_job.failed", outcome, status=status)
            db.commit()
    finally:
        stop_event.set()
        if heartbeat is not None:
            heartbeat.join(timeout=max(2, lease_seconds))
        db.close()


def claim_job(sessions, worker_id: str, lease_seconds: int = 30) -> str | None:
    db = sessions()
    try:
        now = datetime.now(UTC)
        eligible = or_(
            and_(
                ModelJob.status == "queued",
                or_(ModelJob.next_attempt_at.is_(None), ModelJob.next_attempt_at <= now),
            ),
            and_(
                ModelJob.status == "running",
                or_(ModelJob.lease_expires_at.is_(None), ModelJob.lease_expires_at <= now),
            ),
        )
        job = db.scalar(
            select(ModelJob)
            .where(eligible)
            .order_by(ModelJob.created_at.asc())
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if job is None:
            db.rollback()
            return None
        if job.status == "running" and job.cancel_requested:
            job.status = "cancelled"
            job.finished_at = now
            job.lease_owner = None
            job.lease_expires_at = None
            if job.output_version_id:
                version = db.get(ModelVersion, job.output_version_id)
                if version is not None:
                    version.status = "cancelled"
            _audit(db, job, "model_job.cancelled", "success", status="cancelled")
            db.commit()
            return None
        if job.status == "running" and job.attempt_count >= job.max_attempts:
            job.status = "failed"
            job.error_code = "WorkerLeaseExpired"
            job.finished_at = now
            job.lease_owner = None
            job.lease_expires_at = None
            if job.output_version_id:
                version = db.get(ModelVersion, job.output_version_id)
                if version is not None:
                    version.status = "failed"
            _audit(db, job, "model_job.lease_expired", "failed", status="failed")
            db.commit()
            return None
        job.status = "running"
        job.attempt_count += 1
        job.lease_epoch += 1
        job.lease_owner = worker_id
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)
        job.last_heartbeat_at = now
        job.next_attempt_at = None
        if job.started_at is None:
            job.started_at = now
        db.commit()
        return job.id
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="RusaModels model job worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    args = parser.parse_args()
    settings = get_settings()
    if (
        settings.environment in {"staging", "production"}
        and settings.s3_access_key_id
        and not settings.s3_write_access_key_id
    ):
        raise RuntimeError(
            "Quant Worker requires distinct S3 write credentials when static read credentials are used"
        )
    engine = build_engine(settings.database_url)
    sessions = build_session_factory(engine)
    storage = create_storage(settings)
    worker_id = str(uuid.uuid4())
    while True:
        job_id = claim_job(sessions, worker_id)
        if job_id:
            process_job(job_id, sessions, settings, storage, worker_id)
            if args.once:
                break
        elif args.once:
            break
        else:
            time.sleep(max(1, settings.worker_poll_seconds))
    engine.dispose()


if __name__ == "__main__":
    main()
