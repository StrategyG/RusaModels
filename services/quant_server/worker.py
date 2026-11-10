"""Background worker for Hugging Face downloads and model conversion jobs."""
import argparse
import hashlib
import json
import re
import shutil
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from sqlalchemy import select

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


def _upload_tree(storage, files: list[Path], model_id: str, version_id: str) -> tuple[list[Artifact], list[str]]:
    artifacts = []
    keys = []
    try:
        for file_path in files:
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


def _audit(db, job: ModelJob, action: str, outcome: str) -> None:
    db.add(AuditEvent(
        actor_subject=job.created_by,
        actor_role=job.created_by_role,
        action=action,
        resource=f"job:{job.id}",
        outcome=outcome,
        request_id=f"job-{job.id}",
        event_metadata={"job_type": job.job_type, "status": job.status},
    ))


def _download_job(db, job: ModelJob, settings, storage, work_dir: Path) -> list[Artifact]:
    model = db.get(ModelCatalog, job.model_id)
    version = db.get(ModelVersion, job.output_version_id)
    if model is None or version is None:
        raise RuntimeError("Download job references missing model metadata")
    if not model.license_id or not job.parameters.get("license_accepted"):
        raise RuntimeError("Model license must be identified and explicitly accepted")
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    api = HfApi(token=token)
    repo_id = job.parameters["repo_id"]
    revision = job.parameters["revision"]
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
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
    files = [
        path for path in local_dir.rglob("*")
        if path.is_file() and not path.is_symlink() and ".cache" not in path.parts
    ]
    actual_total = sum(path.stat().st_size for path in files)
    if actual_total > settings.max_upload_bytes:
        raise RuntimeError("Downloaded model exceeds configured MAX_UPLOAD_BYTES")
    if actual_total > shutil.disk_usage(work_dir).total:
        raise RuntimeError("Downloaded model exceeds available temporary disk")
    artifacts, _keys = _upload_tree(storage, files, job.model_id, version.id)
    return artifacts


def _conversion_job(db, job: ModelJob, settings, storage, work_dir: Path) -> list[Artifact]:
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
        filename = _safe_filename(Path(artifact.original_filename or artifact.id))
        target = source_dir / filename
        if target.exists():
            target = source_dir / f"{artifact.id[:8]}-{filename}"
        digest = hashlib.sha256()
        with target.open("xb") as output:
            for chunk in storage.read_stream(artifact.storage_key):
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
    engine = ConversionEngine()
    outputs = engine.convert(
        source_dir,
        output_dir,
        target_format=params["target_format"],
        target_precision=params["target_precision"],
        quantization=params.get("quantization"),
    )
    return _upload_tree(storage, outputs, job.model_id, output_version.id)[0]


def process_job(job_id: str, sessions, settings, storage) -> None:
    db = sessions()
    uploaded_keys: list[str] = []
    try:
        job = db.get(ModelJob, job_id)
        if job is None or job.status != "running":
            return
        with tempfile.TemporaryDirectory(prefix="rusamodels-job-") as temporary:
            work_dir = Path(temporary)
            if job.job_type == "download":
                artifacts = _download_job(db, job, settings, storage, work_dir)
            elif job.job_type == "convert":
                artifacts = _conversion_job(db, job, settings, storage, work_dir)
            else:
                raise RuntimeError("Unsupported model job type")
            uploaded_keys = [item.storage_key for item in artifacts]
            output_version = db.get(ModelVersion, job.output_version_id)
            manifest = [
                {"filename": item.original_filename, "size_bytes": item.size_bytes, "sha256": item.sha256}
                for item in sorted(artifacts, key=lambda value: value.original_filename or "")
            ]
            manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            output_version.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            output_version.status = "ready"
            db.add_all(artifacts)
            job.status = "succeeded"
            job.progress_percent = 100
            job.finished_at = datetime.now(UTC)
            _audit(db, job, "model_job.succeeded", "success")
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
        if job is not None:
            job.status = "failed"
            job.error_code = type(exc).__name__[:128]
            job.finished_at = datetime.now(UTC)
            if job.output_version_id:
                version = db.get(ModelVersion, job.output_version_id)
                if version is not None:
                    version.status = "failed"
            _audit(db, job, "model_job.failed", "failed")
            db.commit()
    finally:
        db.close()


def claim_job(sessions) -> str | None:
    db = sessions()
    try:
        job = db.scalar(
            select(ModelJob)
            .where(ModelJob.status == "queued")
            .order_by(ModelJob.created_at.asc())
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if job is None:
            db.rollback()
            return None
        job.status = "running"
        job.started_at = datetime.now(UTC)
        db.commit()
        return job.id
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="RusaModels model job worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    args = parser.parse_args()
    settings = get_settings()
    engine = build_engine(settings.database_url)
    sessions = build_session_factory(engine)
    storage = create_storage(settings)
    while True:
        job_id = claim_job(sessions)
        if job_id:
            process_job(job_id, sessions, settings, storage)
            if args.once:
                break
        elif args.once:
            break
        else:
            time.sleep(max(1, settings.worker_poll_seconds))
    engine.dispose()


if __name__ == "__main__":
    main()
