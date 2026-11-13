"""Read-only verification of a restored model artifact inventory."""
from __future__ import annotations

import hashlib
import hmac
import json

from sqlalchemy import select

from rusamodels.config import get_settings
from rusamodels.database import build_engine, build_session_factory
from rusamodels.manifests import (
    ModelManifest,
    canonical_manifest_sha256,
    manifest_artifacts_match,
)
from rusamodels.models import Artifact, ModelVersion
from rusamodels.storage import create_storage


def verify_artifact_inventory(session, storage) -> dict:
    """Verify every non-purged artifact referenced by the registry."""
    artifacts = session.scalars(
        select(Artifact)
        .where(Artifact.deleted_at.is_(None))
        .order_by(Artifact.id.asc())
    ).all()
    errors = []
    verified = 0
    for artifact in artifacts:
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            for chunk in storage.read_stream(artifact.storage_key):
                digest.update(chunk)
                size_bytes += len(chunk)
        except Exception as exc:
            errors.append({
                "artifact_id": artifact.id,
                "error": "read_failed",
                "error_type": type(exc).__name__,
            })
            continue
        actual_sha256 = digest.hexdigest()
        if (
            size_bytes != artifact.size_bytes
            or not hmac.compare_digest(actual_sha256, artifact.sha256.lower())
        ):
            errors.append({
                "artifact_id": artifact.id,
                "error": "integrity_mismatch",
                "expected_size_bytes": artifact.size_bytes,
                "actual_size_bytes": size_bytes,
                "expected_sha256": artifact.sha256,
                "actual_sha256": actual_sha256,
            })
            continue
        verified += 1
    versions = session.scalars(
        select(ModelVersion)
        .where(ModelVersion.manifest_json.is_not(None))
        .order_by(ModelVersion.id.asc())
    ).all()
    verified_manifests = 0
    for version in versions:
        try:
            manifest = ModelManifest.model_validate(version.manifest_json)
        except Exception as exc:
            errors.append({
                "model_version_id": version.id,
                "error": "manifest_invalid",
                "error_type": type(exc).__name__,
            })
            continue
        digest = canonical_manifest_sha256(manifest)
        if (
            not version.manifest_sha256
            or not hmac.compare_digest(digest, version.manifest_sha256.lower())
        ):
            errors.append({
                "model_version_id": version.id,
                "error": "manifest_digest_mismatch",
            })
            continue
        if version.lifecycle_status != "archived":
            version_artifacts = session.scalars(
                select(Artifact).where(Artifact.version_id == version.id)
            ).all()
            if not manifest_artifacts_match(manifest, version_artifacts):
                errors.append({
                    "model_version_id": version.id,
                    "error": "manifest_artifact_inventory_mismatch",
                })
                continue
        verified_manifests += 1
    return {
        "artifact_count": len(artifacts),
        "verified_count": verified,
        "model_manifest_count": len(versions),
        "verified_manifest_count": verified_manifests,
        "failed_count": len(errors),
        "errors": errors,
    }


def main() -> int:
    settings = get_settings()
    engine = build_engine(settings.database_url)
    session_factory = build_session_factory(engine)
    storage = create_storage(settings)
    session = session_factory()
    try:
        report = verify_artifact_inventory(session, storage)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if report["failed_count"] else 0
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
