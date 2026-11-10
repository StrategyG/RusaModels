"""Shared registry, upload and audit ORM models."""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def new_id() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class ModelCatalog(Base):
    __tablename__ = "model_catalog"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    publisher: Mapped[str | None] = mapped_column(String(255))
    architecture: Mapped[str | None] = mapped_column(String(128))
    license_id: Mapped[str | None] = mapped_column(String(128))
    source_uri: Mapped[str | None] = mapped_column(String(2048))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_catalog.id", ondelete="RESTRICT"), nullable=False)
    version: Mapped[str] = mapped_column(String(128), nullable=False)
    source_revision: Mapped[str | None] = mapped_column(String(255))
    precision: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="registered")
    manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (Index("ix_artifacts_version_id", "version_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    media_type: Mapped[str | None] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    immutable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UploadSession(Base):
    __tablename__ = "upload_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    storage_upload_id: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False, default="application/octet-stream")
    expected_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    expected_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UploadPart(Base):
    __tablename__ = "upload_parts"
    __table_args__ = (UniqueConstraint("upload_id", "part_number", name="uq_upload_part_number"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    upload_id: Mapped[str] = mapped_column(String(36), ForeignKey("upload_sessions.id", ondelete="CASCADE"), nullable=False)
    part_number: Mapped[int] = mapped_column(Integer, nullable=False)
    etag: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class ModelJob(Base):
    __tablename__ = "model_jobs"
    __table_args__ = (
        Index("ix_model_jobs_status_created_at", "status", "created_at"),
        Index("ix_model_jobs_created_by", "created_by"),
        Index("ix_model_jobs_lease_expires_at", "lease_expires_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_catalog.id", ondelete="RESTRICT"), nullable=False)
    input_version_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("model_versions.id", ondelete="RESTRICT"))
    output_version_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("model_versions.id", ondelete="RESTRICT"))
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    lease_owner: Mapped[str | None] = mapped_column(String(255))
    lease_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_by_role: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_occurred_at", "occurred_at"),
        Index("ix_audit_events_actor_subject", "actor_subject"),
        Index("ix_audit_events_request_id", "request_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    actor_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource: Mapped[str] = mapped_column(String(1024), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
