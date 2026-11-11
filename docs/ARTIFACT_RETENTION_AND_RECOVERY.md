# Model registry, artifact retention and recovery

## Scope

This runbook describes the Phase 1 registry contract introduced by the
`0006_model_manifest_artifact_governance` migration.
It complements, but does not replace, the deployment-specific database and
object-storage backup procedures.

## Model version lifecycle

A model version has two independent statuses:

- `status`: operational job state (for example `queued`,
  `ready`, or `failed`).
- `lifecycle_status`: release governance state:
  `draft` → `candidate` → `evaluated` → `approved`,
  followed by `deprecated` and `archived`.

A manifest is attached once to a draft version. It is canonicalized as UTF-8
JSON (sorted keys, compact separators) and SHA-256 is stored in the registry.
The migration reconstructs the legacy worker file-list digest from
registered artifact metadata. Matching values move to
`artifact_index_sha256`; non-matching pre-registered values remain in
`manifest_sha256` as expected model-manifest digests. New workers write the
file-list digest to `artifact_index_sha256`, while `manifest_sha256` is
reserved for the canonical Model Manifest v1 digest.
A different manifest requires a new model version; the existing manifest is
not replaced. Database triggers also reject direct mutation of an attached
manifest digest or payload. Artifact identity/provenance fields and purge
tombstones are protected by database triggers; only retention metadata,
verification timestamps and the initial tombstone transition remain mutable.

The manifest records model/version identity, model type, architecture and
configuration, tokenizer identity/version, source revision, license and usage restrictions,
optional training provenance, every artifact's filename/format/size/SHA-256,
and an optional evaluation report ID.

A version can become a candidate only when its manifest exactly describes
the live artifact set. It can become evaluated only when the manifest carries
an evaluation report ID. Approval rechecks every artifact's bytes and size.
Promoting a new version automatically marks the previously approved version
as deprecated and records `supersedes_version_id` for rollback traceability.

## Artifact provenance and retention

New direct uploads record the actor and request ID. Worker-produced artifacts
record the source job, job type, source version, and actor. Artifact metadata
also records the last successful integrity verification.

Retention classes are:

- `standard`: retained unless an operator schedules expiry.
- `temporary`: intended for explicitly time-bounded outputs.
- `archive`: retained for long-term reference.

`retention_until` is the explicit expiry timestamp. `legal_hold`
blocks deletion regardless of expiry. The admin-only retention API records
policy changes in the append-only audit log. Purging is permitted only after
expiry (or an explicitly expired timestamp), without legal hold, and only for
a draft version without a manifest or an archived version. Purged database
records remain as tombstones with `deleted_at`; their object
bytes are removed from storage.

The API re-hashes the object before serving a download. A mismatch fails closed
and is not returned to the client.

## Backup and restore procedure

1. Stop Quant Server workers and prevent new registry writes.
2. Take a consistent PostgreSQL backup including `model_catalog`,
   `model_versions`, `artifacts`,
   `model_jobs`, and `audit_events`.
3. Snapshot the S3 bucket or object-storage namespace using the provider's
   versioning/backup controls. Production operators should enable bucket
   versioning and a retention policy appropriate to their recovery objectives;
   application metadata alone does not provide object-store WORM protection.
4. Record the database backup identifier, object-store snapshot/version IDs,
   UTC timestamp, and deployment revision together.
5. Restore the database and object storage to a recovery environment before
   production cutover.
6. Run Alembic migrations to the expected head.
7. Run the read-only artifact inventory verifier:

   `python -m rusamodels.recovery`

   It verifies canonical model-manifest digests and their artifact inventories,
   then streams every non-purged registry artifact. It exits non-zero if a
   manifest is invalid, an inventory differs, or an object is missing or its
   byte count/SHA-256 differs from the registry.
8. Confirm model/version lifecycle state and promotion chain, then run the
   serving smoke tests before reopening writes and inference traffic.

Never restore database metadata without its corresponding artifact objects.
A missing or mismatched object must remain unavailable; do not silently
substitute a different artifact at the same model version.

## Operational limits

- Local filesystem storage is a development/edge adapter, not a production
  backup system.
- S3 bucket versioning, Object Lock/WORM, cross-region replication, and
  backup schedules must be configured and tested by the storage operator.
- CI exercises a SQLite plus local-filesystem snapshot/restore path.
  This is a recovery-verifier test, not evidence of a live PostgreSQL/S3
  production restore exercise.
- This runbook does not claim that a live production backup or restore
  exercise has already been performed.
