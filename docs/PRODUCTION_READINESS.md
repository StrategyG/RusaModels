# RusaModels Production Readiness Evidence Gate

## Purpose

The production evidence gate evaluates one exact RusaModels release candidate. It checks that the release commit, container digest, deployment configuration, approved model artifact, evaluation reports, runtime measurements, soak test, security scan, backup/restore, rollback, and deployment controls refer to the same release.

The gate does not create or independently retrieve this evidence. Operators must supply accurate evidence from GitHub Actions, security scanners, the model registry, the deployed runtime, and infrastructure operations.

## Evidence bundle

Create one UTF-8 JSON file following `schemas/production/production-evidence.schema.json`. It must include these sections:

| Section | Required evidence |
|---|---|
| `release` | Git commit, immutable container digest, deployment-config digest, model ID/version/artifact digest |
| `ci` | Successful required checks on the exact release commit and CI run URL |
| `security` | Passed security scan, zero critical/high/secret findings, report URL |
| `model` | Approved lifecycle state, model manifest digest, exact artifact digest |
| `evaluation` | Benchmark, safety, and regression gates passed for the exact model artifact |
| `runtime` | CPU-only health check, named hardware, RAM reserve, TTFT, tokens/second, zero OOM |
| `soak` | At least 24 hours, at least 100 requests, error rate at most 0.1%, zero crashes/OOM |
| `backup_restore` | Verified database and object-storage backups, successful restore, backup no older than 24 hours |
| `rollback` | Successful software and model rollback tests |
| `deployment` | HTTPS, PostgreSQL `sslmode=verify-full`, secret manager, external object storage, persistence, monitoring, alerting |

The model artifact SHA-256 must match across the release, registry, evaluation report, and runtime evidence. The CI commit must equal the release Git commit.

## Run the gate

~~~bash
python -m rusamodels.production_readiness \
  --evidence artifacts/release/production-evidence.json \
  --output artifacts/release/production-readiness-report.json
~~~

Exit codes:
- `0`: every gate passed and the evidence bundle is internally consistent.
- `1`: the report was generated, but one or more production gates failed.
- `2`: the evidence file or command is invalid.

Thresholds can be adjusted explicitly:

~~~bash
python -m rusamodels.production_readiness \
  --evidence artifacts/release/production-evidence.json \
  --output artifacts/release/production-readiness-report.json \
  --min-soak-hours 24 \
  --max-soak-error-rate 0.001 \
  --min-soak-requests 100 \
  --max-backup-age-hours 24
~~~

## Interpretation

A `production_ready` report means the supplied evidence fields satisfied the repository's configured release gates. It is not an independent audit of GitHub, a security scanner, a cloud account, a backup system, or a physical device. The gate validates declared values and cross-artifact consistency; operators must retain the linked raw reports and ensure the evidence is truthful.

Do not fabricate a passing bundle or fill it with placeholders. Until real target-hardware runtime tests, 24-hour soak evidence, verified backup/restore, rollback, and production deployment evidence exist, RusaModels must remain **not production-ready**.

The default thresholds are initial repository policy and may be tightened by the release owner. A change to thresholds must be reviewed and recorded; do not lower them merely to make a candidate pass.
