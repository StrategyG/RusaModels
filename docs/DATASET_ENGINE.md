# Dataset Engineering Engine

## Status

The JSONL dataset release pipeline is implemented as an initial repository capability. It is a reproducible, disk-backed release builder with provenance and integrity verification. It is **not** a complete data-curation or model-training data platform yet.

The pipeline currently provides:

- Strict UTF-8 JSONL ingestion (one JSON object per non-empty line).
- Duplicate JSON-key and non-standard numeric rejection.
- Unicode NFC and newline normalization.
- Configurable email and common-secret redaction; phone redaction is optional.
- Exact canonical-content deduplication.
- Stable record IDs when an input record does not provide an ID.
- Deterministic group-aware train/validation/test splitting.
- Per-source license declarations, file digests, and record-line provenance.
- Immutable dataset release directories.
- SHA-256 manifests and an independent release verifier.
- Disk-backed SQLite verification indexes for record IDs, fingerprints, provenance, and split-group isolation.
- SQLite-backed staging so the pipeline does not need to hold the full dataset in Python memory.

The pipeline does not infer whether a license is legally compatible, identify all PII, perform semantic/near-duplicate detection, classify language, perform quality scoring, annotate examples, or tokenize/pack sequences. Those capabilities remain future work and must not be implied by a successful release build.

## Input contract

Each source is a UTF-8 JSONL file. Every non-empty line must be a JSON object containing at least one non-empty text value. Existing record fields are preserved except for the optional top-level **group_id** field, which is used only to keep related records in one split and is recorded in the provenance sidecar.

- **id**: optional string, maximum 256 characters. If supplied, it is SHA-256 pseudonymized using the dataset ID as namespace. If absent, a stable ID is generated from normalized and redacted content.
- **group_id**: optional string, maximum 256 characters. Use this to keep messages from the same conversation/document/customer grouping in one split. It is pseudonymized before it is written to provenance. Without it, each unique content record is treated as its own group. If exact duplicate content is supplied with conflicting explicit group IDs, the release fails closed and requires the source grouping to be reconciled.
- All other JSON fields are retained and recursively normalized/redacted.

A source must have a unique **source_id** and a declared **license_id**. The pipeline records the declaration; it does not validate the legal meaning or grant rights under that license. Complete the legal/source review before releasing a dataset.

## Configuration

Create a JSON config file. Paths are resolved relative to the config file.

~~~json
{
  "dataset_id": "rusaai-agent-sft-curated",
  "version": "0.1.0",
  "output_root": "../../artifacts/datasets",
  "seed": 42,
  "validation_fraction": 0.1,
  "test_fraction": 0.1,
  "max_record_bytes": 2097152,
  "redaction": {
    "emails": true,
    "secrets": true,
    "phone_numbers": false
  },
  "sources": [
    {
      "source_id": "approved-corpus",
      "path": "../../data/raw/approved-corpus.jsonl",
      "license_id": "LicenseRef-REPLACE-WITH-APPROVED-LICENSE",
      "source_uri": "https://example.invalid/dataset"
    }
  ]
}
~~~

Replace the example source, URI, and license identifier with reviewed values. Do not put credentials, access tokens, or signed URLs in **source_uri**; the loader rejects URI user information, query strings, and fragments. Avoid absolute workstation paths in committed configuration.

## Build a release

Install the repository's base dependencies and run:

~~~bash
python -m rusamodels.data_pipeline --config configs/datasets/agent-sft-release.json
~~~

A successful build publishes:

~~~text
artifacts/datasets/<dataset_id>/<version>/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── provenance.jsonl
├── manifest.json
└── manifest.sha256
~~~

The release directory is immutable. Rebuilding the same dataset ID and version fails; increment the version to publish a new release. Failed builds remove the staging directory and release lock.

The splitter requires at least three distinct groups so all three splits can be populated. It orders groups by a seeded SHA-256 key. This is deterministic, not stratified; inspect the resulting distribution before using a release for training.

## Verify a release

~~~bash
python -m rusamodels.dataset_verification --release artifacts/datasets/rusaai-agent-sft-curated/0.1.0
~~~

## Near-duplicate and split-leakage analysis

Run the SimHash quality check after building and verifying a dataset release:

~~~bash
python -m rusamodels.dataset_quality \
  --release artifacts/datasets/rusaai-agent-sft-curated/0.1.0 \
  --output artifacts/reports/rusaai-agent-sft-curated-0.1.0.near-duplicates.jsonl
~~~

The analyzer uses 64-bit word-trigram SimHash and eight 8-bit LSH bands. It verifies each candidate using exact Hamming distance (default maximum seven bits), excludes records from the same pseudonymized group, and writes a report outside the immutable release directory. The CLI exits with code 1 when a near-duplicate pair crosses train/validation/test splits, making it usable as a release gate. Same-split pairs are also reported for review but do not by themselves trigger the cross-split failure exit code.

This is a lexical similarity heuristic, not semantic equivalence detection. It can miss paraphrases and can flag legitimate short/template-like examples. Review the report, calibrate the threshold on representative corpora, and run capacity tests before applying it to very large datasets. The LSH candidate guarantee is bounded to the configured 0-7 Hamming-distance range; thresholds above seven are rejected.


The verifier checks:

- Manifest digest and directory identity.
- Exact expected file set (no unmanifested files).
- SHA-256 and byte size for each split and provenance file.
- JSONL record IDs and exact-content uniqueness across splits.
- Source IDs, declared licenses, source digests, and source-line provenance.
- Record-to-source mappings and record fingerprints.
- Group isolation across train, validation, and test.
- Source, raw, unique, duplicate, and split record counts.

The verifier cannot recalculate source-file digests unless the original source files are separately available. Retain source snapshots or immutable source references according to the project's data retention policy.

## Redaction and privacy limits

Email and common credential patterns are redacted by default. Phone-number redaction is disabled by default because numeric sequences can be legitimate model content. These regex rules are a defensive preprocessing layer, not a comprehensive PII detector or privacy certification.

Review the redaction counts and sample the output before training. For regulated or sensitive data, perform an additional approved privacy review and use an independently validated detector. Do not place raw sensitive datasets in Git.

## Relationship to the existing Agent SFT dataset

The current data/agent_sft/train.jsonl and data/agent_sft/eval.jsonl are already separate synthetic seed splits. Do not feed both files into one random split build: that would mix a held-out evaluation set into the generated training pool and invalidate the existing holdout. Use one approved raw/curated pool for a new release, or preserve the existing train/eval separation and verify it with the existing SFT dataset validator.

The dataset release pipeline preserves the existing SFT record shape (including id, agent_role, messages, and response), replaces source IDs with deterministic pseudonymous IDs, and writes only the pseudonymous group ID to the provenance sidecar.

## Remaining Phase 2 work

The following roadmap items are not implemented by this initial release builder:

- Capacity-tested near-duplicate detection at representative dataset scale (the initial SimHash/LSH checker is implemented; threshold calibration and large-corpus benchmarks remain).
- Language identification and language-aware quality metrics.
- Configurable quality classifiers and filtering reports.
- Human annotation/review workflow and reviewer provenance.
- Evaluation contamination checks against frozen benchmark sets (the near-duplicate checker currently checks only within one released dataset, not against external frozen benchmark corpora).
- Tokenizer-aware preprocessing and sequence packing.
- Dataset registry integration and registry-level promotion/retention.
- External object-storage upload/download for dataset releases.
- Large-scale load/performance testing on representative datasets.

Do not mark Phase 2 complete until the remaining required capabilities are implemented or explicitly deferred through an approved ADR, with documented trade-offs and tests.
