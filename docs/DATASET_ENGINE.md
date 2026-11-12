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

The pipeline does not infer whether a license is legally compatible, identify all PII, classify language, perform automated toxicity classification, or tokenize/pack sequences. The initial SimHash quality analyzer, external frozen-evaluation contamination check, and human-review gate provide bounded evidence but do not replace semantic review, full safety classification, reviewer identity verification, or signed release attestations.

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


## Export a verified split for byte-level CPU training

Carbon-Tiny consumes UTF-8 text corpora, while the Dataset Engine releases JSONL. Export each split independently; never combine train and validation/test splits.

~~~bash
python scripts/export_dataset_text.py \
  --release artifacts/datasets/rusaai-agent-sft-curated/0.1.0 \
  --split train \
  --output artifacts/corpora/rusaai-agent-sft-curated-0.1.0-train.txt

python scripts/export_dataset_text.py \
  --release artifacts/datasets/rusaai-agent-sft-curated/0.1.0 \
  --split validation \
  --output artifacts/corpora/rusaai-agent-sft-curated-0.1.0-validation.txt
~~~

Each text file has a sibling manifest whose name appends .manifest.json to the output filename. The sidecar contract is defined by `schemas/llm/dataset-text-export-manifest.schema.json`. It records the source dataset ID/version, split, declared license IDs, source release-manifest digest, source split digest, output digest, record count, encoding, and byte-tokenizer identifier. The exporter omits record IDs, group IDs, and metadata from the text corpus. The output and sidecar are written outside the immutable dataset release; existing outputs are not overwritten.

Use the train export as the Carbon trainer's training corpus and the validation export as its held-out evaluation corpus. Copy the dataset identity and reviewed license declaration into the training config. The trainer independently hashes both text files and rejects identical train/evaluation corpora.

## Human review and publication gate

A dataset release is immutable after it is built. Human review evidence is therefore kept in a separate append-only JSONL ledger. Every review event must include the exact `release_manifest_sha256`; the gate rejects events created for another dataset release, even when a pseudonymous record ID happens to be reused.

Each released record requires the latest approved event for all five review types:

- `technical`: quality score, language, task type, and evidence status.
- `rights`: explicit `rights_status=approved`.
- `security`: risk classification.
- `independent_qa`: independent quality score, risk classification, and evidence status.
- `publication_approval`: explicit `publication_authorized=true`.

The independent QA reviewer must differ from the technical reviewer. The publication approver must differ from both and from the rights/security reviewers. Publication approval must be timestamped after the latest technical, rights, security, and independent-QA events; a later review invalidates an earlier publication approval until a new publication decision is recorded. Latest events supersede older events; a later rejection or revision request blocks publication. The default gate requires quality scores of at least 4/5 and a maximum risk class of medium.

Reviewer identifiers must be pseudonymous. Do not put names, email addresses, free-text reviewer notes, or sensitive case details in the ledger. Keep identity mapping and detailed audit notes in an access-controlled review system.

The event contract is defined by `schemas/llm/dataset-review-record.schema.json`. A representative technical event is:

~~~json
{
  "review_id": "review-example-technical-001",
  "release_manifest_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "record_id": "record_0123456789abcdef01234567",
  "review_type": "technical",
  "reviewer_id": "reviewer_technical_01",
  "decision": "approved",
  "quality_score": 5,
  "risk_class": null,
  "rights_status": null,
  "evidence_status": "implementation_observation",
  "language": "id",
  "task_type": "technical_instruction",
  "publication_authorized": false,
  "reviewed_at": "2026-01-01T00:00:00Z"
}
~~~

Run the gate after review events are recorded:

~~~bash
python -m rusamodels.dataset_review \
  --release artifacts/datasets/rusaai-agent-sft-curated/0.1.0 \
  --ledger artifacts/reviews/rusaai-agent-sft-curated-0.1.0.jsonl \
  --report artifacts/reports/rusaai-agent-sft-curated-0.1.0-review.json
~~~

Exit code 0 means every record passed the configured review gate; exit code 1 means the report was generated but one or more records are blocked; exit code 2 means the release or ledger could not be evaluated. The report binds the review ledger digest to the exact dataset manifest digest, and each individual event is required to carry that same digest. This tool validates reviewer identifiers and separation rules; it does not authenticate real-world reviewer identity or cryptographically sign approvals. Production publication must add identity-provider verification, controlled authorization, and an externally managed signature/attestation.

## Frozen evaluation contamination gate

Keep the frozen release evaluation set separate from training, validation, and internal test sources. The frozen set must have a manifest with `purpose=frozen_evaluation`, `locked=true`, a pinned file name, SHA-256, byte size, record count, dataset ID/version, and timezone-aware creation timestamp. The manifest contract is defined by `schemas/llm/frozen-evaluation-manifest.schema.json`.

Example manifest shape (replace the identity, digest, size, and count with the actual frozen file values):

~~~json
{
  "schema_version": "1.0.0",
  "purpose": "frozen_evaluation",
  "dataset_id": "rusaai-frozen-eval",
  "dataset_version": "1.0.0",
  "locked": true,
  "file": "release-eval.jsonl",
  "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "size_bytes": 123456,
  "records": 1000,
  "created_at": "2026-01-01T00:00:00Z"
}
~~~

The `locked` field is an operator attestation; it does not make a filesystem or object-store object immutable. Store the evaluation file in access-controlled, versioned storage and protect the manifest with an external signature or approval record.

Compare all three dataset splits against the frozen evaluation set:

~~~bash
python -m rusamodels.dataset_contamination \
  --release artifacts/datasets/rusaai-agent-sft-curated/0.1.0 \
  --frozen-evaluation data/frozen_eval/release-eval.jsonl \
  --frozen-manifest data/frozen_eval/release-eval.manifest.json \
  --report artifacts/reports/rusaai-agent-sft-curated-0.1.0-contamination.jsonl
~~~

The gate checks exact normalized training-text matches and SimHash near-duplicates using the same byte-independent text rendering used by the text exporter. It writes only pseudonymous record IDs, split, match type, Hamming distance, and text hash; it does not copy evaluation prompt text into the report. Exit code 0 means no candidate match was found, 1 means one or more exact/near matches were found, and 2 means the scan could not be completed.

The SimHash threshold is bounded to 0–7 bits by the eight-band LSH candidate guarantee. Candidate retrieval is capped per dataset record to bound memory use; exceeding the cap fails the scan rather than silently truncating candidates. This is a lexical heuristic, not semantic equivalence detection; paraphrases may be missed. The frozen evaluation manifest is an integrity record, not a cryptographic signature. Keep the frozen set in access-controlled storage, restrict who can read it, and use an externally signed manifest for a production benchmark gate.

## Language identification and quality screening

Install the optional offline language detector:

~~~bash
python -m pip install -e ".[dataset]"
~~~

Analyze the verified release without modifying its immutable JSONL files:

~~~bash
python -m rusamodels.dataset_language \
  --release artifacts/datasets/rusaai-agent-sft-curated/0.1.0 \
  --report artifacts/reports/rusaai-agent-sft-curated-0.1.0-language.jsonl \
  --languages id,en \
  --allowed-languages id,en,mixed \
  --fail-on-rejected
~~~

The analyzer uses the preloaded Lingua language models for offline Indonesian/English identification. It reports language confidence, runner-up language, text length, word count, unique-word ratio, repeated-character ratio, and eligibility reasons for every pseudonymized record ID. The JSONL record contract is defined by `schemas/llm/dataset-language-quality-record.schema.json`; a summary sidecar contains split and language distributions.

The default screening policy requires at least 40 normalized characters, six words, 0.20 unique-word ratio, no more than 0.25 repeated-character ratio, and language confidence of at least 0.65. Tune these thresholds on a human-reviewed sample before using them as a release gate. Short or ambiguous text is labeled `und`; close Indonesian/English confidence values may be labeled `mixed`.

This tool produces eligibility evidence only. It does not delete or rewrite rejected records, identify semantic quality, classify toxicity, detect all unsafe content, or certify that a record is suitable for training. Use its report to drive a reviewed filtering decision, then build a new immutable dataset version. High-risk and unsafe content must remain quarantined through the human security-review gate.

## Pinned-tokenizer preprocessing and sequence packing

Install the optional Transformers tokenizer runtime:

~~~bash
python -m pip install -e ".[tokenization]"
~~~

Download or provision the approved tokenizer snapshot separately, verify its upstream source revision, and keep the files in a local directory. The packer loads only local files with remote code disabled; it does not download tokenizer assets or execute repository-provided custom tokenizer code.

Pack one split at a time:

~~~bash
python -m rusamodels.dataset_tokenization \
  --release artifacts/datasets/rusaai-agent-sft-curated/0.1.0 \
  --split train \
  --tokenizer-path artifacts/tokenizers/approved-tokenizer \
  --tokenizer-id organization/model-tokenizer \
  --tokenizer-revision 0123456789abcdef0123456789abcdef01234567 \
  --sequence-length 2048 \
  --output artifacts/tokenized/rusaai-agent-sft-curated-0.1.0-train.jsonl
~~~

Run the command separately for validation and test. The packer verifies the source release, hashes every local tokenizer asset by relative path and file digest, records the declared 40-character tokenizer revision, inserts EOS between records, and emits fixed-length JSONL sequences with input IDs, attention masks, labels, and pseudonymized source record IDs. The final partial sequence is padded; padded labels use -100 and the attention mask uses zero. The tokenized artifact manifest binds the exact source split, dataset release manifest, tokenizer identity/revision/content digest, sequence length, token counts, and output checksum.

The record contract is defined by `schemas/llm/tokenized-sequence.schema.json`; the artifact manifest is defined by `schemas/llm/dataset-tokenization-manifest.schema.json`. Training loaders must honor attention masks and ignore labels equal to -100. Do not combine train, validation, and test outputs.

Verify the packed output before handing it to a training job:

~~~bash
python -m rusamodels.dataset_tokenized_verification \
  --output artifacts/tokenized/rusaai-agent-sft-curated-0.1.0-train.jsonl \
  --release artifacts/datasets/rusaai-agent-sft-curated/0.1.0 \
  --tokenizer-path artifacts/tokenizers/approved-tokenizer
~~~

The verifier checks output and manifest checksums, sequence lengths, vocabulary bounds, padding placement, attention masks, labels, token counts, the exact source release/split, and (when supplied) the local tokenizer asset digest. It refuses tokenized outputs inside the immutable source release or tokenizer asset directory.

This packer is for plain-text causal-LM continuation/pretraining. It does not replace the Agent SFT tokenizer path: the existing SFT trainer applies the model chat template and masks prompt tokens so loss is computed on the intended assistant completion. Use the SFT workflow for supervised chat adaptation. The tokenizer revision is operator-supplied provenance; the local asset SHA-256 is the byte-level identity actually verified by this tool.

## Dataset inspection and sampling

Inspect split sizes, text-length distributions, task/language/role/risk labels, and deterministic content samples:

~~~bash
python -m rusamodels.dataset_inspect \
  --release artifacts/datasets/rusaai-agent-sft-curated/0.1.0 \
  --sample-per-split 3 \
  --seed 42 \
  --preview-characters 300
~~~

Sampling uses a bounded reservoir per split, so the tool does not load the entire corpus into memory. The seed makes the sample repeatable. Set `--sample-per-split 0` to emit statistics without text previews. Preview text is still dataset content; protect terminal logs and redirected output with the same access controls as the source release.

## Redaction and privacy limits

Email and common credential patterns are redacted by default. Phone-number redaction is disabled by default because numeric sequences can be legitimate model content. These regex rules are a defensive preprocessing layer, not a comprehensive PII detector or privacy certification.

Review the redaction counts and sample the output before training. For regulated or sensitive data, perform an additional approved privacy review and use an independently validated detector. Do not place raw sensitive datasets in Git.

## Relationship to the existing Agent SFT dataset

The current data/agent_sft/train.jsonl and data/agent_sft/eval.jsonl are already separate synthetic seed splits. Do not feed both files into one random split build: that would mix a held-out evaluation set into the generated training pool and invalidate the existing holdout. Use one approved raw/curated pool for a new release, or preserve the existing train/eval separation and verify it with the existing SFT dataset validator.

The dataset release pipeline preserves the existing SFT record shape (including id, agent_role, messages, and response), replaces source IDs with deterministic pseudonymous IDs, and writes only the pseudonymous group ID to the provenance sidecar.

## Remaining Phase 2 work

The following roadmap items are not implemented by this initial release builder:

- Capacity-tested near-duplicate detection at representative dataset scale (the initial SimHash/LSH checker is implemented; threshold calibration and large-corpus benchmarks remain).
- Offline Indonesian/English language identification, lexical-quality metrics, eligibility reports, and deterministic sample inspection are implemented; language coverage and thresholds still require human calibration on representative corpora.
- Semantic quality classifiers, toxicity/unsafe-content classifiers, and a reviewed filter-to-new-release workflow remain open.
- Human review ledger and publication gate are implemented as an initial workflow; integration with an authenticated annotation UI and signed reviewer attestations remains open.
- Frozen evaluation contamination gate is implemented for exact text and SimHash near-duplicates; semantic contamination detection and capacity calibration remain open.
- Pinned local Hugging Face tokenizer preprocessing and fixed-length causal-LM sequence packing are implemented. A production training loader, full-model tokenizer compatibility certification, and capacity benchmarking remain required.
- Dataset registry integration and registry-level promotion/retention.
- External object-storage upload/download for dataset releases.
- Large-scale load/performance testing on representative datasets.

Do not mark Phase 2 complete until the remaining required capabilities are implemented or explicitly deferred through an approved ADR, with documented trade-offs and tests.
