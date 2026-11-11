# RusaModels Evaluation & Benchmark Engine

## Purpose

The benchmark runner evaluates a model exposed through an OpenAI-compatible chat-completions endpoint. It is independent of the RusaAi application and does not orchestrate agents or tools.

## Frozen benchmark suite

A suite is a UTF-8 JSONL file accompanied by a sibling manifest named by appending `.manifest.json` to the suite filename.

Each JSONL record follows `schemas/llm/benchmark-suite-record.schema.json`:

~~~json
{
  "id": "indonesian-greeting-001",
  "category": "language_id",
  "messages": [
    {"role": "user", "content": "Jawab dalam bahasa Indonesia: selamat pagi"}
  ],
  "scorer": "contains",
  "expected": "selamat pagi"
}
~~~

Supported scorers:

- `exact_match`: case-folded and whitespace-normalized exact comparison.
- `contains`: expected string must appear in the prediction.
- `regex`: expected regular expression must match.
- `contains_all`: all expected strings must appear.
- `contains_any`: at least one expected string must appear.
- `json_valid`: prediction must parse as JSON.
- `manual`: stores a prediction for human review and does not contribute to automatic accuracy.

The frozen manifest must contain:

- `schema_version: 1.0.0`
- `suite_id` and `suite_version`
- `frozen: true`
- `file_name`, `file_sha256`, and `size_bytes`
- `record_count`
- sorted `license_ids`

The runner rejects modified suite files, duplicate record IDs, duplicate prompts, invalid roles, and unsupported scoring contracts. Freeze and version a suite before using it for model promotion.

## Run a benchmark

Local development endpoint:

~~~bash
python -m rusamodels.evaluation.benchmark \
  --suite data/benchmarks/rusaai-v1.0.0.jsonl \
  --base-url http://127.0.0.1:8100/openai/v1 \
  --model carbon-tiny \
  --output artifacts/evaluations/carbon-tiny-rusaai-v1.0.0.json \
  --runtime-profile cpu-local \
  --max-tokens 256
~~~

Production endpoint:

~~~bash
export RUSAMODELS_EVAL_API_KEY='...'
python -m rusamodels.evaluation.benchmark \
  --suite data/benchmarks/rusaai-v1.0.0.jsonl \
  --base-url https://models.example.com/openai/v1 \
  --model rusaai-edge \
  --model-artifact-sha256 <64-character-checkpoint-digest> \
  --runtime-profile cpu-production \
  --min-accuracy 0.90 \
  --output artifacts/evaluations/rusaai-edge-v1.0.0.json
~~~

The API key is read from the environment and is never written to the report. HTTPS is required for non-local endpoints. Reports must be written outside the immutable benchmark-suite directory.

## Report contents

Each report records:

- Frozen suite ID/version, suite digest, manifest digest, and declared licenses.
- Model ID, optional artifact SHA-256, and runtime profile.
- Endpoint origin, timestamps, and scoring configuration.
- Per-record prompt hash, prediction, expected answer, score, latency, finish reason, token usage, and request error.
- Aggregate accuracy by category and scorer.
- Mean, p50, and p95 request latency.
- Total prompt and completion tokens.
- Optional minimum-accuracy release gate.

Raw prompts are not copied into the report. Predictions and expected answers are included for review; protect evaluation reports as potentially sensitive artifacts.

## Limitations and release policy

This engine provides deterministic request settings and simple objective scorers. It does not provide semantic judging, factuality verification, safety certification, benchmark contamination detection, or human evaluation. Regex/contains scores are not a substitute for expert review.

A model release must also include a separate safety suite, regression suite, human review where needed, dataset contamination checks, quantization comparison, and runtime benchmarks on named hardware. Do not promote a model solely because this benchmark runner completes successfully.
