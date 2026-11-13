import pytest
from pydantic import ValidationError

from rusamodels.manifests import (
    ManifestTraining,
    ModelManifest,
    canonical_manifest_sha256,
)


def sample_manifest():
    return {
        "schema_version": "1.0",
        "model_id": "rusa-model",
        "model_version": "0.1.0",
        "model_type": "causal_lm",
        "architecture": "decoder_only_transformer",
        "tokenizer": {"id": "rusa-tokenizer", "version": "v1"},
        "source": {
            "repository": "StrategyG/RusaModels",
            "revision": "a" * 40,
            "license": "apache-2.0",
        },
        "artifacts": [{
            "name": "model.gguf",
            "format": "gguf",
            "quantization": "Q4_K_M",
            "size_bytes": 128,
            "sha256": "b" * 64,
        }],
    }


def test_manifest_digest_is_canonical_and_stable():
    first = ModelManifest.model_validate(sample_manifest())
    second_payload = dict(reversed(list(sample_manifest().items())))
    second = ModelManifest.model_validate(second_payload)
    assert canonical_manifest_sha256(first) == canonical_manifest_sha256(second)


def test_manifest_rejects_unknown_fields_and_duplicate_artifact_names():
    payload = sample_manifest()
    payload["unreviewed_runtime_flag"] = True
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(payload)

    payload = sample_manifest()
    payload["artifacts"] = payload["artifacts"] * 2
    with pytest.raises(ValidationError, match="unique"):
        ModelManifest.model_validate(payload)


def test_manifest_requires_sha256_for_artifacts():
    payload = sample_manifest()
    payload["artifacts"][0]["sha256"] = "not-a-digest"
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(payload)


def test_manifest_rejects_non_finite_architecture_values():
    payload = sample_manifest()
    payload["architecture_config"] = {"invalid": float("nan")}
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(payload)



def test_manifest_requires_full_immutable_commit_revisions():
    payload = sample_manifest()
    payload["source"]["revision"] = "a" * 12
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(payload)

    with pytest.raises(ValidationError):
        ManifestTraining(
            dataset_release="dataset-v1",
            config_digest="b" * 64,
            code_revision="c" * 12,
        )
