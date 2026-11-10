import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rusamodels.models import ModelCatalog, ModelJob, ModelVersion
from rusamodels.storage import LocalArtifactStorage
from services.quant_server import worker


def _download_records(api_client):
    db = api_client.app.state.sessions()
    model = ModelCatalog(
        name="Pinned Hub test",
        publisher="Qwen",
        license_id="apache-2.0",
        source_uri="https://huggingface.co/Qwen/demo",
        created_by="test-manager",
    )
    db.add(model)
    db.flush()
    version = ModelVersion(
        model_id=model.id,
        version="hf-main",
        source_revision="main",
        precision="SOURCE",
        status="queued",
        created_by="test-manager",
    )
    db.add(version)
    db.flush()
    job = ModelJob(
        model_id=model.id,
        output_version_id=version.id,
        job_type="download",
        parameters={
            "repo_id": "Qwen/demo",
            "revision": "main",
            "license_accepted": True,
        },
        status="running",
        progress_percent=0,
        created_by="test-manager",
        created_by_role="manager",
    )
    db.add(job)
    db.commit()
    return db, model, version, job


def test_hub_download_pins_resolved_commit_before_snapshot(
    api_client, tmp_path, monkeypatch
):
    db, _model, version, job = _download_records(api_client)
    commit = "a" * 40
    snapshot_calls = []

    class FakeHub:
        def model_info(self, repo_id, revision, files_metadata):
            assert repo_id == "Qwen/demo"
            assert revision == "main"
            assert files_metadata is True
            return SimpleNamespace(
                sha=commit,
                card_data=SimpleNamespace(license="apache-2.0"),
                siblings=[SimpleNamespace(size=64), SimpleNamespace(size=32)],
            )

    def fake_snapshot_download(repo_id, revision, token, local_dir):
        snapshot_calls.append(revision)
        target = Path(local_dir)
        target.mkdir()
        (target / "config.json").write_text(json.dumps({
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
        }))
        (target / "model.safetensors").write_bytes(b"verified-test-weights")
        return str(target)

    monkeypatch.setattr(worker, "HfApi", lambda token=None: FakeHub())
    monkeypatch.setattr(worker, "snapshot_download", fake_snapshot_download)
    settings = SimpleNamespace(hf_token=None, max_upload_bytes=1024 * 1024)
    storage = LocalArtifactStorage(tmp_path / "objects")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    artifacts = worker._download_job(db, job, settings, storage, work_dir)

    assert snapshot_calls == [commit]
    assert version.source_revision == commit
    assert job.parameters["resolved_revision"] == commit
    assert job.parameters["upstream_license"] == "apache-2.0"
    assert artifacts
    assert all(storage.read_bytes(item.storage_key) for item in artifacts)
    db.close()


def test_hub_download_rejects_missing_upstream_license(
    api_client, tmp_path, monkeypatch
):
    db, _model, _version, job = _download_records(api_client)

    class FakeHub:
        def model_info(self, repo_id, **kwargs):
            return SimpleNamespace(
                sha="b" * 40,
                card_data=SimpleNamespace(license=None),
                siblings=[],
            )

    monkeypatch.setattr(worker, "HfApi", lambda token=None: FakeHub())
    settings = SimpleNamespace(hf_token=None, max_upload_bytes=1024 * 1024)
    storage = LocalArtifactStorage(tmp_path / "objects")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(RuntimeError, match="does not declare a license"):
        worker._download_job(db, job, settings, storage, work_dir)
    db.close()


def test_hub_download_rejects_non_commit_revision(
    api_client, tmp_path, monkeypatch
):
    db, _model, _version, job = _download_records(api_client)

    class FakeHub:
        def model_info(self, repo_id, **kwargs):
            return SimpleNamespace(
                sha="main",
                card_data=SimpleNamespace(license="apache-2.0"),
                siblings=[],
            )

    monkeypatch.setattr(worker, "HfApi", lambda token=None: FakeHub())
    settings = SimpleNamespace(hf_token=None, max_upload_bytes=1024 * 1024)
    storage = LocalArtifactStorage(tmp_path / "objects")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(RuntimeError, match="immutable commit revision"):
        worker._download_job(db, job, settings, storage, work_dir)
    db.close()
