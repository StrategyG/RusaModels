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



def test_claim_job_reclaims_expired_lease_with_new_epoch(api_client):
    from datetime import UTC, datetime, timedelta

    db, _model, _version, job = _download_records(api_client)
    job.status = "queued"
    db.commit()
    db.close()
    sessions = api_client.app.state.sessions

    first_claim = worker.claim_job(sessions, "worker-a", lease_seconds=30)
    assert first_claim == job.id
    db = sessions()
    claimed = db.get(ModelJob, job.id)
    assert claimed.status == "running"
    assert claimed.attempt_count == 1
    assert claimed.lease_epoch == 1
    assert claimed.lease_owner == "worker-a"
    claimed.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()
    db.close()

    second_claim = worker.claim_job(sessions, "worker-b", lease_seconds=30)
    assert second_claim == job.id
    db = sessions()
    reclaimed = db.get(ModelJob, job.id)
    assert reclaimed.attempt_count == 2
    assert reclaimed.lease_epoch == 2
    assert reclaimed.lease_owner == "worker-b"
    db.close()


def test_transient_worker_failure_is_requeued_with_bounded_retry(
    api_client, tmp_path, monkeypatch
):
    db, _model, version, job = _download_records(api_client)
    job.status = "queued"
    db.commit()
    db.close()
    sessions = api_client.app.state.sessions
    worker_id = "worker-retry-test"
    job_id = worker.claim_job(sessions, worker_id)
    assert job_id == job.id

    def fail_download(*args, **kwargs):
        raise TimeoutError("simulated upstream timeout")

    monkeypatch.setattr(worker, "_download_job", fail_download)
    settings = SimpleNamespace(hf_token=None, max_upload_bytes=1024 * 1024)
    storage = LocalArtifactStorage(tmp_path / "objects")
    worker.process_job(job_id, sessions, settings, storage, worker_id)

    db = sessions()
    retried = db.get(ModelJob, job.id)
    refreshed_version = db.get(ModelVersion, version.id)
    assert retried.status == "queued"
    assert retried.attempt_count == 1
    assert retried.max_attempts == 3
    assert retried.next_attempt_at is not None
    assert retried.lease_owner is None
    assert refreshed_version.status == "queued"
    db.close()


def test_expired_job_at_attempt_limit_is_failed(api_client):
    from datetime import UTC, datetime, timedelta

    db, _model, version, job = _download_records(api_client)
    job.status = "running"
    job.attempt_count = 3
    job.max_attempts = 3
    job.lease_owner = "dead-worker"
    job.lease_epoch = 3
    job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()
    db.close()

    assert worker.claim_job(api_client.app.state.sessions, "replacement-worker") is None
    db = api_client.app.state.sessions()
    failed = db.get(ModelJob, job.id)
    failed_version = db.get(ModelVersion, version.id)
    assert failed.status == "failed"
    assert failed.error_code == "WorkerLeaseExpired"
    assert failed.lease_owner is None
    assert failed_version.status == "failed"
    db.close()


def test_hub_retry_reuses_previously_pinned_commit(
    api_client, tmp_path, monkeypatch
):
    db, _model, version, job = _download_records(api_client)
    commit = "c" * 40
    job.parameters = {**job.parameters, "resolved_revision": commit}
    db.commit()
    requested_revisions = []

    class FakeHub:
        def model_info(self, repo_id, revision, files_metadata):
            requested_revisions.append(revision)
            return SimpleNamespace(
                sha=commit,
                card_data=SimpleNamespace(license="apache-2.0"),
                siblings=[SimpleNamespace(size=20)],
            )

    def fake_snapshot_download(repo_id, revision, token, local_dir):
        requested_revisions.append(revision)
        target = Path(local_dir)
        target.mkdir()
        (target / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        (target / "weights.safetensors").write_bytes(b"retry-safe-weights")
        return str(target)

    monkeypatch.setattr(worker, "HfApi", lambda token=None: FakeHub())
    monkeypatch.setattr(worker, "snapshot_download", fake_snapshot_download)
    settings = SimpleNamespace(hf_token=None, max_upload_bytes=1024 * 1024)
    storage = LocalArtifactStorage(tmp_path / "objects")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    worker._download_job(db, job, settings, storage, work_dir)

    assert requested_revisions == [commit, commit]
    assert version.source_revision == commit
    db.close()
