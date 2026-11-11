from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _workflow(path: str):
    content = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    return content


def test_production_image_workflow_is_tagged_signed_and_attested():
    workflow = _workflow(".github/workflows/publish-images.yml")
    triggers = workflow.get("on", workflow.get(True, {}))
    assert triggers["push"]["tags"] == ["v*.*.*"]
    assert workflow["permissions"]["packages"] == "write"
    assert workflow["permissions"]["id-token"] == "write"
    assert workflow["permissions"]["attestations"] == "write"

    steps = workflow["jobs"]["publish"]["steps"]
    action_steps = [step for step in steps if step.get("uses")]
    for step in action_steps:
        reference = step["uses"].rsplit("@", maxsplit=1)[-1].split("#", maxsplit=1)[0].strip()
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)

    build_steps = [step for step in steps if step.get("uses", "").startswith("docker/build-push-action@")]
    assert len(build_steps) == 2
    assert all(step["with"]["push"] is True for step in build_steps)
    assert all(step["with"]["sbom"] is True for step in build_steps)
    assert all(step["with"]["provenance"] == "mode=max" for step in build_steps)
    assert any("cosign sign" in step.get("run", "") for step in steps)
    assert sum("attest-build-provenance@" in step.get("uses", "") for step in steps) == 2
    assert any("git merge-base --is-ancestor" in step.get("run", "") for step in workflow["jobs"]["verify-release"]["steps"])


def test_nginx_ingress_limits_inference_and_restricts_management_api():
    config = (ROOT / "deploy" / "nginx" / "rusamodels.conf.example").read_text(encoding="utf-8")
    assert "limit_req_zone $binary_remote_addr zone=rusa_inference_per_ip" in config
    assert "limit_req zone=rusa_inference_per_ip" in config
    assert "client_max_body_size 16m" in config
    assert "proxy_buffering off" in config
    assert "server_name admin.models.example.com" in config
    assert "allow 10.0.0.0/8" in config
    assert "deny all" in config
    assert "127.0.0.1:8001" in config
