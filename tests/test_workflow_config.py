"""Static checks for paths referenced by GitHub Actions workflows."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_workflow_install_and_cache_dependency_paths_exist():
    for workflow in WORKFLOWS.glob("*.yml"):
        content = workflow.read_text(encoding="utf-8")
        referenced_paths = re.findall(
            r"\bpip\s+install\s+-r\s+([^\s\\]+)", content
        )
        referenced_paths.extend(
            re.findall(r"(?m)^\s*cache-dependency-path:\s*([^\s#]+)", content)
        )
        for relative_path in referenced_paths:
            assert (ROOT / relative_path).is_file(), (
                f"{workflow.relative_to(ROOT)} references missing dependency file "
                f"{relative_path!r}"
            )
