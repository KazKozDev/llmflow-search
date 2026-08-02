from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_agent_script_entrypoint_can_import_project_modules():
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "http://127.0.0.1:1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-m", "llmflow_search"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "ModuleNotFoundError" not in output
    assert "tools_data" not in output
    assert "ollama not reachable" in output
