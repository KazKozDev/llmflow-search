from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from llmflow_search import _ensure_default_browser

SRC = Path(__file__).resolve().parents[1] / "src"


def test_agent_script_entrypoint_can_import_project_modules():
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "http://127.0.0.1:1"
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
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


def test_default_server_bootstraps_chromium(monkeypatch):
    monkeypatch.delenv("LLMFLOW_SEARCH_MCP_CMD", raising=False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    _ensure_default_browser()

    assert calls[0][0] == [
        sys.executable,
        "-m",
        "playwright",
        "install",
        "chromium",
    ]
    assert calls[0][1]["check"] is False


def test_custom_server_skips_chromium_bootstrap(monkeypatch):
    monkeypatch.setenv("LLMFLOW_SEARCH_MCP_CMD", "python scripts/stub_mcp_server.py")

    def fail_run(*args, **kwargs):
        raise AssertionError("browser bootstrap should be skipped")

    monkeypatch.setattr(subprocess, "run", fail_run)

    _ensure_default_browser()
