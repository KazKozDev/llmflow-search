import pytest


@pytest.fixture(autouse=True)
def _isolate_cwd_relative_paths(tmp_path, monkeypatch):
    """Tests must never write into the real project tree.

    Several config defaults (output.reports_dir, the smoke-check database, ...) are
    resolved relative to the working directory rather than passed explicitly, so a
    test using default config would otherwise drop real files into ./reports or
    ./data every time the suite runs.
    """
    monkeypatch.chdir(tmp_path)
