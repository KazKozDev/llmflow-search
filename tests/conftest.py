import httpx
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


@pytest.fixture
def mock_transport(monkeypatch):
    """Redirects every httpx.AsyncClient created during the test to a MockTransport.

    Production code builds its own httpx.AsyncClient(...) internally (no way to inject a
    transport from the outside), so this patches the httpx.AsyncClient constructor itself
    to attach the given handler's transport, in place of the real network fetch.
    """
    real_async_client = httpx.AsyncClient

    def _install(handler) -> None:
        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return _install
