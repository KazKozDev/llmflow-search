"""Route tests for the FastAPI web server."""

import importlib
import sys
import types
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from core.background_jobs import Job, JobQueue, JobStatus
from core.event_bus import Event


def _install_web_server_stubs():
    """Register lightweight stand-ins for heavy web server imports."""
    agent_factory_module = types.ModuleType("core.agent_factory")

    class StubAgentFactory:
        @staticmethod
        async def initialize(_config):
            return None

        @staticmethod
        async def shutdown():
            return None

        @staticmethod
        async def create_agent(**_kwargs):
            return None

        @staticmethod
        def get_metrics():
            return {}

    agent_factory_module.AgentFactory = StubAgentFactory

    monitoring_module = types.ModuleType("core.monitoring")

    class StubMetrics:
        def get_stats(self):
            return {}

    monitoring_module.get_metrics = lambda: StubMetrics()

    background_jobs_module = types.ModuleType("core.background_jobs")
    background_jobs_module.Job = Job
    background_jobs_module.JobQueue = JobQueue
    background_jobs_module.JobStatus = JobStatus
    background_jobs_module.get_job_queue = lambda: JobQueue()

    simple_class_modules = {
        "core.memory_module": "MemoryModule",
        "core.llm_service": "LLMService",
        "core.planning_module": "PlanningModule",
        "core.tools_module": "ToolsModule",
        "core.report_generator": "ReportGenerator",
        "core.agent_core": "AgentCore",
        "core.search_intent_analyzer": "SearchIntentAnalyzer",
    }

    sys.modules["core.agent_factory"] = agent_factory_module
    sys.modules["core.monitoring"] = monitoring_module
    sys.modules["core.background_jobs"] = background_jobs_module

    event_bus_module = types.ModuleType("core.event_bus")
    event_bus_module.Event = Event
    sys.modules["core.event_bus"] = event_bus_module

    for module_name, class_name in simple_class_modules.items():
        module = types.ModuleType(module_name)
        setattr(module, class_name, type(class_name, (), {}))
        sys.modules[module_name] = module


_install_web_server_stubs()

web_server = importlib.import_module("web_server")


@asynccontextmanager
async def no_lifespan(_app):
    """Disable startup and shutdown side effects during tests."""
    yield


class WebServerRouteTests(unittest.TestCase):
    """Verify API endpoints exposed by the FastAPI app."""

    def setUp(self):
        web_server.app.router.lifespan_context = no_lifespan
        web_server.active_sessions.clear()

    def test_root_serves_html(self):
        """The root route should serve the main HTML page."""
        with TestClient(web_server.app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("LLMFlow Search Agent", response.text)

    def test_standard_search_creates_session(self):
        """Standard mode should create an in-memory search session."""
        with TestClient(web_server.app) as client:
            response = client.post(
                "/api/search",
                json={
                    "query": "test standard search",
                    "max_iterations": 4,
                    "mode": "standard",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "initialized")
        self.assertTrue(payload["session_id"])
        self.assertIn(payload["session_id"], web_server.active_sessions)

    def test_deep_search_queues_background_job(self):
        """Deep mode should enqueue a background job and return its id."""
        fake_queue = MagicMock()
        fake_queue.submit = AsyncMock(return_value="job-123")

        fake_worker_module = types.SimpleNamespace(
            run_deep_search_worker=MagicMock()
        )

        with patch.object(
            web_server,
            "get_job_queue",
            return_value=fake_queue,
        ):
            with patch.dict(
                sys.modules,
                {"core.deep_search_worker": fake_worker_module},
            ):
                with TestClient(web_server.app) as client:
                    response = client.post(
                        "/api/search",
                        json={
                            "query": "test deep search",
                            "max_iterations": 5,
                            "mode": "deep",
                        },
                    )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["job_id"], "job-123")
        fake_queue.submit.assert_called_once()

    def test_list_tools_returns_expected_entries(self):
        """The tools endpoint should expose the configured tool list."""
        with TestClient(web_server.app) as client:
            response = client.get("/api/tools")

        self.assertEqual(response.status_code, 200)
        tools = response.json()["tools"]
        names = {item["name"] for item in tools}
        self.assertIn("search_duckduckgo", names)
        self.assertIn("search_arxiv", names)
        self.assertIn("search_wayback", names)

    def test_metrics_endpoint_merges_system_and_llm_metrics(self):
        """Metrics route should combine app and LLM factory telemetry."""
        fake_metrics = MagicMock()
        fake_metrics.get_stats.return_value = {"requests": 7}

        with patch.object(
            web_server,
            "get_metrics",
            return_value=fake_metrics,
        ):
            with patch.object(
                web_server.AgentFactory,
                "get_metrics",
                return_value={"latency_ms": 12},
            ):
                with TestClient(web_server.app) as client:
                    response = client.get("/api/metrics")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["system"], {"requests": 7})
        self.assertEqual(payload["llm"], {"latency_ms": 12})
        self.assertIn("timestamp", payload)

    def test_missing_job_returns_404(self):
        """The job status route should report missing jobs with 404."""
        fake_queue = MagicMock()
        fake_queue.get_job.return_value = None

        with patch.object(
            web_server,
            "get_job_queue",
            return_value=fake_queue,
        ):
            with TestClient(web_server.app) as client:
                response = client.get("/api/jobs/missing-job")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Job not found")

    def test_websocket_rejects_unknown_session(self):
        """Unknown session ids should yield an error event and close."""
        with TestClient(web_server.app) as client:
            with client.websocket_connect(
                "/ws/search/unknown-session"
            ) as websocket:
                message = websocket.receive_json()

        self.assertEqual(message["type"], "error")
        self.assertEqual(message["message"], "Session not found")

    def test_websocket_streams_status_and_result_messages(self):
        """A valid session should stream status and final result events."""

        class FakeEvents:
            def __init__(self):
                self.handlers = {}

            def on(self, event_type, handler):
                self.handlers.setdefault(event_type, []).append(handler)

        class FakeMemory:
            @staticmethod
            def get_links():
                return {"https://example.com": "Example Source"}

        class FakeAgent:
            def __init__(self):
                self.events = FakeEvents()
                self.memory = FakeMemory()

            async def process_query(self, query):
                assert query == "stream this"
                return "# Report"

        session_id = "session-123"
        web_server.active_sessions[session_id] = {
            "query": "stream this",
            "max_iterations": 3,
            "created_at": "2026-03-25T00:00:00",
            "status": "initialized",
        }

        with patch.object(
            web_server.AgentFactory,
            "create_agent",
            AsyncMock(return_value=FakeAgent()),
        ):
            with TestClient(web_server.app) as client:
                with client.websocket_connect(
                    f"/ws/search/{session_id}"
                ) as websocket:
                    first = websocket.receive_json()
                    second = websocket.receive_json()
                    third = websocket.receive_json()
                    fourth = websocket.receive_json()

        self.assertEqual(first["type"], "status")
        self.assertEqual(first["message"], "Initializing agent...")
        self.assertEqual(second["type"], "status")
        self.assertIn("Processing query: stream this", second["message"])
        self.assertEqual(third["type"], "result")
        self.assertEqual(third["report"], "# Report")
        self.assertEqual(
            third["sources"],
            [["https://example.com", "Example Source"]],
        )
        self.assertEqual(fourth["type"], "complete")
        self.assertEqual(
            web_server.active_sessions[session_id]["status"],
            "completed",
        )


if __name__ == "__main__":
    unittest.main()
