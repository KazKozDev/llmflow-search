from types import SimpleNamespace

from llmflow_search import llm


def test_model_picker_preserves_non_numeric_input_as_first_task(monkeypatch):
    models = SimpleNamespace(
        models=[
            SimpleNamespace(model="gemma4:26b-mlx", size=8 * 1024**3),
            SimpleNamespace(model="qwen2.5:7b", size=4 * 1024**3),
            SimpleNamespace(model="other-model", size=2 * 1024**3),
        ]
    )
    monkeypatch.setattr(llm.ollama, "list", lambda: models)
    monkeypatch.setattr("builtins.input", lambda _prompt: "first user request")

    assert llm.pick_model() == "gemma4:26b-mlx"
    assert llm.pop_pending_initial_task() == "first user request"
    assert llm.pop_pending_initial_task() == ""


def test_model_picker_asks_once_and_returns_one_model(monkeypatch):
    models = SimpleNamespace(
        models=[
            SimpleNamespace(model="gemma4:26b-mlx", size=8 * 1024**3),
            SimpleNamespace(model="other-model", size=2 * 1024**3),
            SimpleNamespace(model="tiny-model", size=1024**3),
        ]
    )
    prompts = []
    monkeypatch.setattr(llm.ollama, "list", lambda: models)

    def _input(prompt):
        prompts.append(prompt)
        return "2"

    monkeypatch.setattr("builtins.input", _input)

    assert llm.pick_model() == "other-model"
    assert len(prompts) == 1
    assert llm.pop_pending_initial_task() == ""


def test_model_picker_enter_uses_default(monkeypatch):
    models = SimpleNamespace(
        models=[
            SimpleNamespace(model="gemma4:26b-mlx", size=8 * 1024**3),
            SimpleNamespace(model="qwen2.5:7b", size=4 * 1024**3),
            SimpleNamespace(model="tiny-model", size=1024**3),
        ]
    )
    monkeypatch.setattr(llm.ollama, "list", lambda: models)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert llm.pick_model() == "gemma4:26b-mlx"
    assert llm.pop_pending_initial_task() == ""


def test_json_chat_retries_invalid_json_in_json_mode(monkeypatch):
    calls = []

    def fake_chat(model, messages, **kwargs):
        calls.append((model, messages, kwargs))
        return {"content": "not json" if len(calls) == 1 else '{"ok": true}'}

    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)

    assert llm._ollama_chat_json("model", [{"role": "user", "content": "go"}], "system") == '{"ok": true}'
    assert len(calls) == 2
    assert calls[1][2]["json_mode"] is True
    assert calls[1][1][-1]["content"].startswith("Your response was not valid JSON")


def test_schema_chat_retries_with_same_schema(monkeypatch):
    calls = []
    schema = {"type": "object"}

    def fake_chat(model, messages, **kwargs):
        calls.append((model, messages, kwargs))
        return {"content": "not json" if len(calls) == 1 else "{}"}

    monkeypatch.setattr(llm, "_schema_capable_model", lambda _model: "schema-model")
    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)

    assert llm._ollama_chat_schema("mlx-model", [], "system", schema) == "{}"
    assert [call[0] for call in calls] == ["schema-model", "schema-model"]
    assert calls[1][2]["format_schema"] is schema
