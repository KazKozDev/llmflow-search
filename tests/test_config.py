from deep_research.config import load_config


def test_loads_footnote_mcp_configuration(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
search:
  provider: footnote_mcp
  language: en-US
  footnote:
    command: /opt/footnote-mcp
    args: [--quiet]
    provider: scrape
    semantic_rerank: true
    min_search_interval_seconds: 2.5
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.search.provider == "footnote_mcp"
    assert config.search.footnote.command == "/opt/footnote-mcp"
    assert config.search.footnote.args == ["--quiet"]
    assert config.search.footnote.semantic_rerank is True
    assert config.search.footnote.min_search_interval_seconds == 2.5
