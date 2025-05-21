<p align="center">
  <img src="https://github.com/user-attachments/assets/2fd25edf-32a3-42a2-a675-05fd6b3c5e6a" alt="LLMFlow Search Logo" width="550"/>
</p>

<h3 align="center">Deep research. Reliable results.</h3>

LLMFlow Search is an agent that finds accurate answers to complex questions using a smart search strategy.

It automatically refines queries: if the initial results are incomplete or inaccurate, the agent generates additional queries to fill in the gaps.

The agent explores information from various sources — Wikipedia, DuckDuckGo, and websites directly. It:

- Identifies which parts need verification
- Expands or narrows the search as needed
- Detects and resolves contradictions
- Chooses more precise wording
- **NEW**: Analyzes search intent to optimize queries

The result is a coherent, verified answer based on real data. It works in multiple languages and can bypass site restrictions.

## Requirements

* Python 3.8+
* LLM API key (supported providers in config.json)
* Chrome/Chromium (for Selenium-based web searches)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KazKozDev/LLMFlow-Search.git
   cd LLMFlow-Search
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit the .env file, adding your LLM provider API key.

## Execution

Start the application:
```bash
python main.py
```

Additional options:
```bash
python main.py --verbose            # Enable verbose logging
python main.py --disable-intent-analyzer  # Disable the Search Intent Analyzer
python main.py --max-iterations 10  # Increase max search iterations
```

## Configuration

The system uses a `config.json` file for configuration. Example configuration:

```json
{
    "llm": {
        "provider": "default_provider",
        "model": "default_model",
        "temperature": 0.2,
        "max_tokens": 2048
    },
    "search": {
        "max_results": 5,
        "safe_search": true,
        "parse_top_results": 3,
        "use_selenium": true,
        "use_cache": true
    },
    "memory": {
        "path": "./memory",
        "max_items": 100
    },
    "report": {
        "output_file": "report.md",
        "include_sources": true,
        "max_source_length": 1500
    },
    "intent_analyzer": {
        "enabled": true,
        "cache_results": true
    }
}
```

## New Feature: Search Intent Analyzer

The Search Intent Analyzer is a powerful new component that:

1. **Analyzes user search intentions** across various categories:
   - Factual queries
   - Informational queries
   - Navigational queries
   - Transactional queries
   - Educational queries
   - Research queries
   - Local queries
   - Urgent queries
   - Time-sensitive queries

2. **Analyzes key aspects** of each query:
   - Main entities (people, places, things, concepts)
   - Expected content type (articles, videos, maps, images)
   - Temporal context (relevance, historicity)
   - Level of detail (basic/in-depth)
   - Term specialization (general/specialized)
   - Time sensitivity (relevance of current date/time)

3. **Creates optimized queries** for:
   - Google (used by DuckDuckGo search)
   - Wikipedia

This leads to more accurate search results and better report quality.

## Architecture

LLMFlow Search Agent consists of the following main modules:

1. **Planning Module**: Analyzes queries and creates a search plan
2. **Search Intent Analyzer**: Optimizes queries for different search engines
3. **Tools Module**: Provides tools for searching DuckDuckGo, Wikipedia, and web pages
4. **Memory Module**: Stores and retrieves information for context-aware processing
5. **Report Generator**: Synthesizes information into comprehensive reports
6. **LLM Service**: Provides interaction with language models

For more detailed documentation on the core components, see [core/README.md](core/README.md).

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)
