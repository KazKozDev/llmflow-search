
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

The result is a coherent, verified answer based on real data. It works in multiple languages and can bypass site restrictions.

## Requirements

* Python 3.8+
* OpenAI API key
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
   Edit the .env file, adding your OpenAI API key.

## Execution

Start the application:
```bash
python __main__.py
```

## Configuration

The system uses a `config.json` file for configuration. Example configuration:

```json
{
    "llm": {
        "model": "gpt-4o-mini",
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
    }
}
```

## Architecture

LLMFlow Search Agent consists of the following main modules:

1. **Planning Module**: Analyzes queries and creates a search plan
2. **Tools Module**: Provides tools for searching DuckDuckGo, Wikipedia, and web pages
3. **Memory Module**: Stores and retrieves information for context-aware processing
4. **Report Generator**: Synthesizes information into comprehensive reports
5. **LLM Service**: Provides interaction with OpenAI language models

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)
