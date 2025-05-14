
<p align="center">
  <img src="https://github.com/user-attachments/assets/2fd25edf-32a3-42a2-a675-05fd6b3c5e6a" alt="LLMFlow Search Logo" width="650"/>
</p>

<h3 align="center">Deep research. Reliable results.</h3>

LLMFlow Search is an agent that finds accurate answers to complex questions using an advanced search strategy.

The main advantage of the agent is that it refines the search queries itself. If the initial search yields incomplete or inaccurate results, LLMFlow automatically formulates additional queries to fill in the information gaps. It does not pester the user with additional questions, but adjusts the search strategy on its own.

The agent intelligently explores information in various sources - Wikipedia, DuckDuckGo and directly on websites. In doing so, it:

- Determines which parts of the answer require additional verification
- Automatically expands and narrows the search area
- Identifies and resolves inconsistencies between sources
- Finds alternative wording for more efficient searches.

The result is a coherent, validated answer based on multiple pieces of information gathered through a multi-step search. Works in multiple languages and allows you to bypass site restrictions to access the information you need.

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
python -m llmflow
```

### Command Line Parameters

- `--query`, `-q`: Search query
- `--output`, `-o`: Path to save the report (default: report.md)
- `--config`, `-c`: Path to configuration file
- `--verbose`, `-v`: Detailed output for debugging
- `--cache`, `-C`: Use result caching (default: True)

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
