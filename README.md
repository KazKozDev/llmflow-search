
<p align="center">
  <img src="https://github.com/user-attachments/assets/86b23e27-a263-49ed-8167-7bcab0c9d7a1" alt="LLMFlow Search Logo" width="500"/>
</p>
<h3 align="center">In-Depth Insights. Clear Outcomes.</h3>

LLMFlow Search is an agent that finds accurate answers to complex questions using a smart search strategy. It automatically refines queries: if the initial results are incomplete or inaccurate, the agent generates additional queries to fill in the gaps. 

The agent explores information from various sources — Wikipedia, DuckDuckGo, and websites directly. It:

- Analyzes search intent to optimize queries
- Identifies which parts need verification
- Expands or narrows the search as needed
- Detects and resolves contradictions
- Chooses more precise wording

The result is a coherent, verified answer based on real data. It works in multiple languages and can bypass site restrictions.

### Requirements

* Python 3.8+
* LLM API key (supported providers in config.json)
* Chrome/Chromium (for Selenium-based web searches)

<p align="center">
  <img src="https://github.com/user-attachments/assets/d4f738a1-e27e-415a-b44f-1374219057da" alt="report" style="width: 800px;">
</p>

>An example of a report compiled by LLMFlow Search agent

### Installation

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
   echo "# API Keys for LLM Providers
   OPENAI_API_KEY=" > .env
   ```
   Edit the .env file, adding your LLM provider API key.

### Execution

Start the application:
```bash
python main.py
```

### Configuration

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

### Architecture

LLMFlow Search Agent consists of the following main modules:

1. **Planning Module**: Analyzes queries and creates a search plan
2. **Tools Module**: Provides tools for searching DuckDuckGo, Wikipedia, and web pages
3. **Memory Module**: Stores and retrieves information for context-aware processing
4. **Report Generator**: Synthesizes information into comprehensive reports
5. **LLM Service**: Provides interaction with language models

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)
