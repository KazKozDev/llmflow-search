# LLMFlow Search Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/KazKozDev/LLMFlow-Search/main/assets/logo.png" alt="LLMFlow Search Logo" width="200"/>
</p>

<h3 align="center">Deep research. Reliable results.</h3>

LLMFlow Search is an LLM agent that processes complex queries. It searches, analyzes and synthesizes information from multiple web sources, generating context-aware answers.

LLMFlow Search combines accurate results with comprehensive analysis. Ideal for researchers synthesizing findings, students exploring topics, professionals seeking authoritative answers, or anyone looking for more than just standard search results. Every response is based on verifiable sources—no hallucinations, just real content. It leverages OpenAI's powerful language models to analyze and synthesize information from multiple sources. It's designed for easy integration into multi-agent systems, where it gathers and synthesizes web content while other agents handle tasks like analysis or content creation.

## Features

- **Query Analysis**: Processes complex questions and breaks them down into searchable components
- **Multi-Source Search**: Combines results from DuckDuckGo, Wikipedia, and direct web scraping
- **Adaptive Planning**: Adjusts search strategies based on initial findings
- **Comprehensive Reports**: Creates detailed reports with citations and sources
- **Memory System**: Retains information for context-aware processing
- **Multilingual Support**: Works with queries and content in multiple languages
- **Result Caching**: Improves performance for similar queries
- **Clarification Mechanism**: Handles ambiguous queries through refinement
- **Feedback Integration**: Learns from user feedback to improve results
- **Bypass Blocks**: Uses various strategies to access web resources
- **Customizable Configuration**: Allows fine-tuning of search parameters and report generation

## Workflow

The agent follows this process for query handling:
1. **Interpretation**: Processing the query 
2. **Planning**: Decomposing into search subtasks
3. **Search**: Retrieving information from available sources
4. **Analysis**: Processing collected data
5. **Response generation**: Creating a comprehensive answer

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

## Dependencies

- Python 3.8+
- OpenAI API
- Requests
- BeautifulSoup4
- Tenacity
- Markdownify
- Langchain
- Colorlog
- Selenium (for enhanced web searches)
- NumPy
- Pandas
- Fake-useragent
- Requests-cache
- Spacy
- Langdetect

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)
