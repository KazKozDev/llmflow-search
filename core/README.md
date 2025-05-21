# LLMFlow Search Agent Core Components

## Overview
This directory contains the core components of the LLMFlow Search Agent. Each module is responsible for a specific aspect of the search process.

## Components

### Agent Core (`agent_core.py`)
Central coordinator that manages the flow of information and decision making. It orchestrates the search process using all other components.

### LLM Service (`llm_service.py`)
Provides a unified interface for interacting with Language Models. Supports multiple providers through a common interface.

### Memory Module (`memory_module.py`)
Handles storage and retrieval of search context, results, and extracted information during the search process.

### Planning Module (`planning_module.py`)
Creates and manages search plans, determining the best search strategies for different queries. Now integrated with the Search Intent Analyzer for optimized queries.

### Search Intent Analyzer (`search_intent_analyzer.py`)
New component that analyzes user search intentions and optimizes queries for different search engines. Features include:

- **Intent Categorization**: Analyzes queries across various categories (factual, informational, navigational, etc.)
- **Entity Extraction**: Identifies key entities in the query
- **Query Optimization**: Creates optimized queries for different search engines
- **Temporal Context**: Considers time sensitivity in queries
- **Caching**: Caches analysis results for frequently asked queries

### Report Generator (`report_generator.py`)
Creates comprehensive, well-structured reports based on search results and extracted information.

### Tools Module (`tools_module.py`)
Provides the tools needed for web searches, including DuckDuckGo and Wikipedia integration, as well as web page parsing.

## Integration Flow

1. User submits a query through the main entry point
2. The query is processed by the Search Intent Analyzer (if enabled)
3. The Planning Module uses the intent analysis to create an optimized search plan
4. The Agent Core executes the plan using the Tools Module
5. Results are stored in the Memory Module and used to generate a report

## Configuration

The Search Intent Analyzer can be configured in `config.json`:

```json
"intent_analyzer": {
    "enabled": true,
    "cache_results": true
}
```

You can also disable it at runtime with the `--disable-intent-analyzer` command-line option.

## Usage Example

```python
from core.llm_service import LLMService
from core.search_intent_analyzer import SearchIntentAnalyzer

# Initialize the LLM service
llm_service = LLMService(provider="openai", model="gpt-4o-mini")

# Create the Search Intent Analyzer
intent_analyzer = SearchIntentAnalyzer(llm_service)

# Analyze a query
analysis = intent_analyzer.analyze_intent("How does blockchain work for beginners?")

# Use the analysis for search
google_query = analysis["google_query"]["main_query"]
wiki_article = analysis["wikipedia_query"]["main_article"]

print(f"Optimized Google query: {google_query}")
print(f"Optimized Wikipedia article: {wiki_article}")
``` 