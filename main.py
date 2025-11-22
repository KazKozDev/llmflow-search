#!/usr/bin/env python3
"""
LLMFlow Search Agent - Main Entry Point
A production-ready agent that searches the web using DuckDuckGo and Wikipedia
and creates comprehensive reports with sources.
"""
import os
# Suppress tokenizers parallelism warning BEFORE any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import logging
import colorlog
import json
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    # Core modules from the core directory
    from core.agent_core import AgentCore
    from core.llm_service import LLMService
    from core.memory_module import MemoryModule
    from core.planning_module import PlanningModule
    from core.search_intent_analyzer import SearchIntentAnalyzer
    from core.report_generator import ReportGenerator
    from core.tools_module import ToolsModule
except ModuleNotFoundError as e:
    print(f"Module import error: {e}")
    sys.exit(1)

def setup_logging(verbose):
    """Set up colorful logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(handler)
    
    # Set lower log level for external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='LLMFlow Search Agent')
    # parser.add_argument('--query', '-q', type=str, help='Research query')  # Disabled: always prompt user
    parser.add_argument('--output', '-o', type=str, default='report.md', 
                        help='Path to save the report')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Verbose output')
    parser.add_argument('--max-iterations', '-m', type=int, default=10,
                        help='Maximum number of search iterations')
    parser.add_argument('--config', '-c', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--disable-intent-analyzer', '-d', action='store_true',
                        help='Disable the Search Intent Analyzer')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file.
    
    Attempts to load configuration from multiple locations in order of preference:
    1. The specified config_path
    2. A config.json file in the current directory
    3. A config.json file in the parent directory
    4. A config.json file in the user's home directory
    
    If no configuration file is found, uses default settings.
    """
    # Define default configuration
    default_config = {
        "llm": {
            "provider": "default_provider",
            "model": "default_model",
            "temperature": 0.2,
            "max_tokens": 4096
        },
        "search": {
            "max_results": 5,
            "safe_search": True,
            "parse_top_results": 3,
            "use_selenium": True,
            "use_cache": True
        },
        "memory": {
            "path": "./memory",
            "max_items": 100
        },
        "report": {
            "output_file": "report.md",
            "include_sources": True,
            "max_source_length": 3000
        },
        "intent_analyzer": {
            "enabled": True,
            "cache_results": True
        }
    }
    
    # List of potential config file locations
    config_locations = [
        config_path,                                      # Specified path
        os.path.join(os.getcwd(), "config.json"),         # Current directory
        os.path.join(os.path.dirname(os.getcwd()), "config.json"),  # Parent directory
        os.path.join(os.path.expanduser("~"), "config.json")  # Home directory
    ]
    
    # Try each location
    for location in config_locations:
        if os.path.exists(location):
            try:
                with open(location, 'r') as f:
                    config = json.load(f)
                    logging.info(f"Successfully loaded configuration from {location}")
                    
                    # Merge with default config (deep merge)
                    merged_config = default_config.copy()
                    for section in config:
                        if section in merged_config and isinstance(merged_config[section], dict):
                            merged_config[section].update(config[section])
                        else:
                            merged_config[section] = config[section]
                    
                    return merged_config
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing config file {location}: {e}. File must be valid JSON.")
            except IOError as e:
                logging.error(f"Error reading config file {location}: {e}. Check file permissions.")
    
    # If we get here, no valid config was found
    logging.warning("No valid configuration file found. Using default settings.")
    logging.info("To silence this warning, create a config.json file in the project directory.")
    logging.info("Example config locations: ./config.json, ../config.json, or ~/config.json")
    
    return default_config

def main():
    """Main application entry point."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging first
    setup_logging(args.verbose)
    
    # Load configuration once
    config = load_config(args.config)
    
    # Check for API key based on provider in config
    provider = config["llm"]["provider"].lower()
    
    # API key environment variable name will be constructed based on provider
    api_key_env = f"{provider.upper()}_API_KEY"
    
    # Skip API key check for Ollama
    if provider != "ollama" and not os.getenv(api_key_env):
        print(f"Error: {api_key_env} environment variable not set.")
        print("Please set it in a .env file or in your environment.")
        return 1
    
    # Always prompt the user for the query
    try:
        args.query = input("\nEnter your search query: ").strip()
    except EOFError:
        print("\n[ERROR] No interactive input. Exiting.")
        return 1
    if not args.query:
        print("Query cannot be empty.")
        return 1
    
    logging.info("Initializing LLMFlow Search Agent components...")
    
    llm_service = LLMService(
        provider=config["llm"]["provider"],
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )
    
    memory = MemoryModule(memory_path=config["memory"]["path"])
    
    # Initialize the Search Intent Analyzer if enabled
    intent_analyzer = None
    if config["intent_analyzer"]["enabled"] and not args.disable_intent_analyzer:
        logging.info("Initializing Search Intent Analyzer...")
        intent_analyzer = SearchIntentAnalyzer(llm_service)
    
    planning = PlanningModule(llm_service, search_intent_analyzer=intent_analyzer)
    
    tools = ToolsModule(
        memory=memory,
        llm_service=llm_service,
        config=config,
        max_results=config["search"]["max_results"],
        safe_search=config["search"].get("safe_search", True),
        parse_top_results=config["search"]["parse_top_results"]
    )
    
    report_generator = ReportGenerator(memory, llm_service)
    
    agent = AgentCore(
        memory=memory,
        planning=planning,
        tools=tools,
        report_generator=report_generator,
        llm_service=llm_service,
        max_iterations=args.max_iterations
    )
    
    logging.info(f"Processing query: {args.query}")
    
    # Run async process_query
    import asyncio
    try:
        report = asyncio.run(agent.process_query(args.query))
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
        return 1
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        return 1
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logging.info(f"Report saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 