#!/usr/bin/env python3
"""
LLMFlow Search Agent - Search Intent Analyzer Module
Search Intent Analyzer - Analyzes user search intentions and optimizes queries
for different search engines like Google and Wikipedia.
"""
import logging
import json
import datetime
import re
from typing import Dict, List, Any, Optional, Tuple

class SearchIntentAnalyzer:
    def __init__(self, llm_service):
        """
        Initialize the Search Intent Analyzer.
        
        Args:
            llm_service: LLM service for analyzing search intent
        """
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
        self.cache = {}  # Simple in-memory cache
    
    def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze the search intent of a user query and optimize it for different search engines.
        
        Args:
            query: The user's query string
            
        Returns:
            A dictionary with intent analysis and optimized queries
        """
        # Check cache first
        if query in self.cache:
            self.logger.info(f"Using cached intent analysis for: {query}")
            return self.cache[query]
        
        self.logger.info(f"Analyzing search intent for: {query}")
        
        # Get current date and time for context
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # System message for search intent analysis
        system_message = """
        You are an expert in analyzing search intentions. Your task is to transform user messages into optimal search queries for Google and Wikipedia.
        
        Analyze the user's search intention according to categories, key aspects, and return optimized queries.
        
        You MUST respond with valid JSON in the exact format specified below:
        {
            "intent_interpretation": "description of what the user wants",
            "time_sensitivity": {
                "is_sensitive": true/false,
                "explanation": "explanation if time-sensitive"
            },
            "intent_categories": ["category1", "category2"],
            "entities": ["entity1", "entity2"],
            "google_query": {
                "main_query": "optimized query for Google",
                "keywords": ["keyword1", "keyword2"],
                "operators": ["operator1", "operator2"],
                "alternative_queries": ["alt query1", "alt query2"]
            },
            "wikipedia_query": {
                "main_article": "title of the most appropriate article",
                "related_categories": ["category1", "category2"],
                "key_terms": ["term1", "term2"]
            }
        }
        
        Your response should contain ONLY the JSON object, nothing else.
        Do not include any explanations, notes, or additional text outside the JSON structure.
        The JSON must be properly formatted with double quotes around keys and string values.
        """
        
        # Create the prompt with the current date and time
        prompt = f"""
        The current date and time is: {current_datetime}
        Only include time-related parameters in search queries if the request is time-sensitive (e.g., current events, today's weather, recent news, etc.). For non-time-sensitive queries, ignore the date and time information.

        Analysis instructions:
        1. Search intent categories:
        * **Factual query** - looking for specific facts/data
        * **Informational query** - wants to learn general information on a topic
        * **Navigational query** - wants to find a specific site/resource
        * **Transactional query** - wants to buy/download something
        * **Educational query** - wants to learn how to do something (how-to)
        * **Research query** - seeks deep knowledge on a topic
        * **Local query** - looking for something nearby
        * **Urgent query** - needs information quickly
        * **Time-sensitive query** - needs current or timely information
        
        2. Analyze key aspects:
        * **Main entities** (people, places, things, concepts)
        * **Expected content type** (articles, videos, maps, images)
        * **Time context** (relevance, historicity)
        * **Level of detail** (basic/in-depth)
        * **Term specialization** (general/specialized)
        * **Time sensitivity** (is current date/time relevant to the query?)
        
        3. Optimization strategies:
        * **For Google:** use exact phrases, modifiers, operators, break complex questions into simple ones
        * **For Wikipedia:** use the most precise terms from section titles, categories, and articles
        * **For time-sensitive queries:** include relevant date/time parameters or terms like "today," "current," "latest"

        Now analyze the following user message:
        {query}
        """
        
        try:
            # Use LLM to analyze the search intent
            intent_analysis_response = self.llm_service.generate_response(prompt, system_message)
            
            # Parse the response
            intent_analysis = self._extract_intent_analysis(intent_analysis_response)
            
            # Store in cache
            self.cache[query] = intent_analysis
            
            return intent_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing search intent: {str(e)}")
            # Return a basic fallback analysis
            return self._generate_fallback_analysis(query)
    
    def _extract_intent_analysis(self, response: str) -> Dict[str, Any]:
        """
        Extract the intent analysis from the LLM response with robust error handling.
        
        Args:
            response: The LLM response string
            
        Returns:
            The parsed intent analysis as a dictionary
        """
        try:
            # Clean up the response - extract JSON if surrounded by markdown code blocks
            cleaned_response = response.strip()
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned_response)
            if json_match:
                cleaned_response = json_match.group(1).strip()
            
            # Parse the JSON
            intent_analysis = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = ["intent_interpretation", "google_query", "wikipedia_query"]
            for field in required_fields:
                if field not in intent_analysis:
                    self.logger.warning(f"Missing required field '{field}' in intent analysis")
                    intent_analysis[field] = {}
            
            return intent_analysis
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing intent analysis JSON: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            return self._generate_fallback_analysis("parse_error")
        
        except Exception as e:
            self.logger.error(f"Error in intent analysis extraction: {str(e)}")
            return self._generate_fallback_analysis("extraction_error")
    
    def _generate_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """
        Generate a fallback intent analysis when the LLM fails.
        
        Args:
            query: The original query or error type
            
        Returns:
            A basic intent analysis
        """
        if query in ["parse_error", "extraction_error"]:
            # Use the cached query if available
            for cached_query, analysis in self.cache.items():
                return analysis
            
            # Otherwise return a generic analysis
            return {
                "intent_interpretation": "Could not analyze intent",
                "time_sensitivity": {
                    "is_sensitive": False,
                    "explanation": "Unable to determine time sensitivity"
                },
                "intent_categories": ["informational query"],
                "entities": [],
                "google_query": {
                    "main_query": query if query not in ["parse_error", "extraction_error"] else "general information",
                    "keywords": [],
                    "operators": [],
                    "alternative_queries": []
                },
                "wikipedia_query": {
                    "main_article": "",
                    "related_categories": [],
                    "key_terms": []
                }
            }
        else:
            # Basic intent analysis for the query
            return {
                "intent_interpretation": f"General information about {query}",
                "time_sensitivity": {
                    "is_sensitive": False,
                    "explanation": "No time-sensitive elements detected"
                },
                "intent_categories": ["informational query"],
                "entities": [query],
                "google_query": {
                    "main_query": query,
                    "keywords": [query],
                    "operators": [],
                    "alternative_queries": []
                },
                "wikipedia_query": {
                    "main_article": query,
                    "related_categories": [],
                    "key_terms": [query]
                }
            } 