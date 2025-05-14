#!/usr/bin/env python3
"""
LLM Service - Interface to GPT-4o-mini via the OpenAI API.
Handles prompts, responses, and error handling.
"""
import logging
import os
import json
import time
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class LLMService:
    def __init__(self, model="gpt-4o-mini", temperature=0.2, max_tokens=2048):
        """
        Initialize the LLM service.
        
        Args:
            model: The OpenAI model to use
            temperature: Creativity parameter (0-1)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"LLM Service initialized with model: {model}")
    
    @retry(
        retry=retry_if_exception_type((
            ConnectionError, 
            TimeoutError
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=lambda retry_state: logging.warning(
            f"API error, retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    def generate_response(self, prompt, system_message=None):
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            system_message: Optional system message to guide the LLM
            
        Returns:
            The LLM's response as a string
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            self.logger.debug(f"Sending prompt to {self.model}")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=4096  # Увеличено для поддержки более длинных отчетов
            )
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"LLM response received in {elapsed_time:.2f}s")
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    def analyze_search_results(self, query, results):
        """
        Analyze search results using the LLM.
        
        Args:
            query: The original search query
            results: List of search results
            
        Returns:
            Analysis of relevance and key points
        """
        system_message = "You are a research assistant analyzing search results. Extract key information and evaluate relevance."
        
        # Format results for the LLM
        formatted_results = json.dumps(results, indent=2)
        
        prompt = f"""
Analyze these search results for the query: \"{query}\"

Search Results:
{formatted_results}

For each result, provide:
1. Relevance score (1-10)
2. Key information points
3. Whether it's worth exploring further

Output your analysis in a structured JSON format.
"""
        
        try:
            response = self.generate_response(prompt, system_message)
            
            # Try to parse the response as JSON
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # If parsing fails, return the raw response
                self.logger.warning("LLM did not return valid JSON for analysis")
                return {"raw_analysis": response}
                
        except Exception as e:
            self.logger.error(f"Error analyzing search results: {str(e)}")
            return {"error": str(e)}
    
    def determine_parsing_need(self, url, snippet):
        """
        Determine if a search result should be parsed.
        
        Args:
            url: The URL of the search result
            snippet: The snippet from the search result
            
        Returns:
            Boolean indicating whether to parse the result
        """
        system_message = "You are a research assistant deciding which search results to analyze in depth."
        
        prompt = f"""
Should this search result be parsed for more detailed information?

URL: {url}
Snippet: {snippet}

Consider:
- Relevance and informativeness
- Credibility of the source
- Likelihood of containing useful information

Answer only YES or NO.
"""
        
        try:
            response = self.generate_response(prompt, system_message).strip().upper()
            return "YES" in response
        except Exception as e:
            self.logger.error(f"Error determining parsing need: {str(e)}")
            # Default to not parsing on error
            return False
    
    def create_search_plan(self, query):
        """
        Create a search plan for a given query.
        
        Args:
            query: The user's query
            
        Returns:
            A search plan including keywords and search strategies
        """
        system_message = "You are a strategic research assistant planning a comprehensive search approach."
        
        prompt = f"""
Create a search plan for investigating this query: \"{query}\"

Your plan should include:
1. 2-3 sets of search keywords (different phrasings)
2. Specific subtopics to explore
3. The order of searches to perform
4. Any Wikipedia topics that should be specifically searched

Format the response as JSON with these keys:
- main_keywords: List of primary keywords
- alternative_keywords: List of alternative phrasings
- subtopics: List of specific aspects to research
- search_order: List of search steps in order
- wikipedia_topics: List of specific topics to look up on Wikipedia
"""
        
        try:
            response = self.generate_response(prompt, system_message)
            
            # Try to parse the response as JSON
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # If parsing fails, extract information manually
                self.logger.warning("LLM did not return valid JSON for search plan")
                
                # Create a basic plan
                return {
                    "main_keywords": [query],
                    "alternative_keywords": [query + " guide", query + " explained"],
                    "subtopics": [],
                    "search_order": ["main_keywords", "wikipedia", "alternative_keywords"],
                    "wikipedia_topics": [query]
                }
                
        except Exception as e:
            self.logger.error(f"Error creating search plan: {str(e)}")
            
            # Return a minimal plan
            return {
                "main_keywords": [query],
                "alternative_keywords": [],
                "subtopics": [],
                "search_order": ["main_keywords"],
                "wikipedia_topics": [query]
            }
