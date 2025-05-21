#!/usr/bin/env python3
"""
LLMFlow Search Agent - Language Model Service
LLM Service - Provides language model functionality for the agent.
Includes support for multiple providers and models.
"""
import logging
import os
import json
import time
import importlib
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class LLMService:
    def __init__(self, provider, model, temperature=0.2, max_tokens=2048):
        """
        Initialize the LLM service.
        
        Args:
            provider: LLM provider name
            model: The model name to use
            temperature: Creativity parameter (0-1)
            max_tokens: Maximum tokens in response
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        self.client = None
        
        # Get API key from environment variables based on provider name
        self.api_key = os.getenv(f"{self.provider.upper()}_API_KEY")
        if not self.api_key:
            self.logger.warning(f"{self.provider.upper()}_API_KEY not set in environment variables")
        
        # Try to initialize the appropriate client based on provider
        try:
            if self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            elif self.provider == "gemini" or self.provider == "google":
                # For Gemini, we'll use direct API calls
                pass
            elif self.provider == "anthropic":
                try:
                    from anthropic import Anthropic
                    self.client = Anthropic(api_key=self.api_key)
                except ImportError:
                    self.logger.error("Anthropic package not installed. Run: pip install anthropic")
                    raise
            else:
                # Try to dynamically import a module for this provider
                try:
                    module_name = f"{self.provider.lower()}_client"
                    provider_module = importlib.import_module(module_name)
                    self.client = provider_module.get_client(self.api_key)
                except ImportError:
                    self.logger.warning(f"No specialized handling for provider '{self.provider}'. Using generic approach.")
                    # Will implement a generic approach for unsupported providers
        except Exception as e:
            self.logger.error(f"Error initializing client for {self.provider}: {str(e)}")
            # We don't raise here to allow for fallback mechanisms

        self.logger.info(f"LLM Service initialized with provider: {self.provider}, model: {self.model}")
    
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
        if self.provider == "openai":
            return self._chat_completion_generate(prompt, system_message)
        elif self.provider == "gemini" or self.provider == "google":
            return self._api_generate(prompt, system_message)
        elif self.provider == "anthropic":
            return self._anthropic_generate(prompt, system_message)
        else:
            return self._generic_generate(prompt, system_message)
    
    def _chat_completion_generate(self, prompt, system_message=None):
        """Generate response using chat completion API."""
        if not self.client:
            raise ValueError("Client not initialized")
            
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            self.logger.debug(f"Sending prompt to {self.model} via {self.provider}")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"LLM response received in {elapsed_time:.2f}s")
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _api_generate(self, prompt, system_message=None):
        """Generate response using direct REST API."""
        import urllib.request
        import ssl
        
        # Combine system message and prompt if both are provided
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = json.dumps({
            "contents": [
                {
                    "parts": [
                        {"text": full_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": self.max_tokens,
                "stopSequences": []
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }).encode('utf-8')
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        try:
            self.logger.debug(f"Sending prompt to {self.model} via API")
            start_time = time.time()
            
            # Disable SSL verification (use with caution)
            context = ssl._create_unverified_context()
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            
            with urllib.request.urlopen(req, context=context) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                elapsed_time = time.time() - start_time
                self.logger.debug(f"API response received in {elapsed_time:.2f}s")
                
                return result['candidates'][0]['content']['parts'][0]['text']
                
        except Exception as e:
            self.logger.error(f"Error generating API response: {str(e)}")
            raise
    
    def _anthropic_generate(self, prompt, system_message=None):
        """Generate response using Anthropic API."""
        if not self.client:
            raise ValueError("Anthropic client not initialized")
            
        try:
            self.logger.debug(f"Sending prompt to {self.model} via Anthropic")
            start_time = time.time()
            
            # Format based on Anthropic's API
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Anthropic response received in {elapsed_time:.2f}s")
            
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Error generating Anthropic response: {str(e)}")
            raise
    
    def _generic_generate(self, prompt, system_message=None):
        """Generic fallback for unsupported providers."""
        self.logger.warning(f"Using generic handler for provider {self.provider}. Functionality may be limited.")
        
        # Implement a very basic fallback mechanism here
        # This could be extended to support other providers as needed
        return f"[LLM Service] Provider {self.provider} with model {self.model} is not fully supported. Please implement specific handling for this provider."
    
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
        Create a search plan for the query.
        
        Args:
            query: The user's query
            
        Returns:
            A search plan with keywords and strategies
        """
        system_message = """You are a research planning assistant. Your task is to create an 
effective search plan for answering a complex query."""
        
        prompt = f"""
For the query: "{query}"

Create a search plan including:
1. A breakdown of the query into 2-3 component questions
2. 4-6 effective search keywords or phrases for each component
3. Suggested search strategies
4. Potential reliable sources to prioritize
5. Specific facts that need verification

Provide your plan in a structured JSON format.
"""
        
        try:
            response = self.generate_response(prompt, system_message)
            
            # Try to parse the response as JSON
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # If parsing fails, return the raw response
                self.logger.warning("LLM did not return valid JSON for search plan")
                return {"raw_plan": response}
                
        except Exception as e:
            self.logger.error(f"Error creating search plan: {str(e)}")
            return {"error": str(e)}
