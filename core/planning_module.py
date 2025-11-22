#!/usr/bin/env python3
"""
LLMFlow Search Agent - Planning and Strategy
Planning Module - Creates and manages search plans for the agent.
Determines the best search strategies for different queries.
Integrates with Search Intent Analyzer for optimized query formulation.
"""
import logging
import json
import time
import re
from typing import Dict, Any, List

class PlanningModule:
    def __init__(self, llm_service, search_intent_analyzer=None):
        """
        Initialize the planning module.
        
        Args:
            llm_service: LLM service for generating and revising plans
            search_intent_analyzer: Search Intent Analyzer for optimizing queries (optional)
        """
        self.llm_service = llm_service
        self.search_intent_analyzer = search_intent_analyzer
        self.logger = logging.getLogger(__name__)
    
    def create_plan(self, query):
        """
        Create a search plan for the query.
        
        Args:
            query: The user's query
            
        Returns:
            A plan dictionary with search steps
        """
        self.logger.info(f"Creating search plan for: {query}")
        
        # Use search intent analyzer if available
        if self.search_intent_analyzer:
            self.logger.info("Using Search Intent Analyzer for query optimization")
            return self._create_intent_based_plan(query)
        else:
            self.logger.info("Using standard planning approach")
            return self._create_standard_plan(query)
    
    def _create_intent_based_plan(self, query):
        """
        Create a search plan using the Search Intent Analyzer.
        
        Args:
            query: The user's query
            
        Returns:
            A plan dictionary with optimized search steps
        """
        # Analyze the search intent
        intent_analysis = self.search_intent_analyzer.analyze_intent(query)
        
        # Create a structured plan
        plan = {
            "query": query,
            "created_at": time.time(),
            "intent_analysis": {
                "interpretation": intent_analysis.get("intent_interpretation", ""),
                "categories": intent_analysis.get("intent_categories", []),
                "entities": intent_analysis.get("entities", []),
                "time_sensitive": intent_analysis.get("time_sensitivity", {}).get("is_sensitive", False)
            },
            "steps": []
        }
        
        # Add DuckDuckGo search steps using Google query optimizations
        google_query = intent_analysis.get("google_query", {})
        main_query = google_query.get("main_query", query)
        
        # Add main query search
        plan["steps"].append({
            "type": "search_duckduckgo",
            "query": main_query,
            "description": f"Search DuckDuckGo for optimized query: {main_query}"
        })
        
        # Add alternative query searches
        for alt_query in google_query.get("alternative_queries", [])[:2]:  # Limit to 2 alternatives
            if alt_query and alt_query != main_query:
                plan["steps"].append({
                    "type": "search_duckduckgo",
                    "query": alt_query,
                    "description": f"Search DuckDuckGo for alternative query: {alt_query}"
                })
        
        # Add specialized tool searches based on recommendations
        recommended_tools = intent_analysis.get("recommended_tools", [])
        tool_queries = intent_analysis.get("tool_queries", {})
        
        for tool_name in recommended_tools:
            # Skip DuckDuckGo and Wikipedia as they're handled separately
            if tool_name in ["search_duckduckgo", "search_wikipedia"]:
                continue
            
            if tool_name in tool_queries and tool_queries[tool_name]:
                plan["steps"].append({
                    "type": tool_name,
                    "query": tool_queries[tool_name],
                    "description": f"Search {tool_name.replace('search_', '')} for: {tool_queries[tool_name]}"
                })
        
        # Add Wikipedia search using optimized Wikipedia query
        wikipedia_query = intent_analysis.get("wikipedia_query", {})
        main_article = wikipedia_query.get("main_article", "")
        
        if main_article:
            plan["steps"].append({
                "type": "search_wikipedia",
                "query": main_article,
                "description": f"Search Wikipedia for: {main_article}"
            })
        
        # Add searches for related categories from Wikipedia (limit to 1)
        for category in wikipedia_query.get("related_categories", [])[:1]:
            if category:
                plan["steps"].append({
                    "type": "search_wikipedia",
                    "query": category,
                    "description": f"Search Wikipedia for related category: {category}"
                })
        
        self.logger.info(f"Created intent-based plan with {len(plan['steps'])} steps")
        return plan
    
    def _create_standard_plan(self, query):
        """
        Create a standard search plan without the intent analyzer.
        This is the original planning logic.
        
        Args:
            query: The user's query
            
        Returns:
            A plan dictionary with search steps
        """
        # Enhanced system prompt for creating a plan
        system_message = """
        You are a professional research planner tasked with creating an efficient search strategy.
        You MUST respond with valid JSON in the exact format specified below:
        {
            "main_keywords": ["primary search query"],
            "wikipedia_topics": ["topic 1", "topic 2"],
            "alternative_keywords": ["alternative query 1", "alternative query 2"],
            "subtopics": ["subtopic 1", "subtopic 2"]
        }
        
        Your response should contain ONLY the JSON object, nothing else.
        Do not include any explanations, notes, or additional text outside the JSON structure.
        The JSON must be properly formatted with double quotes around keys and string values.
        Limit each category to 1-2 items to create a focused and efficient search plan.
        """
        
        # Enhanced prompt for creating a plan
        prompt = f"""
        Create an efficient search plan for the query: "{query}"
        
        Consider:
        1. The main keywords to search for directly
        2. Wikipedia topics that would provide good background information
        3. Alternative keywords or phrasings that might yield different results
        4. Specific subtopics worth exploring separately
        
        Ensure your plan is comprehensive but focused, with 1-2 items per category.
        """
        
        # Use LLM to create a search plan
        search_plan_response = self.llm_service.generate_response(prompt, system_message)
        
        # Parse the response with improved error handling
        search_plan = self._extract_search_plan(search_plan_response, query)
        
        # Create a structured plan with steps
        plan = {
            "query": query,
            "created_at": time.time(),
            "steps": []
        }
        
        # Add DuckDuckGo search for main keywords
        for keywords in search_plan.get("main_keywords", [query]):
            plan["steps"].append({
                "type": "search_duckduckgo",
                "query": keywords,
                "description": f"Search DuckDuckGo for: {keywords}"
            })
        
        # Add Wikipedia searches
        for topic in search_plan.get("wikipedia_topics", []):
            plan["steps"].append({
                "type": "search_wikipedia",
                "query": topic,
                "description": f"Search Wikipedia for: {topic}"
            })
        
        # Add searches for alternative keywords
        for alt_keywords in search_plan.get("alternative_keywords", []):
            plan["steps"].append({
                "type": "search_duckduckgo",
                "query": alt_keywords,
                "description": f"Search DuckDuckGo for alternative keywords: {alt_keywords}"
            })
        
        # Add searches for subtopics
        for subtopic in search_plan.get("subtopics", []):
            plan["steps"].append({
                "type": "search_duckduckgo",
                "query": f"{query} {subtopic}",
                "description": f"Search for subtopic: {subtopic}"
            })
        
        self.logger.info(f"Created plan with {len(plan['steps'])} steps")
        return plan

    def revise_plan(self, plan, memory, current_step):
        """
        Revise the search plan based on results.
        
        Args:
            plan: Current search plan
            memory: Short-term memory items
            current_step: The step that was just executed
            
        Returns:
            Updated plan
        """
        self.logger.info("Revising search plan based on results")
        
        # For DuckDuckGo searches, generate follow-up searches based on results
        if current_step["type"] == "search_duckduckgo":
            # Find the search results for the current step
            search_results = None
            for item in reversed(memory):
                if (item.get("type") == "search_results" and 
                    item.get("source") == "duckduckgo" and 
                    item.get("query") == current_step["query"]):
                    search_results = item
                    break
            
            # Check if search_results or its 'results' list is empty
            if not search_results or not search_results.get("results") or len(search_results.get("results")) == 0:
                self.logger.debug("No search results found for revision, keeping current plan.")
                return plan
            
            # Generate follow-up searches using the LLM
            system_message = """
            You are a research assistant identifying follow-up searches based on initial results.
            You MUST respond with valid JSON in the exact format specified below:
            {
                "follow_up_searches": ["search query 1", "search query 2"]
            }
            
            Your response should contain ONLY the JSON object, nothing else.
            Do not include any explanations, notes, or additional text outside the JSON structure.
            The JSON must be properly formatted with double quotes around keys and string values.
            Limit to 2-3 follow-up searches that are most likely to yield additional relevant information.
            """
            
            # Format the results for the prompt
            results_text = ""
            for i, result in enumerate(search_results.get("results", [])[:5]):
                # Handle different result formats
                title = result.get('title', '')
                content = result.get('content', '') or result.get('snippet', '')
                results_text += f"{i+1}. {title}: {content}\n"
            
            # Create the prompt
            prompt = f"""
            Based on the following search results for the query "{current_step['query']}":
            
            {results_text}
            
            Identify 2-3 follow-up search queries that would help gather additional relevant information.
            Focus on aspects not covered in these results or areas that need deeper exploration.
            """
            
            try:
                # Generate the response
                response = self.llm_service.generate_response(prompt, system_message)
                
                # Extract follow-up searches with improved extraction
                follow_up_searches = self._extract_follow_up_searches(response)
                
                # Add follow-up searches to the plan
                for query in follow_up_searches:
                    # Don't add duplicate searches
                    if not any(step["query"] == query for step in plan["steps"]):
                        plan["steps"].append({
                            "type": "search_duckduckgo",
                            "query": query,
                            "description": f"Follow-up search: {query}"
                        })
                        self.logger.info(f"Added follow-up search: {query}")
                
            except Exception as e:
                self.logger.error(f"Error generating follow-up searches: {str(e)}")
        
        return plan
    
    def _extract_search_plan(self, response, default_query):
        """
        Extract search plan from LLM response with robust error handling.
        
        Args:
            response: LLM response
            default_query: Default query to use if parsing fails
            
        Returns:
            Parsed search plan
        """
        try:
            # Clean up the response - extract JSON if surrounded by markdown code blocks
            cleaned_response = response.strip()
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned_response)
            if json_match:
                cleaned_response = json_match.group(1).strip()
            
            # Parse the search plan JSON
            search_plan = json.loads(cleaned_response)
            
            return search_plan
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing search plan JSON: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            
            # Return a basic search plan
            return {
                "main_keywords": [default_query],
                "wikipedia_topics": [default_query],
                "alternative_keywords": [],
                "subtopics": []
            }
        
        except Exception as e:
            self.logger.error(f"Error in search plan extraction: {str(e)}")
            return {
                "main_keywords": [default_query],
                "wikipedia_topics": [default_query],
                "alternative_keywords": [],
                "subtopics": []
            }
    
    def _extract_follow_up_searches(self, response):
        """
        Extract follow-up searches from LLM response with robust error handling.
        
        Args:
            response: LLM response
            
        Returns:
            List of follow-up search queries
        """
        try:
            # Clean up the response - extract JSON if surrounded by markdown code blocks
            cleaned_response = response.strip()
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned_response)
            if json_match:
                cleaned_response = json_match.group(1).strip()
            
            # Parse the follow-up searches JSON
            follow_up_data = json.loads(cleaned_response)
            
            # Extract the follow-up searches
            follow_up_searches = follow_up_data.get("follow_up_searches", [])
            
            # Filter out empty searches and limit to 3
            follow_up_searches = [q for q in follow_up_searches if q.strip()][:3]
            
            return follow_up_searches
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing follow-up searches JSON: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            return []
        
        except Exception as e:
            self.logger.error(f"Error in follow-up searches extraction: {str(e)}")
            return []
