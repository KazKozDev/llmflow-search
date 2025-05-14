#!/usr/bin/env python3
"""
Planning Module - Creates and revises search plans using LLM.
Implements Chain-of-Thought, Tree-of-Thought, and ReAct approaches.
"""
import logging
import json
import time
import re

class PlanningModule:
    def __init__(self, llm_service):
        """
        Initialize the planning module.
        
        Args:
            llm_service: LLM service for generating and revising plans
        """
        self.llm_service = llm_service
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
        
        # Улучшенный системный промпт для создания плана
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
        
        # Улучшенный промпт для создания плана
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
            
            if not search_results or not search_results.get("results"):
                self.logger.warning("No search results found for revision")
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
            response: Text response from LLM
            default_query: Default query to use if extraction fails
            
        Returns:
            Dictionary with search plan components
        """
        default_plan = {
            "main_keywords": [default_query],
            "wikipedia_topics": [],
            "alternative_keywords": [],
            "subtopics": []
        }
        
        # Strategy 1: Direct JSON parsing
        try:
            # Clean response
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.replace('```json', '', 1)
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response.rsplit('```', 1)[0]
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            plan = json.loads(cleaned_response)
            
            # Validate structure
            required_keys = ["main_keywords", "wikipedia_topics", "alternative_keywords", "subtopics"]
            if all(key in plan for key in required_keys):
                # Validate types
                if all(isinstance(plan[key], list) for key in required_keys):
                    # Validate content
                    if all(isinstance(item, str) for key in required_keys for item in plan[key]):
                        self.logger.info("Successfully parsed search plan JSON")
                        return plan
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse search plan as JSON, trying alternative extraction")
        except Exception as e:
            self.logger.warning(f"Error during search plan JSON parsing: {str(e)}")
        
        # Strategy 2: Extract JSON pattern
        try:
            json_pattern = r'\{[^\{\}]*"main_keywords"[^\{\}]*\}'
            json_matches = re.findall(json_pattern, response)
            if json_matches:
                try:
                    plan = json.loads(json_matches[0])
                    if "main_keywords" in plan:
                        self.logger.info("Extracted search plan JSON from response text")
                        return plan
                except:
                    pass
        except Exception as e:
            self.logger.warning(f"Error during search plan JSON pattern extraction: {str(e)}")
        
        # Strategy 3: Extract components separately
        try:
            extracted_plan = default_plan.copy()
            
            # Extract main keywords
            main_keywords_pattern = r'"main_keywords"\s*:\s*\[(.*?)\]'
            main_keywords_match = re.search(main_keywords_pattern, response, re.DOTALL)
            if main_keywords_match:
                keywords_text = main_keywords_match.group(1)
                keywords = re.findall(r'"([^"]*)"', keywords_text)
                if keywords:
                    extracted_plan["main_keywords"] = keywords
            
            # Extract wikipedia topics
            wiki_pattern = r'"wikipedia_topics"\s*:\s*\[(.*?)\]'
            wiki_match = re.search(wiki_pattern, response, re.DOTALL)
            if wiki_match:
                wiki_text = wiki_match.group(1)
                topics = re.findall(r'"([^"]*)"', wiki_text)
                if topics:
                    extracted_plan["wikipedia_topics"] = topics
            
            # Extract alternative keywords
            alt_pattern = r'"alternative_keywords"\s*:\s*\[(.*?)\]'
            alt_match = re.search(alt_pattern, response, re.DOTALL)
            if alt_match:
                alt_text = alt_match.group(1)
                alternatives = re.findall(r'"([^"]*)"', alt_text)
                if alternatives:
                    extracted_plan["alternative_keywords"] = alternatives
            
            # Extract subtopics
            sub_pattern = r'"subtopics"\s*:\s*\[(.*?)\]'
            sub_match = re.search(sub_pattern, response, re.DOTALL)
            if sub_match:
                sub_text = sub_match.group(1)
                subtopics = re.findall(r'"([^"]*)"', sub_text)
                if subtopics:
                    extracted_plan["subtopics"] = subtopics
            
            # If we extracted anything useful
            if (len(extracted_plan["main_keywords"]) > 0 or 
                len(extracted_plan["wikipedia_topics"]) > 0 or 
                len(extracted_plan["alternative_keywords"]) > 0 or 
                len(extracted_plan["subtopics"]) > 0):
                self.logger.info("Extracted search plan components from text")
                return extracted_plan
        except Exception as e:
            self.logger.warning(f"Error during component extraction: {str(e)}")
        
        # If all strategies fail, return default plan
        self.logger.warning("All extraction strategies failed, using default search plan")
        return default_plan
    
    def _extract_follow_up_searches(self, response):
        """
        Извлекает поисковые запросы из ответа LLM с использованием многоуровневой стратегии.
        
        Args:
            response: Текстовый ответ от LLM
            
        Returns:
            Список поисковых запросов
        """
        # Стратегия 1: Прямой парсинг JSON
        try:
            # Очищаем ответ от возможных маркеров кода и лишних символов
            cleaned_response = response.strip()
            # Удаляем маркеры кода, если они есть
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.replace('```json', '', 1)
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response.rsplit('```', 1)[0]
            cleaned_response = cleaned_response.strip()
            
            # Пытаемся распарсить JSON
            follow_ups = json.loads(cleaned_response)
            if "follow_up_searches" in follow_ups and isinstance(follow_ups["follow_up_searches"], list):
                searches = follow_ups["follow_up_searches"]
                # Проверяем, что все элементы - строки
                if all(isinstance(s, str) for s in searches):
                    self.logger.info(f"Successfully parsed JSON response: {searches}")
                    return searches
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse LLM response as JSON, trying alternative extraction")
        except Exception as e:
            self.logger.warning(f"Error during JSON parsing: {str(e)}")
        
        # Стратегия 2: Поиск JSON в тексте
        try:
            # Ищем паттерн JSON объекта с ключом follow_up_searches
            json_pattern = r'\{[^\{\}]*"follow_up_searches"[^\{\}]*\}'
            json_matches = re.findall(json_pattern, response)
            if json_matches:
                try:
                    follow_ups = json.loads(json_matches[0])
                    searches = follow_ups.get("follow_up_searches", [])
                    if searches and all(isinstance(s, str) for s in searches):
                        self.logger.info(f"Extracted JSON from response text: {searches}")
                        return searches
                except:
                    pass
        except Exception as e:
            self.logger.warning(f"Error during JSON pattern extraction: {str(e)}")
        
        # Стратегия 3: Извлечение цитат
        try:
            quoted_strings = re.findall(r'"([^"]*)"', response)
            if quoted_strings:
                # Фильтруем строки, которые похожи на поисковые запросы
                filtered_queries = [q for q in quoted_strings 
                                  if len(q) > 3 and not any(c in q for c in '{}[]')]
                if filtered_queries:
                    # Удаляем дубликаты, сохраняя порядок
                    unique_queries = []
                    for query in filtered_queries:
                        if query not in unique_queries:
                            unique_queries.append(query)
                    searches = unique_queries[:3]  # Ограничиваем до 3 запросов
                    self.logger.info(f"Extracted quoted strings as queries: {searches}")
                    return searches
        except Exception as e:
            self.logger.warning(f"Error during quoted string extraction: {str(e)}")
        
        # Стратегия 4: Извлечение строк после ключевых слов
        try:
            keywords = ["follow-up search", "additional query", "search for", "related topic"]
            lines = response.split('\n')
            extracted_queries = []
            
            for line in lines:
                for keyword in keywords:
                    if keyword.lower() in line.lower():
                        # Извлекаем текст после ключевого слова
                        query = line.lower().split(keyword.lower(), 1)[1].strip()
                        # Очищаем от пунктуации в начале
                        query = query.lstrip('":,.- ')
                        if query and len(query) > 3:
                            extracted_queries.append(query)
            
            if extracted_queries:
                searches = extracted_queries[:3]  # Ограничиваем до 3 запросов
                self.logger.info(f"Extracted queries from text: {searches}")
                return searches
        except Exception as e:
            self.logger.warning(f"Error during keyword extraction: {str(e)}")
        
        # Если все стратегии не сработали, возвращаем пустой список
        self.logger.warning("All extraction strategies failed, returning empty list")
        return []
