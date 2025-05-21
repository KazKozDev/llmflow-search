#!/usr/bin/env python3
"""
LLMFlow Search Agent - Agent Core Module
Agent Core - Central coordinator of the LLMFlow Search Agent.
Manages the flow of information and decision making.
"""
import logging
import time
import json

class AgentCore:
    def __init__(self, memory, planning, tools, report_generator, llm_service, max_iterations=10):
        """
        Initialize the agent core.
        
        Args:
            memory: Memory module for storing information
            planning: Planning module for search strategies
            tools: Tools module for web searches
            report_generator: Report generator for creating final output
            llm_service: LLM service for language model interactions
            max_iterations: Maximum number of search iterations
        """
        self.memory = memory
        self.planning = planning
        self.tools = tools
        self.report_generator = report_generator
        self.llm_service = llm_service
        self.max_iterations = max_iterations
        
        self.logger = logging.getLogger(__name__)
    
    def process_query(self, query):
        """
        Process a user query and generate a comprehensive report.
        
        Args:
            query: The user's query string
            
        Returns:
            A markdown report with sources
        """
        self.logger.info(f"Processing query: {query}")
        
        # Store query in memory
        self.memory.add_to_short_term({
            "type": "query",
            "content": query,
            "timestamp": time.time()
        })
        
        # Create search plan
        self.logger.info("Creating search plan...")
        plan = self.planning.create_plan(query)
        
        # Log the plan
        self.logger.info(f"Search plan created with {len(plan['steps'])} steps")
        self.logger.debug(f"Plan details: {json.dumps(plan, indent=2)}")
        
        # Execute search loop
        current_step = 0
        iterations = 0
        
        while (iterations < self.max_iterations and 
               current_step < len(plan['steps'])):
            
            # Get current step
            step = plan['steps'][current_step]
            iterations += 1
            
            self.logger.info(f"Executing step {current_step+1}/{len(plan['steps'])}: {step['description']}")
            
            # Execute the step based on type
            if step['type'] == 'search_duckduckgo':
                self._execute_duckduckgo_search(step, plan)
            elif step['type'] == 'search_wikipedia':
                self._execute_wikipedia_search(step, plan)
            
            # Check if we need to revise the plan
            if iterations < self.max_iterations:
                revised_plan = self.planning.revise_plan(
                    plan, 
                    self.memory.get_short_term(),
                    step
                )
                
                # If plan was revised, log changes
                if len(revised_plan['steps']) > len(plan['steps']):
                    new_steps = len(revised_plan['steps']) - len(plan['steps'])
                    self.logger.info(f"Plan revised: added {new_steps} new steps")
                    plan = revised_plan
            
            # Move to next step
            current_step += 1
        
        # Once all searches complete, generate the report
        self.logger.info(f"Search complete after {iterations} iterations. Generating report...")
        
        # Save query and results to long-term memory
        self.memory.add_to_long_term({
            "type": "complete_query",
            "query": query,
            "steps_executed": iterations,
            "links": self.memory.get_links(),
            "timestamp": time.time()
        })
        
        report = self.report_generator.generate_report(query)
        
        return report
    
    def _execute_duckduckgo_search(self, step, plan):
        """
        Execute a DuckDuckGo search step.
        
        Args:
            step: The search step to execute
            plan: The overall search plan
        """
        # Execute search
        search_results = self.tools.search_duckduckgo(step['query'])
        
        # Add to memory
        self.memory.add_to_short_term({
            "type": "search_results",
            "source": "duckduckgo",
            "query": step['query'],
            "results": search_results,
            "timestamp": time.time()
        })
        
        # For each result, decide if we should parse it
        for result in search_results[:self.tools.parse_top_results]:
            should_parse = self.llm_service.determine_parsing_need(
                result['url'], 
                result.get('content', '') or result.get('snippet', '')
            )
            
            if should_parse:
                self.logger.info(f"Parsing URL: {result['url']}")
                try:
                    parsed_content = self.tools.parse_duckduckgo_result(result['url'])
                    
                    # Store parsed content
                    self.memory.add_to_short_term({
                        "type": "parsed_content",
                        "source_url": result['url'],
                        "title": result['title'],
                        "content": parsed_content,
                        "timestamp": time.time()
                    })
                    
                    # Store link
                    self.memory.add_to_links(result['url'], result['title'])
                    
                except Exception as e:
                    self.logger.error(f"Error parsing {result['url']}: {str(e)}")
            else:
                self.logger.info(f"Skipping parsing for: {result['url']}")
    
    def _execute_wikipedia_search(self, step, plan):
        """
        Execute a Wikipedia search step.
        
        Args:
            step: The search step to execute
            plan: The overall search plan
        """
        # Execute search
        wiki_result = self.tools.search_wikipedia(step['query'])
        
        # Add to memory
        self.memory.add_to_short_term({
            "type": "search_results",
            "source": "wikipedia",
            "query": step['query'],
            "result": wiki_result,
            "timestamp": time.time()
        })
        
        # Store link if page was found
        if wiki_result.get('page_found'):
            self.memory.add_to_links(
                wiki_result['url'],
                f"Wikipedia: {wiki_result['title']}"
            )
