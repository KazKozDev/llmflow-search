#!/usr/bin/env python3
"""
LLMFlow Search Agent - Agent Core Module
Agent Core - Central coordinator of the LLMFlow Search Agent.
Manages the flow of information and decision making.
"""
import logging
import time
import json
import asyncio

class AgentCore:
    def __init__(self, memory, planning, tools, report_generator, llm_service, max_iterations=10):
        """
        Initialize the agent core.
        
        Args:
            memory: Memory module for storing information
            planning: Planning module for search strategies
            tools: Tools module for web searches
            report_generator: Report generator for creating final output
            llm_service: LLMService for language model interactions
            max_iterations: Maximum number of search iterations
        """
        self.memory = memory
        self.planning = planning
        self.tools = tools
        self.report_generator = report_generator
        self.llm_service = llm_service
        self.max_iterations = max_iterations
        
        self.logger = logging.getLogger(__name__)
    
    async def process_query(self, query):
        """
        Process a query through the agent workflow.
        
        Args:
            query: The user's query
            
        Returns:
            A markdown-formatted report
        """
        self.logger.info(f"Processing query: {query}")
        
        # Store current query for context in parsing decisions
        self.current_query = query
        
        # Store query in memory
        self.memory.add_to_short_term({
            "type": "query",
            "content": query,
            "timestamp": time.time()
        })
        
        # Create search plan
        self.logger.info("Creating search plan...")
        # Planning is still sync for now as it uses LLM (which we made async capable but planning module calls it)
        # We can update PlanningModule later, for now we wrap it or assume it's fast enough
        # Actually, let's make planning async too if possible, but for now let's keep it sync to minimize changes
        # Wait, PlanningModule calls llm_service.generate_response which is sync.
        # We should probably update PlanningModule to use generate_response_async if we want full async
        # But for now, let's run it in executor if it's slow, or just keep it sync.
        # The main bottleneck is search, so parallelizing search is key.
        
        plan = self.planning.create_plan(query)
        
        # Log the plan
        self.logger.info(f"Search plan created with {len(plan['steps'])} steps")
        self.logger.debug(f"Plan details: {json.dumps(plan, indent=2)}")
        
        # Execute search loop
        current_step_idx = 0
        iterations = 0
        
        while (iterations < self.max_iterations and 
               current_step_idx < len(plan['steps'])):
            
            # Get current batch of steps (we can execute multiple steps in parallel if they are independent)
            # For now, let's execute one step at a time but asynchronously
            # Or better, look ahead and execute all "search" steps in parallel?
            # The current planning structure is linear.
            # Let's just execute the current step asynchronously.
            
            step = plan['steps'][current_step_idx]
            iterations += 1
            
            self.logger.info(f"Executing step {current_step_idx+1}/{len(plan['steps'])}: {step['description']}")
            
            # Execute the step based on type
            # Dynamic tool execution
            tool_name = step['type']
            if tool_name.startswith('search_'):
                await self._execute_search_step(step, plan)
            else:
                self.logger.warning(f"Unknown step type: {tool_name}")

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
            current_step_idx += 1
        
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
        
        # Report generation uses LLM, let's make it async if possible, or run in executor
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, self.report_generator.generate_report, query)
        
        return report
    
    async def _execute_search_step(self, step, plan):
        """
        Execute a generic search step asynchronously.
        """
        tool_name = step['type']
        query = step['query']
        
        try:
            # Execute search using the tool
            # The tools module should handle finding the right tool
            results = await self.tools.execute_tool(tool_name, query=query)
            
            # Normalize results to a standard format if possible
            # But for now, we just store what we get
            
            # Add to memory
            self.memory.add_to_short_term({
                "type": "search_results",
                "source": tool_name,
                "query": query,
                "results": results,
                "timestamp": time.time()
            })
            
            # Handle specific tool result formats for links and parsing
            if tool_name == 'search_duckduckgo':
                # DuckDuckGo returns a list of dicts
                for result in results[:self.tools.parse_top_results]:
                    await self._process_search_result(result)
                    
            elif tool_name == 'search_wikipedia':
                 # Wikipedia returns a dict
                 if results.get('page_found'):
                    self.memory.add_to_links(
                        results['url'],
                        f"Wikipedia: {results['title']}"
                    )
            
            elif tool_name == 'search_arxiv':
                # ArXiv returns list of dicts
                for result in results:
                    self.memory.add_to_links(result['url'], f"ArXiv: {result['title']}")
                    
            elif tool_name == 'search_youtube':
                # YouTube returns list of dicts
                for result in results:
                    self.memory.add_to_links(result['url'], f"YouTube: {result['title']}")
            
            # Generic link extraction for other tools if they return list of dicts with 'url' and 'title'
            elif isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and 'url' in item and 'title' in item:
                        self.memory.add_to_links(item['url'], f"{tool_name}: {item['title']}")

        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            self.memory.add_to_short_term({
                "type": "error",
                "source": tool_name,
                "error": str(e),
                "timestamp": time.time()
            })

    async def _process_search_result(self, result):
        """Process a single search result: decide to parse and parse if needed."""
        # Always add to links so it appears in references
        self.memory.add_to_links(result['url'], result['title'])
        
        # Get current progress for context
        current_items = self.memory.get_relevant_content(self.current_query, max_items=100) if hasattr(self, 'current_query') else []
        parsed_count = sum(1 for item in current_items if item.get('type') == 'parsed_content')
        
        # Use LLM to decide with full context
        prompt = f"""Decide if this search result should be parsed for detailed content extraction.

QUERY: {getattr(self, 'current_query', 'Unknown')}
URL: {result['url']}
TITLE: {result['title']}
SNIPPET: {result.get('snippet', 'No snippet available')}

CURRENT PROGRESS:
- Collected items: {len(current_items)}
- Already parsed: {parsed_count} pages

DECISION CRITERIA:
- Does this look like a PRIMARY source for the query?
- Is the snippet informative enough OR do we need full content?
- Would parsing add significant new information?
- Is this from a reputable source (Wikipedia, official sites, academic)?

Answer YES to parse if this is a key source. Answer NO if snippet is sufficient.
Answer:"""

        system_message = """You are a research strategist deciding which sources to analyze in depth.

PRIORITY - Parse these:
✓ Wikipedia articles (authoritative, comprehensive)
✓ Official websites (.gov, .edu, .org)
✓ Biographical sites (biography.com, britannica.com)
✓ Academic papers and research
✓ Primary sources for the topic

SKIP - Don't parse these:
✗ Listicles and low-quality content
✗ Ads and commercial pages
✗ Social media posts
✗ Duplicate information you already have
✗ If snippet already contains the key info

Your goal: Maximize information quality while minimizing redundant parsing."""

        should_parse = await self.llm_service.generate_response_async(prompt, system_message)
        should_parse = "YES" in should_parse.upper()
        
        if should_parse:
            self.logger.info(f"Parsing URL: {result['url']}")
            try:
                # Fix: Pass the full result dict, not just the URL string
                parsed_content = await self.tools.parse_duckduckgo_result(result)
                
                # Store parsed content
                self.memory.add_to_short_term({
                    "type": "parsed_content",
                    "source_url": result['url'],
                    "title": result['title'],
                    "content": parsed_content,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                self.logger.error(f"Error parsing {result['url']}: {str(e)}")
        else:
            self.logger.info(f"Skipping parsing for: {result['url']}")
