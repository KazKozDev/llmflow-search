#!/usr/bin/env python3
"""
LLMFlow Search Agent - Report Generation System
Report Generator - Creates academically formatted reports using LLM synthesis.
Collects information from various sources and generates a clean, structured report
with numbered citations and a reference list.
"""
import logging
import time
import re
from datetime import datetime

class ReportGenerator:
    def __init__(self, memory, llm_service):
        """
        Initialize the report generator.
        Args:
            memory: MemoryModule instance
            llm_service: LLMService instance
        """
        self.memory = memory
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)

    def generate_report(self, query):
        """
        Generate an academically formatted report for the query, using memory and sources.
        Args:
            query: The original user query
        Returns:
            Markdown string with academic-style citations
        """
        self.logger.info(f"Generating report for query: {query}")
        
        # Gather relevant content
        relevant_items = self.memory.get_relevant_content(query, max_items=20)
        links = self.memory.get_links()
        
        # Prepare sources for citation
        sources = []
        for url, title in links.items():
            sources.append({
                'url': url,
                'title': title
            })
        
        # If we have fewer sources than links, add missing ones
        if len(sources) < len(links):
            for item in relevant_items:
                source_url = item.get('source_url')
                if source_url and source_url not in [s['url'] for s in sources]:
                    sources.append({
                        'url': source_url,
                        'title': item.get('title', 'Unknown Source')
                    })
        
        # Extract key information from relevant items
        information_blocks = []
        for item in relevant_items:
            title = item.get('title', '')
            content = item.get('content', '')
            source_url = item.get('source_url', None)
            
            # Find matching source index for citation
            source_index = None
            for i, source in enumerate(sources):
                if source['url'] == source_url:
                    source_index = i + 1  # 1-based indexing for citations
                    break
            
            # If we didn't find a matching source, add it
            if source_index is None and source_url:
                sources.append({
                    'url': source_url,
                    'title': title or 'Unknown Source'
                })
                source_index = len(sources)
            
            # Create information block with citation
            information_blocks.append({
                'title': title,
                'content': content[:2000],  # Limit content length
                'source_index': source_index
            })
        
        # Generate the final report using LLM
        final_report = self._generate_final_report(query, information_blocks, sources)
        
        return final_report
    
    def _generate_final_report(self, query, information_blocks, sources):
        """
        Generate the final report using LLM synthesis of collected information.
        """
        try:
            # Prepare information for LLM - providing more information for a more comprehensive report
            information_text = ""
            for block in information_blocks:
                source_citation = f"[{block['source_index']}]" if block['source_index'] else ""
                content_preview = block['content'][:1000] + "..." if len(block['content']) > 1000 else block['content']
                information_text += f"### {block['title']} {source_citation}\n\n{content_preview}\n\n"
            
            # Prepare sources list
            sources_text = ""
            if sources:
                sources_text = "\n\n## References\n\n"
                for i, source in enumerate(sources, 1):
                    sources_text += f"[{i}] {source['title']} - {source['url']}\n"
            else:
                sources_text = "\n\n*No references found.*"
            
            # Current date and time
            current_datetime = datetime.now()
            current_date = current_datetime.strftime("%Y-%m-%d")
            current_time = current_datetime.strftime("%H:%M:%S")
            
            # Create universal prompt for LLM that works with any query type
            prompt = f"""Create an informative analytical report on the query: "{query}"

Below is information from different sources. Your task is to thoroughly analyze all the information and create a coherent, comprehensive text:

{information_text}

Report requirements:

1. Format and style:
   - Write the report as a coherent text with minimal division into sections
   - Avoid excessive structuring and formatting
   - Use smooth transitions between topics and ideas

2. Content:
   - Include all important facts and data from sources
   - Synthesize information into a single coherent narrative
   - Add your analysis and conclusions, organically weaving them into the text
   - **CRITICAL: Use ONLY the information provided above. DO NOT invent facts or sources.**

3. Academic Citation:
   - Use proper academic citation format
   - Cite sources by indicating the number in square brackets [1], [2], etc.
   - Cite each specific fact or statement with the appropriate numbered reference
   - Every paragraph should contain at least one or more citations
   - All factual claims must be supported by citations
   - **CRITICAL: Use ONLY the citation numbers provided in the source text (e.g. [1], [2]). DO NOT invent new citation numbers.**

4. Time context:
   - The current date is {current_date}
   - The current time is {current_time}
   - Use this information as the context for your report - this is when the report is being generated
   - If time-sensitive information is discussed, consider this timestamp as the reference point

Volume and style:
- Write the report as a coherent text of about 3000 words
- Use academic style with minimal formatting
- Maintain formal, scholarly language throughout

DO NOT INCLUDE the list of sources - I will add it automatically.
DO NOT add a "References" or "Bibliography" section at the end.

Report date: {current_date} {current_time}
"""
            
            # Get report from LLM with enhanced system message for more detailed reports
            system_message = f"""You are an experienced analyst and expert in creating comprehensive and in-depth analytical reports using academic citation standards.

Your strengths:
1. Deep analysis - extract all important details and patterns
2. Comprehensive coverage - examine the topic from different angles
3. Structure - create a logical and easy-to-understand structure
4. Informativeness - enrich the report with specific data, facts, and examples
5. Accurate academic citation - indicate sources for all key facts using numbers in square brackets [1], [2], etc.
6. Analytical approach - make your own conclusions and forecasts

Today's date is {current_date} and the current time is {current_time}. Use this as your reference point for any time-sensitive information.

Always maintain proper academic citation practices. Each factual statement should be supported by a numbered citation in square brackets that corresponds to the numbered reference list.

**CRITICAL RULE:** You must ONLY use the sources provided in the context. NEVER invent, hallucinate, or create "illustrative" references. If you don't have enough information, state that based on the available sources. Do not make up books, articles, or links. Do not output a References section."""
            report_content = self.llm_service.generate_response(prompt, system_message)
            
            # Add title and sources
            title = f"# {query.title()}"
            datetime_line = f"*Report generated on {current_date} at {current_time}*"
            
            # Clean up the report content to remove any HTML or unwanted formatting
            clean_report = re.sub(r'<[^>]+>', '', report_content)
            clean_report = re.sub(r'\[\s*\]', '', clean_report)  # Remove empty citations
            
            # Post-processing: Remove any "References" or "Sources" section the LLM might have added
            # Matches "## References", "### Sources", "References:", etc. at the end of the text
            clean_report = re.sub(r'(?i)\n+#+\s*(References|Sources|Bibliography|Citations).*$', '', clean_report, flags=re.DOTALL)
            
            clean_report = re.sub(r'\n{3,}', '\n\n', clean_report)  # Remove excessive newlines
            
            # Combine everything
            final_report = f"{title}\n\n{datetime_line}\n\n{clean_report}\n{sources_text}"
            
            return final_report
        except Exception as e:
            self.logger.error(f"Error generating final report: {str(e)}")
            # Fallback to a simple report
            return self._generate_fallback_report(query, information_blocks, sources)
    
    def _generate_fallback_report(self, query, information_blocks, sources):
        """
        Generate a simple fallback report if LLM synthesis fails.
        """
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%Y-%m-%d")
        current_time = current_datetime.strftime("%H:%M:%S")
        
        # Create basic report structure
        report = [
            f"# {query.title()}",
            f"*Report generated on {current_date} at {current_time}*",
            "## Introduction",
            f"This report contains information on the query: '{query}'. The report presents data from {len(sources)} sources.",
            "## Main Information"
        ]
        
        # Create a summary of key points from information blocks
        summary_points = []
        for block in information_blocks:
            # Extract first sentence or first 100 chars as a summary point
            content = block['content']
            first_sentence = content.split('.')[0] if '.' in content[:200] else content[:100]
            
            if block['source_index']:
                first_sentence += f" [{block['source_index']}]"
            
            if first_sentence.strip() and len(first_sentence) > 20:  # Only add meaningful points
                summary_points.append(first_sentence.strip())
        
        # Add summary points (up to 10)
        for point in summary_points[:10]:
            report.append(f"- {point}")
        
        # Add sources
        report.append("## References")
        for i, source in enumerate(sources, 1):
            report.append(f"[{i}] {source['title']} - {source['url']}")
        
        return "\n\n".join(report)
    
    def _generate_executive_summary(self, query, relevant_items):
        """
        Generate an executive summary using LLM.
        """
        try:
            # Extract key information from relevant items
            content_summary = ""
            for item in relevant_items[:5]:  # Use top 5 items for summary
                title = item.get('title', '')
                snippet = item.get('content', '')[:300]  # First 300 chars
                content_summary += f"- {title}: {snippet}...\n\n"
            
            # Current date and time
            current_datetime = datetime.now()
            current_date = current_datetime.strftime("%Y-%m-%d")
            current_time = current_datetime.strftime("%H:%M:%S")
            
            # Create prompt for LLM
            prompt = f"""Create a brief summary (3-4 paragraphs) for the query: \"{query}\"
            
            Based on the following information:
            {content_summary}
            
            Current date and time: {current_date} {current_time}
            
            The summary should be informative, objective, and well-structured.
            Highlight 3-4 key points and write them as a bulleted list after the main text.
            """
            
            # Get summary from LLM
            system_message = f"You are a professional analyst who creates concise and informative summaries based on provided data. Today's date is {current_date} and the current time is {current_time}."
            summary = self.llm_service.generate_response(prompt, system_message)
            
            return summary
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return "*Failed to generate summary due to technical error.*"
    
    def _group_by_topics(self, query, relevant_items):
        """
        Group content by topics using LLM and keyword matching.
        """
        try:
            # Extract topics specifically for items using LLM
            topics_raw = self._extract_topics_with_llm(query, relevant_items)
            
            # Create topic structure
            topics = []
            if topics_raw:
                # Use LLM to assign items to topics
                topics = self._assign_items_to_topics(topics_raw, relevant_items)
            
            # Handle unassigned items
            assigned_item_ids = set()
            for _, items in topics:
                for item in items:
                    assigned_item_ids.add(id(item))
            
            unassigned = [item for item in relevant_items if id(item) not in assigned_item_ids]
            
            # Add unassigned items to a general category
            if unassigned:
                topics.append(("Additional Information", unassigned))
            
            # Sort topics by importance (number of items)
            topics.sort(key=lambda x: len(x[1]), reverse=True)
            
            return topics
        except Exception as e:
            self.logger.error(f"Error grouping by topics: {str(e)}")
            # Fallback: return all items under a single topic
            return [("Search Results", relevant_items)]
    
    def _assign_items_to_topics(self, topics, items):
        """
        Use LLM to assign items to topics.
        """
        try:
            # Format topics and items for LLM
            topics_text = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics)])
            
            items_text = ""
            for i, item in enumerate(items):
                title = item.get('title', '')
                snippet = item.get('content', '')[:200]  # First 200 chars
                items_text += f"Item {i+1}: {title}\nSnippet: {snippet}...\n\n"
            
            # Create prompt for LLM
            prompt = f"""Assign each item to the most relevant topic from the list below.
            
            Topics:
            {topics_text}
            
            Items:
            {items_text}
            
            Return your answer in the format:
            Topic 1:
            - Item X
            - Item Y
            
            Topic 2:
            - Item Z
            
            And so on...
            """
            
            # Get assignments from LLM
            system_message = "You are a professional analyst who categorizes information effectively."
            assignments_text = self.llm_service.generate_response(prompt, system_message)
            
            # Parse the assignments
            topic_pattern = r'Topic\s+(\d+):'
            item_pattern = r'Item\s+(\d+)'
            
            result = []
            current_topic = None
            current_items = []
            
            for line in assignments_text.split('\n'):
                line = line.strip()
                
                if not line:
                    continue
                
                topic_match = re.search(topic_pattern, line)
                if topic_match:
                    # Save previous topic and start new one
                    if current_topic is not None and current_items:
                        result.append((current_topic, current_items))
                        current_items = []
                    
                    topic_idx = int(topic_match.group(1)) - 1
                    if 0 <= topic_idx < len(topics):
                        current_topic = topics[topic_idx]
                    else:
                        current_topic = "Other"
                    continue
                
                item_match = re.search(item_pattern, line)
                if item_match and current_topic is not None:
                    item_idx = int(item_match.group(1)) - 1
                    if 0 <= item_idx < len(items):
                        current_items.append(items[item_idx])
            
            # Add the last topic
            if current_topic is not None and current_items:
                result.append((current_topic, current_items))
            
            return result
        except Exception as e:
            self.logger.error(f"Error assigning items to topics: {str(e)}")
            # Fallback: create one topic with all items
            return [(topics[0] if topics else "Search Results", items)]
    
    def _extract_topics_with_llm(self, query, relevant_items):
        """
        Extract topics using LLM.
        """
        try:
            # Extract titles and snippets
            content_for_topics = ""
            for item in relevant_items[:10]:  # Use top 10 items
                title = item.get('title', '')
                content_for_topics += f"- {title}\n"
            
            # Current date and time
            current_datetime = datetime.now()
            current_date = current_datetime.strftime("%Y-%m-%d")
            current_time = current_datetime.strftime("%H:%M:%S")
            
            # Create prompt for LLM
            prompt = f"""Based on the following titles, identify 3-5 main topics for grouping information related to the query: \"{query}\"
            
            Titles:
            {content_for_topics}
            
            Current date and time: {current_date} {current_time}
            
            Return only a list of topics without explanations, each topic on a new line.
            """
            
            # Get topics from LLM
            system_message = f"You are a professional analyst who groups information by topics. Today is {current_date} at {current_time}."
            topics_text = self.llm_service.generate_response(prompt, system_message)
            
            # Parse topics (one per line)
            topics = [t.strip() for t in topics_text.split('\n') if t.strip()]
            
            # Remove any numbering or bullets
            topics = [re.sub(r'^[\d\-\*\.\s]+', '', t).strip() for t in topics]
            
            # Filter out empty topics
            topics = [t for t in topics if t]
            
            return topics
        except Exception as e:
            self.logger.error(f"Error extracting topics: {str(e)}")
            return []  # Return empty list instead of hardcoded topics