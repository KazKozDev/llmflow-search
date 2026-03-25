#!/usr/bin/env python3
"""Report generation for the LLMFlow Search agent."""

from datetime import datetime
import logging
import re
from textwrap import dedent


class ReportGenerator:
    """Create markdown reports from collected search results."""

    def __init__(self, memory, llm_service):
        """Initialize the report generator.

        Args:
            memory: MemoryModule instance.
            llm_service: LLM service instance.
        """
        self.memory = memory
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)

    def generate_report(self, query):
        """Generate an academically formatted report for a query.

        Args:
            query: The original user query.

        Returns:
            Markdown string with academic-style citations.
        """
        self.logger.info("Generating report for query: %s", query)

        relevant_items = self.memory.get_relevant_content(query, max_items=20)
        links = self.memory.get_links()
        sources, information_blocks = self._build_source_material(
            relevant_items,
            links,
        )

        return self._generate_final_report(
            query,
            information_blocks,
            sources,
        )

    def _build_source_material(self, relevant_items, links):
        """Convert memory items and links into report inputs."""
        sources = []
        source_lookup = {}

        def ensure_source(url, title):
            if not url:
                return None

            if url in source_lookup:
                return source_lookup[url]

            sources.append(
                {
                    "url": url,
                    "title": title or "Unknown Source",
                }
            )
            source_lookup[url] = len(sources)
            return source_lookup[url]

        for url, title in links.items():
            ensure_source(url, title)

        information_blocks = []
        for item in relevant_items:
            if item.get("type") == "search_results":
                self._append_search_result_blocks(
                    information_blocks,
                    item,
                    ensure_source,
                )
                continue

            title = item.get("title", "") or item.get("query", "")
            content = item.get("content", "") or item.get("snippet", "")
            source_url = item.get("source_url") or item.get("url")

            if not title and not content:
                continue

            source_index = ensure_source(source_url, title)
            information_blocks.append(
                {
                    "title": title,
                    "content": content[:2000],
                    "source_index": source_index,
                }
            )

        return sources, information_blocks

    def _append_search_result_blocks(
        self,
        information_blocks,
        item,
        ensure_source,
    ):
        """Expand raw search result lists into information blocks."""
        for result in item.get("results", [])[:5]:
            if not isinstance(result, dict):
                continue

            title = result.get("title", "")
            content = result.get("content") or result.get("snippet", "")
            source_url = result.get("url")

            if not title and not content:
                continue

            source_index = ensure_source(source_url, title)
            information_blocks.append(
                {
                    "title": title or item.get("query", "Search Result"),
                    "content": content[:2000],
                    "source_index": source_index,
                }
            )

    def _generate_final_report(self, query, information_blocks, sources):
        """Generate the final report using LLM synthesis."""
        try:
            information_text = self._build_information_text(information_blocks)
            source_catalog_text = self._build_source_catalog_text(sources)
            sources_text = self._build_references_text(sources)
            current_date, current_time = self._current_date_time()

            prompt = self._build_report_prompt(
                query,
                information_text,
                source_catalog_text,
                current_date,
                current_time,
            )
            system_message = self._build_system_message(
                current_date,
                current_time,
            )
            report_content = self.llm_service.generate_response(
                prompt,
                system_message,
            )

            title = f"# {query.title()}"
            datetime_line = (
                f"*Report generated on {current_date} at {current_time}*"
            )
            clean_report = self._clean_report_content(report_content)

            return (
                f"{title}\n\n{datetime_line}\n\n"
                f"{clean_report}\n{sources_text}"
            )
        except Exception as error:
            self.logger.error("Error generating final report: %s", error)
            return self._generate_fallback_report(
                query,
                information_blocks,
                sources,
            )

    def _build_information_text(self, information_blocks):
        """Build the information section passed to the LLM."""
        blocks_text = []
        for block in information_blocks:
            source_index = block.get("source_index")
            source_citation = f"[{source_index}]" if source_index else ""
            content = block.get("content", "")
            content_preview = content[:1000]
            if len(content) > 1000:
                content_preview = f"{content_preview}..."

            blocks_text.append(
                f"### {block.get('title', '')} {source_citation}\n\n"
                f"{content_preview}\n"
            )

        return "\n".join(blocks_text)

    def _build_source_catalog_text(self, sources):
        """Build the numbered source catalog for the prompt."""
        if not sources:
            return "No sources were collected.\n"

        lines = []
        for index, source in enumerate(sources, 1):
            lines.append(f"[{index}] {source['title']} - {source['url']}")

        return "\n".join(lines) + "\n"

    def _build_references_text(self, sources):
        """Build the references appendix appended to the report."""
        if not sources:
            return "\n\n*No references found.*"

        lines = ["", "", "## References", ""]
        for index, source in enumerate(sources, 1):
            lines.append(f"[{index}] {source['title']} - {source['url']}")

        return "\n".join(lines)

    def _build_report_prompt(
        self,
        query,
        information_text,
        source_catalog_text,
        current_date,
        current_time,
    ):
        """Create the main prompt for report synthesis."""
        return dedent(
            f"""
            Create an informative analytical report on the query: "{query}"

            Below is information from different sources. Your task is to
            thoroughly analyze all the information and create a coherent,
            comprehensive text:

            {information_text}

            Available source list:

            {source_catalog_text}

            Report requirements:

            1. Format and style:
               - Write the report as a coherent text with minimal division into
                 sections
               - Avoid excessive structuring and formatting
               - Use smooth transitions between topics and ideas

            2. Content:
               - Include all important facts and data from sources
               - Synthesize information into a single coherent narrative
               - Add your analysis and conclusions, organically weaving them
                 into the text
               - CRITICAL: Use ONLY the information provided above. DO NOT
                 invent facts or sources.
               - If only titles or snippets are available for some sources, use
                 them cautiously and state limitations precisely instead of
                 claiming that no sources were provided.

            3. Academic Citation:
               - Use proper academic citation format
               - Cite sources by indicating the number in square brackets [1],
                 [2], etc.
               - Cite each specific fact or statement with the appropriate
                 numbered reference
               - Every paragraph should contain at least one or more citations
               - All factual claims must be supported by citations
               - CRITICAL: Use ONLY the citation numbers provided in the source
                 text (e.g. [1], [2]). DO NOT invent new citation numbers.

            4. Time context:
               - The current date is {current_date}
               - The current time is {current_time}
               - Use this information as the context for your report; this is
                 when the report is being generated
               - If time-sensitive information is discussed, consider this
                 timestamp as the reference point

            Volume and style:
            - Write the report as a coherent text of about 3000 words
            - Use academic style with minimal formatting
            - Maintain formal, scholarly language throughout

            DO NOT INCLUDE the list of sources; I will add it automatically.
            DO NOT add a "References" or "Bibliography" section at the end.

            Report date: {current_date} {current_time}
            """
        ).strip()

    def _build_system_message(self, current_date, current_time):
        """Create the system message for report generation."""
        return dedent(
            f"""
            You are an experienced analyst and expert in creating
            comprehensive and in-depth analytical reports using academic
            citation standards.

            Your strengths:
            1. Deep analysis - extract all important details and patterns
            2. Comprehensive coverage - examine the topic from different angles
            3. Structure - create a logical and easy-to-understand structure
            4. Informativeness - enrich the report with specific data, facts,
               and examples
            5. Accurate academic citation - indicate sources for all key facts
               using numbers in square brackets [1], [2], etc.
            6. Analytical approach - make your own conclusions and forecasts

            Today's date is {current_date} and the current time is
            {current_time}. Use this as your reference point for any
            time-sensitive information.

            Always maintain proper academic citation practices. Each factual
            statement should be supported by a numbered citation in square
            brackets that corresponds to the numbered reference list.

            CRITICAL RULE: You must ONLY use the sources provided in the
            context. NEVER invent, hallucinate, or create illustrative
            references. If you don't have enough information, state that based
            on the available sources. Do not make up books, articles, or links.
            Do not output a References section.
            """
        ).strip()

    def _clean_report_content(self, report_content):
        """Normalize the generated report before returning it."""
        clean_report = re.sub(r"<[^>]+>", "", report_content)
        clean_report = re.sub(r"\[\s*\]", "", clean_report)
        clean_report = re.sub(
            r"(?i)\n+#+\s*(References|Sources|Bibliography|Citations).*$",
            "",
            clean_report,
            flags=re.DOTALL,
        )
        return re.sub(r"\n{3,}", "\n\n", clean_report)

    def _generate_fallback_report(self, query, information_blocks, sources):
        """Generate a simple fallback report if LLM synthesis fails."""
        current_date, current_time = self._current_date_time()
        report = [
            f"# {query.title()}",
            f"*Report generated on {current_date} at {current_time}*",
            "## Introduction",
            (
                f"This report contains information on the query: '{query}'. "
                f"The report presents data from {len(sources)} sources."
            ),
            "## Main Information",
        ]

        summary_points = []
        for block in information_blocks:
            content = block["content"]
            if "." in content[:200]:
                first_sentence = content.split(".")[0]
            else:
                first_sentence = content[:100]

            if block["source_index"]:
                first_sentence += f" [{block['source_index']}]"

            if first_sentence.strip() and len(first_sentence) > 20:
                summary_points.append(first_sentence.strip())

        for point in summary_points[:10]:
            report.append(f"- {point}")

        report.append("## References")
        for index, source in enumerate(sources, 1):
            report.append(f"[{index}] {source['title']} - {source['url']}")

        return "\n\n".join(report)

    def _generate_executive_summary(self, query, relevant_items):
        """Generate an executive summary using the LLM."""
        try:
            content_summary = ""
            for item in relevant_items[:5]:
                title = item.get("title", "")
                snippet = item.get("content", "")[:300]
                content_summary += f"- {title}: {snippet}...\n\n"

            current_date, current_time = self._current_date_time()
            prompt = dedent(
                f"""
                Create a brief summary (3-4 paragraphs) for the query:
                \"{query}\"

                Based on the following information:
                {content_summary}

                Current date and time: {current_date} {current_time}

                The summary should be informative, objective, and
                well-structured. Highlight 3-4 key points and write them as a
                bulleted list after the main text.
                """
            ).strip()
            system_message = dedent(
                f"""
                You are a professional analyst who creates concise and
                informative summaries based on provided data. Today's date is
                {current_date} and the current time is {current_time}.
                """
            ).strip()
            return self.llm_service.generate_response(prompt, system_message)
        except Exception as error:
            self.logger.error("Error generating summary: %s", error)
            return "*Failed to generate summary due to technical error.*"

    def _group_by_topics(self, query, relevant_items):
        """Group content by topics using LLM and keyword matching."""
        try:
            topics_raw = self._extract_topics_with_llm(query, relevant_items)
            topics = []
            if topics_raw:
                topics = self._assign_items_to_topics(
                    topics_raw,
                    relevant_items,
                )

            assigned_item_ids = set()
            for _, items in topics:
                for item in items:
                    assigned_item_ids.add(id(item))

            unassigned = [
                item
                for item in relevant_items
                if id(item) not in assigned_item_ids
            ]

            if unassigned:
                topics.append(("Additional Information", unassigned))

            topics.sort(
                key=lambda topic_items: len(topic_items[1]),
                reverse=True,
            )
            return topics
        except Exception as error:
            self.logger.error("Error grouping by topics: %s", error)
            return [("Search Results", relevant_items)]

    def _assign_items_to_topics(self, topics, items):
        """Use the LLM to assign items to extracted topics."""
        try:
            topics_text = "\n".join(
                f"{index + 1}. {topic}"
                for index, topic in enumerate(topics)
            )

            items_text = ""
            for index, item in enumerate(items):
                title = item.get("title", "")
                snippet = item.get("content", "")[:200]
                items_text += (
                    f"Item {index + 1}: {title}\n"
                    f"Snippet: {snippet}...\n\n"
                )

            prompt = dedent(
                f"""
                Assign each item to the most relevant topic from the list
                below.

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
            ).strip()
            system_message = (
                "You are a professional analyst who categorizes "
                "information effectively."
            )
            assignments_text = self.llm_service.generate_response(
                prompt,
                system_message,
            )

            topic_pattern = r"Topic\s+(\d+):"
            item_pattern = r"Item\s+(\d+)"

            result = []
            current_topic = None
            current_items = []

            for line in assignments_text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                topic_match = re.search(topic_pattern, line)
                if topic_match:
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

            if current_topic is not None and current_items:
                result.append((current_topic, current_items))

            return result
        except Exception as error:
            self.logger.error("Error assigning items to topics: %s", error)
            return [(topics[0] if topics else "Search Results", items)]

    def _extract_topics_with_llm(self, query, relevant_items):
        """Extract grouping topics using the LLM."""
        try:
            content_for_topics = ""
            for item in relevant_items[:10]:
                title = item.get("title", "")
                content_for_topics += f"- {title}\n"

            current_date, current_time = self._current_date_time()
            prompt = dedent(
                f"""
                Based on the following titles, identify 3-5 main topics for
                grouping information related to the query: "{query}"

                Titles:
                {content_for_topics}

                Current date and time: {current_date} {current_time}

                Return only a list of topics without explanations, each topic
                on a new line.
                """
            ).strip()
            system_message = dedent(
                f"""
                You are a professional analyst who groups information by
                topics. Today is {current_date} at {current_time}.
                """
            ).strip()
            topics_text = self.llm_service.generate_response(
                prompt,
                system_message,
            )

            topics = [
                topic.strip()
                for topic in topics_text.split("\n")
                if topic.strip()
            ]
            topics = [
                re.sub(r"^[\d\-\*\.\s]+", "", topic).strip()
                for topic in topics
            ]
            return [topic for topic in topics if topic]
        except Exception as error:
            self.logger.error("Error extracting topics: %s", error)
            return []

    def _current_date_time(self):
        """Return the current date and time as formatted strings."""
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%Y-%m-%d")
        current_time = current_datetime.strftime("%H:%M:%S")
        return current_date, current_time
