#!/usr/bin/env python3
"""
LLMFlow Search Agent - Memory Management System
Memory Module - Manages short-term and long-term memory for the agent.
Stores and retrieves information from past queries and searches.
"""
import json
import os
import time
import logging
from datetime import datetime

class MemoryModule:
    def __init__(self, memory_path="./memory"):
        """
        Initialize the memory module.
        
        Args:
            memory_path: Path to the directory for storing persistent memory
        """
        self.short_term = []  # In-memory storage for current session
        self.links = {}  # Dictionary of URL -> title for sources
        
        # Create memory directory if it doesn't exist
        self.memory_path = memory_path
        os.makedirs(memory_path, exist_ok=True)
        
        # Path to long-term memory file
        self.long_term_file = os.path.join(memory_path, "long_term_memory.json")
        
        # Load long-term memory
        self.long_term = self._load_long_term()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Memory module initialized with path: {memory_path}")
    
    def add_to_short_term(self, item):
        """
        Add an item to short-term memory.
        
        Args:
            item: Dictionary containing the memory item
        """
        if not isinstance(item, dict):
            raise TypeError("Memory item must be a dictionary")
        
        # Ensure item has a timestamp
        if 'timestamp' not in item:
            item['timestamp'] = time.time()
        
        self.short_term.append(item)
        self.logger.debug(f"Added item to short-term memory: {item.get('type')}")
    
    def get_short_term(self):
        """
        Get all items from short-term memory.
        
        Returns:
            List of memory items
        """
        return self.short_term
    
    def add_to_long_term(self, item):
        """
        Add an item to long-term memory.
        
        Args:
            item: Dictionary containing the memory item
        """
        if not isinstance(item, dict):
            raise TypeError("Memory item must be a dictionary")
        
        # Ensure item has a timestamp
        if 'timestamp' not in item:
            item['timestamp'] = time.time()
        
        self.long_term.append(item)
        self._save_long_term()
        
        self.logger.debug(f"Added item to long-term memory: {item.get('type')}")
    
    def get_long_term(self):
        """
        Get all items from long-term memory.
        
        Returns:
            List of memory items
        """
        return self.long_term
    
    def add_to_links(self, url, title):
        """
        Add a source link.
        
        Args:
            url: URL of the source
            title: Title or description of the source
        """
        if not url:
            return
        
        self.links[url] = title
        self.logger.debug(f"Added link: {url}")
    
    def get_links(self):
        """
        Get all source links.
        
        Returns:
            Dictionary mapping URLs to titles
        """
        return self.links
    
    def clear_short_term(self):
        """Clear short-term memory."""
        self.short_term = []
        self.logger.debug("Short-term memory cleared")
    
    def clear_links(self):
        """Clear source links."""
        self.links = {}
        self.logger.debug("Links cleared")
    
    def _load_long_term(self):
        """
        Load long-term memory from file.
        
        Returns:
            List of memory items
        """
        if os.path.exists(self.long_term_file):
            try:
                with open(self.long_term_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning("Long-term memory file corrupted, creating new one")
                return []
            except Exception as e:
                self.logger.error(f"Error loading long-term memory: {str(e)}")
                return []
        
        return []
    
    def _save_long_term(self):
        """Save long-term memory to file."""
        try:
            with open(self.long_term_file, 'w', encoding='utf-8') as f:
                json.dump(self.long_term, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving long-term memory: {str(e)}")
    
    def get_relevant_content(self, query, max_items=10):
        """
        Get content relevant to the query.
        
        Args:
            query: The query to find relevant content for
            max_items: Maximum number of items to return
            
        Returns:
            List of relevant memory items
        """
        # In a production system, this would use embedding-based semantic search
        # For this implementation, we'll use a simple keyword-based approach
        
        query_words = set(query.lower().split())
        scored_items = []
        
        # Score short-term memory items
        for item in self.short_term:
            score = self._calculate_relevance_score(item, query_words)
            if score > 0:
                scored_items.append((score, item))
        
        # Sort by relevance score (descending)
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # Return the top items
        return [item for _, item in scored_items[:max_items]]
    
    def _calculate_relevance_score(self, item, query_words):
        """
        Calculate a simple relevance score for a memory item.
        
        Args:
            item: Memory item
            query_words: Set of words from the query
            
        Returns:
            Relevance score (higher is more relevant)
        """
        score = 0
        
        # Check content field
        if 'content' in item:
            content = str(item['content']).lower()
            for word in query_words:
                if word in content:
                    score += 1
        
        # Check title field
        if 'title' in item:
            title = str(item['title']).lower()
            for word in query_words:
                if word in title:
                    score += 2  # Title matches are more important
        
        # Parse results are more valuable
        if item.get('type') == 'parsed_content':
            score *= 2
        
        # Recency bonus (within the last hour)
        if time.time() - item.get('timestamp', 0) < 3600:
            score += 1
        
        return score
