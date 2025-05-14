import requests
import wikipedia
from bs4 import BeautifulSoup
import streamlit as st
from typing import Dict, Optional, Tuple
import re

class TextRetriever:
    """Retrieves text content from external sources like Wikipedia."""
    
    def __init__(self):
        """Initialize the text retriever."""
        self.cache = {}
        # Set Wikipedia language to English
        wikipedia.set_lang("en")
    
    def _clean_title(self, title: str) -> str:
        """Clean the title for better search results.
        
        Args:
            title: The title to clean
            
        Returns:
            str: Cleaned title
        """
        # Remove common prefixes/suffixes
        title = re.sub(r'^(the|a|an)\s+', '', title.lower())
        # Remove special characters
        title = re.sub(r'[^\w\s]', '', title)
        return title.strip()
    
    def search_wikipedia(self, query: str, max_results: int = 3) -> Tuple[str, Dict]:
        """Search Wikipedia for relevant information.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of (text content, metadata)
        """
        try:
            # Clean the query
            clean_query = self._clean_title(query)
            
            # Try different search variations
            search_variations = [
                clean_query,  # Original cleaned query
                f"{clean_query} novel",  # Add novel suffix
                f"{clean_query} book",  # Add book suffix
                query  # Original query with case preserved
            ]
            
            # Try each variation until we get a result
            for search_term in search_variations:
                try:
                    # Try direct page lookup first
                    page = wikipedia.page(search_term)
                    content = page.content
                    metadata = {
                        "source": "wikipedia",
                        "title": page.title,
                        "url": page.url,
                        "summary": page.summary
                    }
                    return content, metadata
                except wikipedia.exceptions.PageError:
                    continue
                except wikipedia.exceptions.DisambiguationError as e:
                    # If disambiguation page, try the first option
                    try:
                        page = wikipedia.page(e.options[0])
                        content = page.content
                        metadata = {
                            "source": "wikipedia",
                            "title": page.title,
                            "url": page.url,
                            "summary": page.summary
                        }
                        return content, metadata
                    except Exception:
                        continue
            
            # If direct lookups fail, try search
            for search_term in search_variations:
                search_results = wikipedia.search(search_term, results=max_results)
                if search_results:
                    try:
                        page = wikipedia.page(search_results[0])
                        content = page.content
                        metadata = {
                            "source": "wikipedia",
                            "title": page.title,
                            "url": page.url,
                            "summary": page.summary
                        }
                        return content, metadata
                    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                        continue
            
            return "", {"source": "wikipedia", "error": "No results found"}
            
        except Exception as e:
            st.error(f"Wikipedia error: {str(e)}")
            return "", {"source": "wikipedia", "error": str(e)}
    
    def get_text(self, query: str) -> Tuple[str, Dict]:
        """Get text content from Wikipedia.
        
        Args:
            query: Search query
            
        Returns:
            Tuple of (text content, metadata)
        """
        # Get content from Wikipedia
        content, metadata = self.search_wikipedia(query)
        
        if content:
            return content, {"sources": ["wikipedia"], "wikipedia": metadata}
        
        return "", {"sources": [], "error": "No content found from any source"} 