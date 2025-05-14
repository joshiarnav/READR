import requests
import wikipedia
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st

class KnowledgeRetriever:
    """Retrieves external knowledge to enhance literary understanding."""
    
    def __init__(self):
        """Initialize the knowledge retriever."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        # Initialize OpenAI embeddings with minimal parameters to avoid compatibility issues
        self.embeddings = OpenAIEmbeddings()
        self.cache = {}
    
    def search_wikipedia(self, query, max_results=3):
        """Search Wikipedia for relevant information.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            list: List of relevant Wikipedia articles
        """
        try:
            # Check cache first
            cache_key = f"wiki_{query}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Search Wikipedia
            search_results = wikipedia.search(query, results=max_results)
            articles = []
            
            for title in search_results:
                try:
                    # Get article content
                    page = wikipedia.page(title)
                    summary = page.summary
                    content = page.content[:5000]  # Limit content length
                    
                    articles.append({
                        "title": title,
                        "summary": summary,
                        "content": content,
                        "url": page.url,
                        "source": "wikipedia"
                    })
                except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                    continue
            
            # Cache results
            self.cache[cache_key] = articles
            return articles
        except Exception as e:
            st.error(f"Error searching Wikipedia: {str(e)}")
            return []
    
    def fetch_literary_analysis(self, title, author=None):
        """Fetch literary analysis from trusted academic sources.
        
        Args:
            title: Title of the literary work
            author: Author of the literary work
            
        Returns:
            list: List of literary analyses
        """
        try:
            # Check cache first
            cache_key = f"lit_{title}_{author}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Construct search query
            query = f"{title}"
            if author:
                query += f" {author} literary analysis"
            else:
                query += " literary analysis"
            
            # Use Wikipedia as a source of literary analysis
            analyses = self.search_wikipedia(query)
            
            # Cache results
            self.cache[cache_key] = analyses
            return analyses
        except Exception as e:
            st.error(f"Error fetching literary analysis: {str(e)}")
            return []
    
    def fetch_historical_context(self, period, event=None):
        """Fetch historical context information.
        
        Args:
            period: Historical period
            event: Specific historical event
            
        Returns:
            list: List of historical context information
        """
        try:
            # Check cache first
            cache_key = f"hist_{period}_{event}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Construct search query
            query = f"{period} historical context"
            if event:
                query += f" {event}"
            
            # Use Wikipedia as a source of historical context
            context_info = self.search_wikipedia(query)
            
            # Cache results
            self.cache[cache_key] = context_info
            return context_info
        except Exception as e:
            st.error(f"Error fetching historical context: {str(e)}")
            return []
    
    def create_knowledge_base(self, information_list):
        """Create a vector store from retrieved information.
        
        Args:
            information_list: List of information dictionaries
            
        Returns:
            Chroma: Vector store containing knowledge chunks
        """
        texts = []
        metadatas = []
        
        for info in information_list:
            # Combine title, summary, and content
            full_text = f"Title: {info['title']}\n\nSummary: {info['summary']}\n\nContent: {info['content']}"
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(full_text)
            
            # Create metadata for each chunk
            for i, _ in enumerate(chunks):
                texts.append(chunks[i])
                metadatas.append({
                    "title": info["title"],
                    "source": info["source"],
                    "url": info.get("url", ""),
                    "chunk_id": i
                })
        
        # Create vector store
        if texts:
            vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            return vectorstore
        else:
            return None
