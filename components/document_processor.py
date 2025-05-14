import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
import os
import json

class DocumentProcessor:
    """Handles document processing, chunking, and vectorization for the READR system."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        # Initialize OpenAI embeddings with minimal parameters to avoid compatibility issues
        self.embeddings = OpenAIEmbeddings()
    
    def extract_pdf_text(self, pdf_file):
        """Extract text from a PDF file.
        
        Args:
            pdf_file: The uploaded PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Avoid None issues
                text.append(page_text.strip())
        return "\n\n".join(text)
    
    def process_text(self, text: str, metadata: dict) -> Chroma:
        """Process text from external sources and create a vector store.
        
        Args:
            text: The text content to process
            metadata: Metadata about the text source
            
        Returns:
            Chroma: Vector store containing text chunks
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create metadata for each chunk
        chunk_metadata = []
        for i, _ in enumerate(chunks):
            # Create metadata with source information
            metadata = {
                "chunk_id": str(i),
                "source": metadata.get("sources", ["unknown"])[0],
                "title": metadata.get("wikipedia", {}).get("title") or metadata.get("google_books", {}).get("title", "Unknown"),
                "url": metadata.get("wikipedia", {}).get("url") or metadata.get("google_books", {}).get("preview_url", ""),
                "authors": ", ".join(metadata.get("google_books", {}).get("authors", [])) if metadata.get("google_books") else ""
            }
            chunk_metadata.append(metadata)
        
        # Create vector store
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            metadatas=chunk_metadata,
            persist_directory="./chroma_db"
        )
        
        return vectorstore
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents and create a vector store.
        
        Args:
            uploaded_files: List of uploaded files
            
        Returns:
            Chroma: Vector store containing document chunks
        """
        combined_text = ""
        metadata_list = []
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                file_content = self.extract_pdf_text(uploaded_file)
            else:  # Assume text file
                file_content = uploaded_file.read().decode("utf-8")
            
            # Add file metadata
            combined_text += f"\n\n--- File: {file_name} ---\n\n{file_content}"
            metadata_list.append({
                "filename": file_name,
                "content_type": uploaded_file.type,
                "size": len(file_content)
            })
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(combined_text)
        
        # Create metadata for each chunk
        chunk_metadata = []
        for i, _ in enumerate(chunks):
            # Create metadata with only simple types (strings, numbers, booleans)
            metadata = {
                "chunk_id": str(i),
                "source": "document",
                "document_count": str(len(uploaded_files)),
                # Convert complex metadata to a string representation
                "document_info": json.dumps([{"filename": m["filename"], "content_type": m["content_type"]} for m in metadata_list])
            }
            chunk_metadata.append(metadata)
        
        # Create vector store
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            metadatas=chunk_metadata,
            persist_directory="./chroma_db"
        )
        
        return vectorstore
