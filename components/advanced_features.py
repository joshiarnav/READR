import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import json
from typing import Dict, List, Any, Optional

class StreamHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a Streamlit app."""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token."""
        self.text += token
        self.container.markdown(self.text)

class AdvancedFeatures:
    """Implements advanced features for the READR application."""
    
    def __init__(self):
        """Initialize advanced features."""
        # Initialize LLM with API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                temperature=0.7,
                model="gpt-3.5-turbo",
                api_key=api_key
            )
    
    def create_conversational_chain(self, vectorstore):
        """Create a conversational retrieval chain for more natural interactions.
        
        Args:
            vectorstore: Vector store containing document chunks
            
        Returns:
            ConversationalRetrievalChain: Chain for conversational retrieval
        """
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create custom prompt for literary analysis
        prompt_template = """
        You are a literary scholar and expert on classic literature. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "chat_history", "question"]
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        
        return chain
    
    def generate_literary_visualizations(self, text, analysis_type):
        """Generate visualization data for literary analysis.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis
            
        Returns:
            dict: Visualization data
        """
        if not self.llm:
            st.error("LLM not initialized. Please check your OpenAI API key.")
            return {}
            
        # Prepare prompt based on analysis type
        if analysis_type == "Character Analysis":
            prompt = f"""Analyze the characters in the following text and generate a JSON object with:
            1. A "characters" key containing a dictionary where:
               - Each key is a character name
               - Each value is a dictionary with:
                 - "traits": list of character traits
                 - "relationships": dictionary of relationships with other characters
                 - "development": description of character development
                 - "importance": number from 1-10
            Text: {text[:2000]}...
            Example format:
            {{
                "characters": {{
                    "Captain Ahab": {{
                        "traits": ["obsessive", "determined", "tragic"],
                        "relationships": {{
                            "Ishmael": "narrator and observer",
                            "Moby Dick": "obsession and nemesis"
                        }},
                        "development": "Starts as respected captain, descends into obsession",
                        "importance": 10
                    }}
                }}
            }}
            Output only valid JSON without explanation."""
        
        elif analysis_type == "Themes":
            prompt = f"""Identify the main themes in the following text and generate a JSON object with:
            1. A "themes" key containing a dictionary where:
               - Each key is a theme name
               - Each value is a dictionary with:
                 - "description": detailed explanation of the theme
                 - "evidence": list of key examples from the text
                 - "importance": number from 1-10
            Text: {text[:2000]}...
            Example format:
            {{
                "themes": {{
                    "Revenge": {{
                        "description": "The destructive nature of revenge and obsession",
                        "evidence": ["Ahab's quest for revenge", "The final confrontation"],
                        "importance": 9
                    }}
                }}
            }}
            Output only valid JSON without explanation."""
        
        elif analysis_type == "Symbolism":
            prompt = f"""Identify symbols in the following text and generate a JSON object with:
            1. A "symbols" key containing a dictionary where:
               - Each key is a symbol name
               - Each value is a dictionary with:
                 - "meaning": detailed explanation of the symbol's meaning
                 - "occurrences": list of key occurrences in the text
                 - "significance": number from 1-10
            Text: {text[:2000]}...
            Example format:
            {{
                "symbols": {{
                    "The White Whale": {{
                        "meaning": "Represents the unknowable and nature's power",
                        "occurrences": ["First mentioned in Chapter 1", "Final confrontation"],
                        "significance": 10
                    }}
                }}
            }}
            Output only valid JSON without explanation."""
        
        else:  # General analysis
            prompt = f"""Perform a general literary analysis of the following text and generate a JSON object with:
            1. "characters": dictionary of character information
            2. "themes": dictionary of theme information
            3. "symbols": dictionary of symbol information
            Each section should follow the format shown in the previous examples.
            Text: {text[:2000]}...
            Output only valid JSON without explanation."""
        
        try:
            # Generate analysis
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            response_text = response.content
            # Find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end+1]
                visualization_data = json.loads(json_str)
            else:
                visualization_data = json.loads(response_text)
                
            # Ensure the data has the expected structure
            if analysis_type == "Character Analysis" and "characters" not in visualization_data:
                visualization_data = {"characters": visualization_data}
            elif analysis_type == "Themes" and "themes" not in visualization_data:
                visualization_data = {"themes": visualization_data}
            elif analysis_type == "Symbolism" and "symbols" not in visualization_data:
                visualization_data = {"symbols": visualization_data}
                
            return visualization_data
            
        except Exception as e:
            st.error(f"Error generating visualization data: {str(e)}")
            return {}
    
    def compare_literary_works(self, text1, text2):
        """Compare two literary texts and identify similarities and differences.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            dict: Comparison results
        """
        prompt = f"""Compare the following two literary texts and identify similarities and differences in:
        1. Themes
        2. Style
        3. Characters
        4. Setting
        5. Tone
        
        Text 1: {text1[:1000]}...
        
        Text 2: {text2[:1000]}...
        
        Generate a JSON object with these categories and detailed comparisons.
        Output only valid JSON without explanation."""
        
        # Generate comparison
        response = self.llm.invoke(prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            response_text = response.content
            # Find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end+1]
                comparison_data = json.loads(json_str)
            else:
                comparison_data = json.loads(response_text)
            return comparison_data
        except Exception as e:
            st.error(f"Error parsing comparison data: {str(e)}")
            return {}
    
    def generate_study_guide(self, text, title=None, author=None):
        """Generate a comprehensive study guide for a literary text.
        
        Args:
            text: Literary text
            title: Title of the work
            author: Author of the work
            
        Returns:
            dict: Study guide content
        """
        # Prepare context
        context = f"Title: {title}\nAuthor: {author}\n" if title and author else ""
        
        prompt = f"""Generate a comprehensive study guide for the following literary text:
        
        {context}Text: {text[:2000]}...
        
        Include the following sections in JSON format:
        1. Summary
        2. Key Characters (name, description, significance)
        3. Major Themes (name, explanation, textual evidence)
        4. Important Symbols (for each symbol, use its actual name/object as the key, not "Symbol 1", "Symbol 2", etc. Include meaning and occurrences)
        5. Discussion Questions (at least 5 thought-provoking questions)
        6. Key Passages (passage text, page/location, significance)
        7. Historical Context (relevant historical information)
        8. Critical Perspectives (different ways to interpret the text)
        
        For the Important Symbols section, use the actual symbol names as keys. For example:
        {{
            "Important Symbols": {{
                "The White Whale": {{
                    "meaning": "Represents the unknowable and nature's power",
                    "occurrences": "Appears throughout the novel as Ahab's obsession"
                }},
                "The Sea": {{
                    "meaning": "Symbolizes both life and death",
                    "occurrences": "Present in all major scenes"
                }}
            }}
        }}
        
        Output only valid JSON without explanation."""
        
        # Generate study guide
        response = self.llm.invoke(prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            response_text = response.content
            # Find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end+1]
                study_guide = json.loads(json_str)
            else:
                study_guide = json.loads(response_text)
            return study_guide
        except Exception as e:
            st.error(f"Error parsing study guide data: {str(e)}")
            return {}
    
    def track_reading_progress(self, text, current_position):
        """Track reading progress and provide contextual insights.
        
        Args:
            text: Full text
            current_position: Current reading position
            
        Returns:
            dict: Progress data and contextual insights
        """
        # Calculate progress percentage
        total_length = len(text)
        progress_percentage = (current_position / total_length) * 100 if total_length > 0 else 0
        
        # Get current context (text around current position)
        start_pos = max(0, current_position - 500)
        end_pos = min(total_length, current_position + 500)
        current_context = text[start_pos:end_pos]
        
        # Generate contextual insights
        prompt = f"""The reader is currently at this point in the text:
        
        {current_context}
        
        Generate contextual insights in JSON format including:
        1. Current scene summary
        2. Active characters
        3. Important elements to pay attention to
        4. Questions to consider while reading this section
        
        Output only valid JSON without explanation."""
        
        # Generate insights
        response = self.llm.invoke(prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            response_text = response.content
            # Find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end+1]
                insights = json.loads(json_str)
            else:
                insights = json.loads(response_text)
                
            # Combine progress data with insights
            progress_data = {
                "progress_percentage": progress_percentage,
                "current_position": current_position,
                "total_length": total_length,
                "insights": insights
            }
            
            return progress_data
        except Exception as e:
            st.error(f"Error parsing progress data: {str(e)}")
            return {
                "progress_percentage": progress_percentage,
                "current_position": current_position,
                "total_length": total_length,
                "insights": {}
            }
