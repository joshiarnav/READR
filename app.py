import streamlit as st
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Import components
from components.document_processor import DocumentProcessor
from components.agent_system import AgentSystem
from components.knowledge_retriever import KnowledgeRetriever
from components.ui_components import setup_ui, display_response
from components.advanced_features import AdvancedFeatures
from components.visualizations import LiteraryVisualizations
from components.text_retriever import TextRetriever

# Set page configuration
st.set_page_config(page_title="READR: Literary Companion", layout="wide")

# Setup UI components
setup_ui()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to READR! Please enter a book or literary source to begin exploring."}]

if "document_processor" not in st.session_state:
    st.session_state["document_processor"] = DocumentProcessor()

if "knowledge_retriever" not in st.session_state:
    st.session_state["knowledge_retriever"] = KnowledgeRetriever()

if "agent_system" not in st.session_state:
    st.session_state["agent_system"] = AgentSystem()
    
if "advanced_features" not in st.session_state:
    st.session_state["advanced_features"] = AdvancedFeatures()
    
if "visualizations" not in st.session_state:
    st.session_state["visualizations"] = LiteraryVisualizations()
    
if "text_retriever" not in st.session_state:
    st.session_state["text_retriever"] = TextRetriever()
    
if "current_tab" not in st.session_state:
    st.session_state["current_tab"] = "Chat"
    
if "study_guide" not in st.session_state:
    st.session_state["study_guide"] = None
    
if "visualization_data" not in st.session_state:
    st.session_state["visualization_data"] = None
    
if "reading_position" not in st.session_state:
    st.session_state["reading_position"] = 0

if "combined_text" not in st.session_state:
    st.session_state["combined_text"] = ""

if "source_metadata" not in st.session_state:
    st.session_state["source_metadata"] = {}

if "knowledge_base" not in st.session_state:
    st.session_state["knowledge_base"] = None

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Visualizations", "Study Guide", "Reading Mode"])

# Text input in sidebar
with st.sidebar:
    st.header("Enter Literary Source")
    source_query = st.text_input(
        "Enter book title or literary source",
        placeholder="e.g., The Great Gatsby, Moby Dick, etc."
    )
    
    # Analysis options
    st.header("Analysis Options")
    analysis_type = st.selectbox(
        "Select analysis focus:",
        ["General", "Historical Context", "Character Analysis", "Symbolism", "Themes", "Literary Techniques"]
    )
    st.session_state["analysis_type"] = analysis_type
    
    # Advanced options
    st.header("Advanced Features")
    generate_study_guide = st.button("Generate Study Guide", disabled=not st.session_state.get("combined_text"))
    generate_visualizations = st.button("Generate Visualizations", disabled=not st.session_state.get("combined_text"))

# Process text retrieval when source query changes
if source_query and source_query != st.session_state.get("last_query", ""):
    st.session_state["last_query"] = source_query
    
    # Reset all state variables for the new book
    st.session_state["messages"] = [{"role": "assistant", "content": f"Welcome to READR! I've loaded {source_query}. What would you like to know about it?"}]
    st.session_state["study_guide"] = None
    st.session_state["visualization_data"] = None
    st.session_state["reading_position"] = 0
    st.session_state["combined_text"] = ""
    st.session_state["source_metadata"] = {}
    st.session_state["knowledge_base"] = None
    
    # Retrieve text from external sources
    with st.spinner("Retrieving content..."):
        text_retriever = st.session_state["text_retriever"]
        content, metadata = text_retriever.get_text(source_query)
        
        if content:
            # Process the retrieved text
            document_processor = st.session_state["document_processor"]
            knowledge_base = document_processor.process_text(content, metadata)
            st.session_state["knowledge_base"] = knowledge_base
            st.session_state["combined_text"] = content
            st.session_state["source_metadata"] = metadata
            
            # Display source information
            st.sidebar.success(f"Retrieved content from: {', '.join(metadata['sources'])}")
        else:
            st.sidebar.error("No content found for the given source. Please try a different search term.")
            st.session_state["combined_text"] = ""
            st.session_state["source_metadata"] = {}
            st.session_state["knowledge_base"] = None

# Handle study guide generation
if generate_study_guide and st.session_state.get("combined_text"):
    with st.spinner("Generating comprehensive study guide..."):
        advanced_features = st.session_state["advanced_features"]
        study_guide = advanced_features.generate_study_guide(st.session_state["combined_text"])
        st.session_state["study_guide"] = study_guide
        st.session_state["current_tab"] = "Study Guide"
        st.rerun()

# Handle visualization generation
if generate_visualizations and st.session_state.get("combined_text"):
    with st.spinner("Analyzing text and generating visualizations..."):
        advanced_features = st.session_state["advanced_features"]
        visualization_data = advanced_features.generate_literary_visualizations(
            st.session_state["combined_text"], 
            st.session_state.get("analysis_type", "General")
        )
        st.session_state["visualization_data"] = visualization_data
        st.session_state["current_tab"] = "Visualizations"
        st.rerun()

# Tab 1: Chat Interface
with tab1:
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Chat input
    question = st.chat_input(
        "Ask about the literary text...",
        disabled=not st.session_state.get("combined_text"),
    )
    
    # Process question
    if question and st.session_state.get("combined_text"):
        # Add user question to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        
        # Get response from agent system
        agent_system = st.session_state["agent_system"]
        knowledge_retriever = st.session_state["knowledge_retriever"]
        knowledge_base = st.session_state["knowledge_base"]
        analysis_type = st.session_state.get("analysis_type", "General")
        
        # Display streaming response
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            # Generate response
            response = agent_system.generate_response(
                question=question,
                knowledge_base=knowledge_base,
                knowledge_retriever=knowledge_retriever,
                analysis_type=analysis_type,
                chat_history=st.session_state.messages,
                response_container=response_container
            )
            
            # Add response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Tab 2: Visualizations
with tab2:
    st.header("Literary Analysis Visualizations")
    
    if not source_query:
        st.info("Please enter a literary source to generate visualizations.")
    elif st.session_state["visualization_data"] is None:
        st.info("Click 'Generate Visualizations' in the sidebar to analyze the text.")
    else:
        visualization_data = st.session_state["visualization_data"]
        visualizations = st.session_state["visualizations"]
        
        # Display visualizations based on analysis type
        analysis_type = st.session_state.get("analysis_type", "General")
        
        if analysis_type == "Character Analysis":
            st.subheader("Character Network")
            try:
                if "characters" in visualization_data:
                    character_fig = visualizations.character_network(visualization_data["characters"])
                    st.plotly_chart(character_fig, use_container_width=True)
                else:
                    st.warning("No character data available for visualization.")
            except Exception as e:
                st.error(f"Error generating character network: {str(e)}")
        
        elif analysis_type == "Themes":
            st.subheader("Theme Analysis")
            try:
                if "themes" in visualization_data:
                    theme_fig = visualizations.theme_radar_chart(visualization_data["themes"])
                    st.plotly_chart(theme_fig, use_container_width=True)
                else:
                    st.warning("No theme data available for visualization.")
            except Exception as e:
                st.error(f"Error generating theme analysis: {str(e)}")
        
        elif analysis_type == "Symbolism":
            st.subheader("Symbol Analysis")
            try:
                if "symbols" in visualization_data:
                    symbol_fig = visualizations.symbol_bubble_chart(visualization_data["symbols"])
                    st.plotly_chart(symbol_fig, use_container_width=True)
                else:
                    st.warning("No symbol data available for visualization.")
            except Exception as e:
                st.error(f"Error generating symbol analysis: {str(e)}")
        
        else:  # General analysis
            # Try to display all available visualizations
            if "characters" in visualization_data:
                st.subheader("Character Network")
                try:
                    character_fig = visualizations.character_network(visualization_data["characters"])
                    st.plotly_chart(character_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating character network: {str(e)}")
            
            if "themes" in visualization_data:
                st.subheader("Theme Analysis")
                try:
                    theme_fig = visualizations.theme_radar_chart(visualization_data["themes"])
                    st.plotly_chart(theme_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating theme analysis: {str(e)}")
            
            if "symbols" in visualization_data:
                st.subheader("Symbol Analysis")
                try:
                    symbol_fig = visualizations.symbol_bubble_chart(visualization_data["symbols"])
                    st.plotly_chart(symbol_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating symbol analysis: {str(e)}")
            
            if not any(key in visualization_data for key in ["characters", "themes", "symbols"]):
                st.warning("No visualization data available. Try selecting a specific analysis type.")

# Tab 3: Study Guide
with tab3:
    if not source_query:
        st.info("Please enter a literary source to generate a study guide.")
    elif st.session_state["study_guide"] is None:
        st.info("Click 'Generate Study Guide' in the sidebar to create a comprehensive study guide.")
    else:
        study_guide_data = st.session_state["study_guide"]
        visualizations = st.session_state["visualizations"]
        visualizations.create_study_guide_ui(study_guide_data)

# Tab 4: Reading Mode
with tab4:
    st.header("Reading Mode")
    
    if not source_query:
        st.info("Please enter a literary source to use Reading Mode.")
    else:
        combined_text = st.session_state["combined_text"]
        source_metadata = st.session_state.get("source_metadata", {})
        
        # Create a container for the main content
        main_content = st.container()
        
        with main_content:
            # Display source information
            if source_metadata:
                st.subheader("Source Information")
                if "wikipedia" in source_metadata:
                    wiki = source_metadata["wikipedia"]
                    st.markdown(f"**Wikipedia:** [{wiki['title']}]({wiki['url']})")
                    st.markdown(wiki["summary"])
                
                if "google_books" in source_metadata:
                    books = source_metadata["google_books"]
                    st.markdown(f"**Google Books:** [{books['title']}]({books['preview_url']})")
                    if books.get("authors"):
                        st.markdown(f"Authors: {', '.join(books['authors'])}")
            
            # Display text with progress tracking
            st.subheader("Text")
            text_container = st.container()
            with text_container:
                # Calculate the visible portion of text based on reading position
                chunk_size = 5000  # Number of characters to show at once
                start_pos = max(0, st.session_state["reading_position"] - chunk_size // 2)
                end_pos = min(len(combined_text), start_pos + chunk_size)
                
                # Display the current chunk of text
                current_text = combined_text[start_pos:end_pos]
                st.markdown(current_text)
                
                # Add navigation buttons
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("Previous Section", disabled=start_pos == 0):
                        st.session_state["reading_position"] = max(0, start_pos - chunk_size)
                        st.rerun()
                with col3:
                    if st.button("Next Section", disabled=end_pos >= len(combined_text)):
                        st.session_state["reading_position"] = min(len(combined_text), end_pos)
                        st.rerun()
        
        # Create a fixed container at the bottom for progress tracking
        st.markdown("---")  # Add a separator
        progress_container = st.container()
        with progress_container:
            # Calculate progress percentage
            progress_percentage = (st.session_state["reading_position"] / len(combined_text)) * 100 if combined_text else 0
            
            # Display progress information
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write(f"Progress: {progress_percentage:.1f}%")
            with col2:
                # Create a more precise progress slider
                new_position = st.slider(
                    "Reading Position",
                    0,
                    len(combined_text),
                    st.session_state["reading_position"],
                    key="reading_position_slider",
                    label_visibility="collapsed"
                )
                if new_position != st.session_state["reading_position"]:
                    st.session_state["reading_position"] = new_position
                    st.rerun()
            with col3:
                if st.button("Get Reading Insights"):
                    with st.spinner("Analyzing your current reading position..."):
                        advanced_features = st.session_state["advanced_features"]
                        progress_data = advanced_features.track_reading_progress(combined_text, st.session_state["reading_position"])
                        visualizations = st.session_state["visualizations"]
                        visualizations.display_reading_progress(progress_data)
