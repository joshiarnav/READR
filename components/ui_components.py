import streamlit as st

def setup_ui():
    """Set up the UI components for the READR application."""
    # Set up the main title and description
    st.title("📚 READR: Literary Companion")
    st.markdown(
        """
        *Your intelligent reading companion for classic literature.*
        
        READR uses advanced AI to provide deeper insights into literary works, 
        helping you understand historical context, symbolism, character motivations, and more.
        """
    )
    
    # Set up sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/storytelling.png", width=100)
        st.markdown("## How to use READR")
        st.markdown(
            """
            1. Upload a literary text (PDF or TXT)
            2. Ask questions about the text
            3. Explore different types of analysis
            
            **Example questions:**
            - What are the main themes in this text?
            - Explain the historical context of this story.
            - What does the [symbol] represent?
            - Analyze the character development of [character].
            - How does this work reflect the author's life?
            """
        )
        
        # Add about section
        st.markdown("---")
        st.markdown("### About READR")
        st.markdown(
            """
            READR uses Retrieval-Augmented Generation (RAG) and a multi-agent system 
            to provide comprehensive literary analysis and context.
            """
        )

def display_response(response, container):
    """Display a response in the given container.
    
    Args:
        response: Response text
        container: Streamlit container
    """
    container.markdown(response)

def display_document_info(document_processor, uploaded_files):
    """Display information about processed documents.
    
    Args:
        document_processor: DocumentProcessor instance
        uploaded_files: List of uploaded files
    """
    with st.expander("Document Information", expanded=False):
        for i, file in enumerate(uploaded_files):
            st.markdown(f"**Document {i+1}: {file.name}**")
            
            # Display file metadata
            st.markdown(f"Type: {file.type}")
            st.markdown(f"Size: {file.size} bytes")
            
            # Display document statistics
            if file.type == "application/pdf":
                pdf_reader = document_processor.extract_pdf_text(file)
                st.markdown(f"Pages: {len(pdf_reader.pages)}")
            
            st.markdown("---")

def display_analysis_options():
    """Display analysis options in the sidebar.
    
    Returns:
        str: Selected analysis type
    """
    with st.sidebar:
        st.header("Analysis Options")
        analysis_type = st.selectbox(
            "Select analysis focus:",
            ["General", "Historical Context", "Character Analysis", "Symbolism", "Themes", "Literary Techniques"]
        )
        
        return analysis_type
