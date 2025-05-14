import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
from typing import Dict, List, Any

class LiteraryVisualizations:
    """Provides visualization tools for literary analysis."""
    
    def __init__(self):
        """Initialize the visualization component."""
        pass
    
    def character_network(self, character_data):
        """Create a character relationship network visualization.
        
        Args:
            character_data: Dictionary of character information including relationships
            
        Returns:
            plotly.graph_objects.Figure: Network visualization
        """
        # Create a graph
        G = nx.Graph()
        
        # Sort characters by their original importance to maintain relative ranking
        sorted_characters = sorted(
            character_data.items(),
            key=lambda x: x[1].get('importance', 5),
            reverse=True
        )
        
        # Redistribute importance values between 3 and 8
        total_chars = len(sorted_characters)
        if total_chars > 0:
            # Calculate importance step size
            step = 5.0 / (total_chars - 1) if total_chars > 1 else 0
            
            # Create new character data with redistributed importance
            redistributed_data = {}
            for i, (char, info) in enumerate(sorted_characters):
                # Calculate new importance value between 3 and 8
                new_importance = 8 - (i * step)
                # Create new info dict with updated importance
                new_info = info.copy()
                new_info['importance'] = new_importance
                redistributed_data[char] = new_info
                # Add node with new importance
                G.add_node(char, size=new_importance)
        else:
            # If no characters, use original data
            for character, info in character_data.items():
                importance = info.get('importance', 5)
                G.add_node(character, size=importance)
        
        # Add edges (relationships)
        for character, info in character_data.items():
            if 'relationships' in info:
                for related_char, relationship in info['relationships'].items():
                    if related_char in character_data:  # Ensure the related character exists
                        # Add edge with relationship as attribute
                        G.add_edge(character, related_char, relationship=relationship)
        
        # Get node positions using a layout algorithm
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            relationship = G.edges[edge].get('relationship', '')
            # Truncate relationship to 2-3 words if it's too long
            relationship_words = relationship.split()
            if len(relationship_words) > 3:
                relationship = ' '.join(relationship_words[:3]) + '...'
            edge_text.append(relationship)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines+text',
            textposition='middle center',
            textfont=dict(size=10, color='#666'),
            hoverlabel=dict(bgcolor='white', font_size=12)
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        # Calculate size range based on importance values
        min_size = 20  # Minimum node size
        max_size = 60  # Maximum node size
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create hover text with character information
            char_info = redistributed_data.get(node, character_data[node])
            hover_text = f"<b>{node}</b><br>"
            if 'traits' in char_info:
                hover_text += f"Traits: {', '.join(char_info['traits'])}<br>"
            if 'development' in char_info:
                hover_text += f"Development: {char_info['development']}<br>"
            if 'importance' in char_info:
                hover_text += f"Importance: {char_info['importance']:.1f}/10"
            
            node_text.append(hover_text)
            
            # Calculate node size based on importance
            importance = G.nodes[node]['size']
            # Scale size between min_size and max_size based on importance (3-8)
            size = min_size + (max_size - min_size) * (importance - 3) / 5
            node_size.append(size)
            node_color.append(importance)  # Color based on importance
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],  # Show character names
            textposition="top center",
            textfont=dict(size=12, color='black'),
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white'),
                colorbar=dict(
                    thickness=15,
                    title='Character<br>Importance',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Character Relationship Network',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='rgba(255,255,255,1)'
                        ))
        
        # Add annotations for relationship descriptions
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            relationship = G.edges[edge].get('relationship', '')
            # Truncate relationship to 2-3 words if it's too long
            relationship_words = relationship.split()
            if len(relationship_words) > 3:
                relationship = ' '.join(relationship_words[:3]) + '...'
            
            # Calculate midpoint for annotation
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=relationship,
                showarrow=False,
                font=dict(size=10, color='#666'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#888',
                borderwidth=1,
                borderpad=4
            )
        
        return fig
    
    def theme_radar_chart(self, theme_data):
        """Create a radar chart of themes and their importance.
        
        Args:
            theme_data: Dictionary of theme information
            
        Returns:
            plotly.graph_objects.Figure: Radar chart
        """
        # Extract theme names and importance values
        themes = list(theme_data.keys())
        importance = [theme_data[theme].get('importance', 5) for theme in themes]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=importance,
            theta=themes,
            fill='toself',
            name='Theme Importance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=False,
            title='Theme Analysis'
        )
        
        return fig
    
    def symbol_bubble_chart(self, symbol_data):
        """Create a bubble chart of symbols and their significance.
        
        Args:
            symbol_data: Dictionary of symbol information
            
        Returns:
            plotly.graph_objects.Figure: Bubble chart
        """
        # Extract data
        symbols = list(symbol_data.keys())
        significance = [symbol_data[symbol].get('significance', 5) for symbol in symbols]
        occurrences = [symbol_data[symbol].get('occurrences', 1) for symbol in symbols]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Symbol': symbols,
            'Significance': significance,
            'Occurrences': occurrences
        })
        
        # Create bubble chart
        fig = px.scatter(
            df, x='Occurrences', y='Significance', size='Occurrences',
            hover_name='Symbol', text='Symbol',
            title='Symbol Analysis',
            labels={'Occurrences': 'Number of Occurrences', 'Significance': 'Symbolic Significance (1-10)'},
            size_max=60
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(yaxis_range=[0, 10])
        
        return fig
    
    def sentiment_timeline(self, text_chunks, sentiments):
        """Create a sentiment timeline visualization.
        
        Args:
            text_chunks: List of text chunks
            sentiments: List of sentiment scores for each chunk
            
        Returns:
            plotly.graph_objects.Figure: Line chart
        """
        # Create DataFrame
        df = pd.DataFrame({
            'Chunk': list(range(1, len(text_chunks) + 1)),
            'Sentiment': sentiments,
            'Text': text_chunks
        })
        
        # Create line chart
        fig = px.line(
            df, x='Chunk', y='Sentiment',
            title='Emotional Arc of the Text',
            labels={'Chunk': 'Text Progression', 'Sentiment': 'Emotional Tone (-1 to 1)'},
            hover_data=['Text']
        )
        
        # Add reference line for neutral sentiment
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Add annotations for extreme points
        max_idx = df['Sentiment'].idxmax()
        min_idx = df['Sentiment'].idxmin()
        
        fig.add_annotation(
            x=df.loc[max_idx, 'Chunk'],
            y=df.loc[max_idx, 'Sentiment'],
            text="Peak",
            showarrow=True,
            arrowhead=1
        )
        
        fig.add_annotation(
            x=df.loc[min_idx, 'Chunk'],
            y=df.loc[min_idx, 'Sentiment'],
            text="Valley",
            showarrow=True,
            arrowhead=1
        )
        
        return fig
    
    def create_study_guide_ui(self, study_guide_data):
        """Create an interactive study guide UI.
        
        Args:
            study_guide_data: Dictionary containing study guide information
        """
        st.header("Interactive Study Guide")
        
        # Create tabs for different sections
        tabs = st.tabs(["Summary", "Characters", "Themes", "Symbols", "Discussion Questions", "Key Passages"])
        
        # Summary tab
        with tabs[0]:
            st.subheader("Summary")
            st.write(study_guide_data.get("Summary", "No summary available."))
        
        # Characters tab
        with tabs[1]:
            st.subheader("Key Characters")
            characters = study_guide_data.get("Key Characters", [])
            
            if characters:
                # Handle both dictionary and list formats
                if isinstance(characters, dict):
                    # If it's a dictionary, iterate through items
                    for char_name, char_info in characters.items():
                        with st.expander(char_name):
                            st.write(f"**Description:** {char_info.get('description', 'No description available.')}")
                            st.write(f"**Significance:** {char_info.get('significance', 'No significance information available.')}")
                else:
                    # If it's a list, each item should be a character dictionary
                    for char in characters:
                        char_name = char.get('name', f"Character {characters.index(char) + 1}")
                        with st.expander(char_name):
                            st.write(f"**Description:** {char.get('description', 'No description available.')}")
                            st.write(f"**Significance:** {char.get('significance', 'No significance information available.')}")
                            
            else:
                st.write("No character information available.")
        
        # Themes tab
        with tabs[2]:
            st.subheader("Major Themes")
            themes = study_guide_data.get("Major Themes", [])
            
            if themes:
                # Handle both dictionary and list formats
                if isinstance(themes, dict):
                    # If it's a dictionary, iterate through items
                    for theme_name, theme_info in themes.items():
                        with st.expander(theme_name):
                            st.write(f"**Explanation:** {theme_info.get('explanation', 'No explanation available.')}")
                            st.write(f"**Textual Evidence:** {theme_info.get('textual evidence', 'No evidence available.')}")
                else:
                    # If it's a list, each item should be a theme dictionary
                    for theme in themes:
                        theme_name = theme.get('name', f"Theme {themes.index(theme) + 1}")
                        with st.expander(theme_name):
                            st.write(f"**Explanation:** {theme.get('explanation', 'No explanation available.')}")
                            st.write(f"**Textual Evidence:** {theme.get('textual evidence', 'No evidence available.')}")
                            
            else:
                st.write("No theme information available.")
        
        # Symbols tab
        with tabs[3]:
            st.subheader("Important Symbols")
            symbols = study_guide_data.get("Important Symbols", [])
            
            if symbols:
                # Handle both dictionary and list formats
                if isinstance(symbols, dict):
                    # If it's a dictionary, iterate through items
                    for symbol_name, symbol_info in symbols.items():
                        with st.expander(symbol_name):
                            st.write(f"**Meaning:** {symbol_info.get('meaning', 'No meaning available.')}")
                            st.write(f"**Occurrences:** {symbol_info.get('occurrences', 'No occurrence information available.')}")
                else:
                    # If it's a list, each item should be a symbol dictionary
                    for symbol in symbols:
                        symbol_name = symbol.get('name', f"Symbol {symbols.index(symbol) + 1}")
                        with st.expander(symbol_name):
                            st.write(f"**Meaning:** {symbol.get('meaning', 'No meaning available.')}")
                            st.write(f"**Occurrences:** {symbol.get('occurrences', 'No occurrence information available.')}")
                            
            else:
                st.write("No symbol information available.")
        
        # Discussion Questions tab
        with tabs[4]:
            st.subheader("Discussion Questions")
            questions = study_guide_data.get("Discussion Questions", [])
            
            if questions:
                for i, question in enumerate(questions):
                    st.write(f"{i+1}. {question}")
            else:
                st.write("No discussion questions available.")
        
        # Key Passages tab
        with tabs[5]:
            st.subheader("Key Passages")
            passages = study_guide_data.get("Key Passages", [])
            
            if passages:
                for i, passage in enumerate(passages):
                    with st.expander(f"Passage {i+1}"):
                        st.write(f"**Text:** {passage.get('passage text', 'No text available.')}")
                        st.write(f"**Location:** {passage.get('page/location', 'No location information available.')}")
                        st.write(f"**Significance:** {passage.get('significance', 'No significance information available.')}")
            else:
                st.write("No key passages available.")
    
    def display_reading_progress(self, progress_data):
        """Display reading progress with insights.
        
        Args:
            progress_data: Dictionary containing progress information and insights
        """
        st.header("Reading Progress")
        
        # Display progress bar
        progress_percentage = progress_data.get("progress_percentage", 0)
        st.progress(progress_percentage / 100)
        st.write(f"Progress: {progress_percentage:.1f}%")
        
        # Display insights
        insights = progress_data.get("insights", {})
        
        if insights:
            with st.expander("Current Scene Summary", expanded=True):
                st.write(insights.get("Current scene summary", "No summary available."))
            
            with st.expander("Active Characters"):
                active_chars = insights.get("Active characters", [])
                if active_chars:
                    for char in active_chars:
                        st.write(f"- {char}")
                else:
                    st.write("No character information available.")
            
            with st.expander("Important Elements"):
                elements = insights.get("Important elements to pay attention to", [])
                if elements:
                    for element in elements:
                        st.write(f"- {element}")
                else:
                    st.write("No important elements information available.")
            
            with st.expander("Questions to Consider"):
                questions = insights.get("Questions to consider while reading this section", [])
                if questions:
                    for i, question in enumerate(questions):
                        st.write(f"{i+1}. {question}")
                else:
                    st.write("No questions available.")
        else:
            st.write("No reading insights available.")
