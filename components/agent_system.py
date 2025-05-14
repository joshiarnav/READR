from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
# ChatOpenAI is already imported above
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
import streamlit as st

class AgentSystem:
    """Multi-agent system for literary analysis and context retrieval."""
    
    def __init__(self):
        """Initialize the agent system."""
        # Initialize LLM with minimal parameters to avoid compatibility issues
        self.llm = ChatOpenAI(temperature=0.7)
        
        # Create specialized agents
        self.reader_agent = self._create_reader_agent()
        self.context_agent = self._create_context_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.synthesis_agent = self._create_synthesis_agent()
    
    def _create_reader_agent(self):
        """Create an agent for reading and understanding the literary text."""
        # Define tools for the reader agent
        @tool
        def search_document(query):
            """Search the uploaded document for relevant information."""
            # This is a placeholder - actual implementation will use the knowledge base
            return "Information from document based on query: " + query
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Reader Agent specialized in understanding literary texts. "
                      "Your role is to carefully read and comprehend the uploaded document, "
                      "identifying key elements such as plot points, characters, settings, "
                      "and narrative structure. You provide factual information directly from the text."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        tools = [search_document]
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    def _create_context_agent(self):
        """Create an agent for retrieving historical and biographical context."""
        # Define tools for the context agent
        @tool
        def search_historical_context(period, event=None):
            """Search for historical context information."""
            # This is a placeholder - actual implementation will use the knowledge retriever
            return f"Historical context for {period} and {event if event else 'general'}"
        
        @tool
        def search_author_info(author_name):
            """Search for information about the author."""
            # This is a placeholder - actual implementation will use the knowledge retriever
            return f"Information about author: {author_name}"
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Context Agent specialized in providing historical, cultural, "
                      "and biographical context for literary works. You help readers understand "
                      "the time period, cultural influences, and author's background that shaped the text."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        tools = [search_historical_context, search_author_info]
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    def _create_analysis_agent(self):
        """Create an agent for literary analysis and critical perspectives."""
        # Define tools for the analysis agent
        @tool
        def analyze_symbolism(symbol, context):
            """Analyze symbolism in the literary text."""
            # This is a placeholder - actual implementation will use the LLM
            return f"Analysis of symbol '{symbol}' in context: {context}"
        
        @tool
        def analyze_themes(text_excerpt):
            """Identify and analyze themes in the literary text."""
            # This is a placeholder - actual implementation will use the LLM
            return f"Thematic analysis of: {text_excerpt}"
        
        @tool
        def analyze_character(character_name):
            """Analyze a character from the literary text."""
            # This is a placeholder - actual implementation will use the LLM
            return f"Character analysis of: {character_name}"
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an Analysis Agent specialized in literary criticism and analysis. "
                      "You identify and explain literary devices, symbolism, themes, character development, "
                      "and narrative techniques. You provide critical perspectives on the text."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        tools = [analyze_symbolism, analyze_themes, analyze_character]
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    def _create_synthesis_agent(self):
        """Create an agent for synthesizing insights from other agents."""
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Synthesis Agent that combines insights from multiple sources "
                      "to provide comprehensive responses about literary texts. You integrate "
                      "textual evidence, historical context, and critical analysis into cohesive, "
                      "insightful answers that deepen the reader's understanding and appreciation of the text."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        # Create chain using the latest LangChain API
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
    
    def _retrieve_relevant_context(self, question, knowledge_base, top_k=5):
        """Retrieve relevant context from the knowledge base.
        
        Args:
            question: User question
            knowledge_base: Vector store containing document chunks
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            str: Relevant context
        """
        if not knowledge_base:
            return ""
        
        # Retrieve relevant documents
        docs = knowledge_base.similarity_search(question, k=top_k)
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        return context
    
    def _retrieve_external_knowledge(self, question, knowledge_retriever, analysis_type):
        """Retrieve external knowledge based on the question and analysis type.
        
        Args:
            question: User question
            knowledge_retriever: Knowledge retriever instance
            analysis_type: Type of analysis
            
        Returns:
            str: External knowledge
        """
        # Extract potential entities from the question
        # This is a simplified approach - in a real system, we would use NER
        words = question.split()
        potential_entities = [word for word in words if len(word) > 3 and word[0].isupper()]
        
        external_info = []
        
        # Retrieve information based on analysis type
        if analysis_type == "Historical Context":
            # Search for historical context
            for entity in potential_entities:
                context_info = knowledge_retriever.fetch_historical_context(entity)
                if context_info:
                    external_info.extend(context_info)
        
        elif analysis_type == "Character Analysis" or analysis_type == "Symbolism" or analysis_type == "Themes":
            # Search for literary analysis
            for entity in potential_entities:
                analysis_info = knowledge_retriever.fetch_literary_analysis(entity)
                if analysis_info:
                    external_info.extend(analysis_info)
        
        else:  # General analysis
            # Search Wikipedia for general information
            wiki_info = knowledge_retriever.search_wikipedia(question)
            if wiki_info:
                external_info.extend(wiki_info)
        
        # Format external knowledge
        if external_info:
            # Create knowledge base from external info
            external_kb = knowledge_retriever.create_knowledge_base(external_info)
            
            # Retrieve relevant chunks
            if external_kb:
                docs = external_kb.similarity_search(question, k=3)
                external_knowledge = "\n\n".join([doc.page_content for doc in docs])
                return external_knowledge
        
        return ""
    
    def generate_response(self, question, knowledge_base, knowledge_retriever, analysis_type, chat_history, response_container):
        """Generate a response to the user's question using the multi-agent system.
        
        Args:
            question: User question
            knowledge_base: Vector store containing document chunks
            knowledge_retriever: Knowledge retriever instance
            analysis_type: Type of analysis
            chat_history: Conversation history
            response_container: Streamlit container for displaying the response
            
        Returns:
            str: Generated response
        """
        # Retrieve relevant context from the document
        document_context = self._retrieve_relevant_context(question, knowledge_base)
        
        # Retrieve external knowledge
        external_knowledge = self._retrieve_external_knowledge(question, knowledge_retriever, analysis_type)
        
        # Combine context
        combined_context = f"Document Context:\n{document_context}\n\nExternal Knowledge:\n{external_knowledge}"
        
        # Prepare inputs for agents
        reader_input = f"Based on the following document context, answer the question: {question}\n\nContext: {document_context}"
        context_input = f"Provide historical, cultural, or biographical context relevant to this question: {question}\n\nExternal Knowledge: {external_knowledge}"
        analysis_input = f"Analyze the literary elements related to this question: {question}\n\nDocument Context: {document_context}\n\nExternal Knowledge: {external_knowledge}"
        
        # Get responses from individual agents
        reader_response = self.reader_agent.invoke({"input": reader_input, "chat_history": chat_history})
        context_response = self.context_agent.invoke({"input": context_input, "chat_history": chat_history})
        analysis_response = self.analysis_agent.invoke({"input": analysis_input, "chat_history": chat_history})
        
        # Format agent responses
        reader_output = reader_response.get("output", "")
        context_output = context_response.get("output", "")
        analysis_output = analysis_response.get("output", "")
        
        # Prepare input for synthesis agent
        synthesis_input = f"Question: {question}\n\n"
        synthesis_input += f"Reader Agent (Text Information): {reader_output}\n\n"
        synthesis_input += f"Context Agent (Historical/Biographical Context): {context_output}\n\n"
        synthesis_input += f"Analysis Agent (Literary Analysis): {analysis_output}\n\n"
        synthesis_input += "Please synthesize these insights into a comprehensive, cohesive response that addresses the question."
        
        # Generate final response using the new chain format
        final_response = self.synthesis_agent.invoke({
            "input": synthesis_input,
            "chat_history": chat_history
        })
        
        # Update response container
        response_container.markdown(final_response)
        
        return final_response
