"""chatter
"""

import os
from typing import List, TypedDict, Annotated
from neo4j import GraphDatabase

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver

from .utils import is_ollama_running

from .agents import SupervisorAgent, SQLAgent, VectorAgent, FormatterAgent, HybridAgent
from .nodes import (
    QueryTranslatorNode, SupervisorNode, SQLQueryNode, VectorEmbeddingsQueryNode, HybridQueryNode, FormatterNode, RefuserNode
)


class AgentState(TypedDict):
    """Defines the structure of the state shared across various nodes in the workflow.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_question: str  
    translated_query: str

class MovieChatter:
    def __init__(self, prepared_area='', sql_db_fname='', embedding_store_fname='', embedding_model='',
                 graph_url=None, graph_username=None, graph_password=None, model_provider='',
                 model_name='', openai_api_key='', with_history=False, verbose=False, **kwargs):

        self.prepared_area = prepared_area
        self.sql_db_fname = sql_db_fname
        self.embedding_store_fname = embedding_store_fname
        self.embedding_model = embedding_model
        self.graph_url = graph_url
        self.graph_username = graph_username
        self.graph_password = graph_password
        self.model_provider = model_provider
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.with_history = with_history
        self.verbose = verbose

        # Placeholders for assignment in boot() to avoid IDE warnings.
        self.embedding_store_path = None
        self.sql_db_uri = None
        self.message_history = None
        self.llm = None
        self.embeddings = None
        self.supervisor_agent = None
        self.supervisor_node = None
        self.sql_agent = None
        self.sql_node = None
        self.vector_agent = None
        self.vector_node = None
        self.hybrid_agent = None
        self.hybrid_node = None
        self.graph_agent = None
        self.graph_node = None
        self.formatter_agent = None
        self.formatter_node = None
        self.refuser = None
        self.state_graph = None

        self.boot()

        return

    def boot(self):
        # Resolve relative prepared paths against this package directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        def _abs(p):
            return p if os.path.isabs(p) else os.path.abspath(os.path.join(base_dir, p))

        self.prepared_area = _abs(self.prepared_area)
        self.sql_db_uri = os.path.join(self.prepared_area, self.sql_db_fname)
        self.embedding_store_path = os.path.join(self.prepared_area, self.embedding_store_fname)
        self.message_history = ChatMessageHistory()

        # Set up LLM and embeddings based on provider
        if self.model_provider == 'openai':
            # Use API key from config if provided, otherwise fall back to environment variable
            api_key = self.openai_api_key if self.openai_api_key else None
            
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=0,
                api_key=api_key,
            )
            # OpenAI embeddings are the default for OpenAI provider
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=api_key,
            )
            
        elif self.model_provider == 'ollama':
            if is_ollama_running():
                self.llm = ChatOllama(
                    model=self.model_name,
                    temperature=0,
                    num_predict=-2,   # fill context
                    num_ctx=16384,    # context size for generating next token
                )
                # Auto-detect embedding backend to match prepared store
                if self.embedding_model.startswith('text-') or self.embedding_model.startswith('text-embedding-'):
                    self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
                else:
                    self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            else:
                raise RuntimeError('Ollama is not available.')
        else:
            raise RuntimeError(f"Invalid model_provider in configuration: '{self.model_provider}'. Must be 'openai' or 'ollama'.")

        # Graph functionality removed: skip Neo4j checks

        # Set up supervisor node.
        self.supervisor_agent = SupervisorAgent(self.llm, self.verbose)
        self.supervisor_node = SupervisorNode(self.supervisor_agent, self.verbose)

        # Set up SQL agent and graph node.
        self.sql_agent = SQLAgent(self.sql_db_uri, self.llm)
        self.sql_node = SQLQueryNode(self.sql_agent, verbose=self.verbose)

        # Set up vector store agent and graph node.
        self.vector_agent = VectorAgent(self.embedding_store_path, self.embeddings)
        self.vector_node = VectorEmbeddingsQueryNode(self.vector_agent, verbose=self.verbose)

        # Set up hybrid agent and node.
        self.hybrid_agent = HybridAgent(self.sql_db_uri, self.embedding_store_path, self.llm, self.embeddings)
        self.hybrid_node = HybridQueryNode(self.hybrid_agent, verbose=self.verbose)

        # Skip graph agent setup - for now but I'll see if I can add it back in later

        # Set up formatter node for the final response. -- ask yesim and basar if this is okay or if we wanna change it
        persona_directives = {
            'playful_matchmaker': (
                "You're that friend who always knows exactly what movie someone should watch next! "
                "You're genuinely excited about cinema and love helping people discover their new favorites. "
                "Keep it conversational and warm — like you're texting a friend, not writing a review. "
                "Throw in light humor when it feels natural, but never try too hard. "
                "Always spoiler-free unless they specifically ask otherwise."
            )
        }
        self.persona_instruction = persona_directives['playful_matchmaker']
        self.formatter_agent = FormatterAgent(self.llm, persona_directives=persona_directives)
        self.formatter_node = FormatterNode(self.formatter_agent, verbose=self.verbose)

        # Set up refuser node to refuse to answer.
        self.refuser = RefuserNode(self.llm, verbose=self.verbose)

        # Set up query translator node.
        self.query_translator_node = QueryTranslatorNode(self.llm, verbose=self.verbose)

        # Create the final state graph.
        self.state_graph = self._create_graph(
            self.query_translator_node,
            self.supervisor_node,
            self.sql_node,
            self.vector_node,
            self.hybrid_node,
            None,
            self.formatter_node,
            self.refuser,
        )

        return

    @staticmethod
    def _create_graph(query_translator_node, supervisor_node, sql_node, vector_node, hybrid_node, graph_node, formatter_node, refuser):
        def route_message(state):
            next_agent = END
            message = state['messages'][-1]
            if not isinstance(message, AIMessage):
                return END
            content = message.content
            if isinstance(content, str) and content.startswith('GOTO:'):
                next_agent = content.split('|')[0].replace('GOTO:', '').strip()

            return next_agent

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node('query_translator', query_translator_node)
        graph_builder.add_node('supervisor_node', supervisor_node)
        graph_builder.add_node('sql_agent', sql_node)
        graph_builder.add_node('vector_agent', vector_node)
        graph_builder.add_node('hybrid_agent', hybrid_node)
        # graph_builder.add_node('graph_agent', graph_node)
        graph_builder.add_node('formatter_agent', formatter_node)
        graph_builder.add_node('refuser', refuser)

        graph_builder.add_edge('query_translator', 'supervisor_node')
        graph_builder.add_conditional_edges(
            'supervisor_node',
            route_message, {
                'sql_agent': 'sql_agent',
                'vector_agent': 'vector_agent',
                'hybrid_agent': 'hybrid_agent',
                # 'graph_agent': 'graph_agent',
                'formatter_agent': 'formatter_agent',
                'refuser': 'refuser',
                END: END
            }
        )

        for agent in ['vector_agent', 'sql_agent', 'hybrid_agent']:
            graph_builder.add_edge(agent, 'formatter_agent')

        # graph_builder.add_edge('sql_agent', END)
        graph_builder.set_entry_point('query_translator')
        graph = graph_builder.compile(checkpointer=InMemorySaver())

        return graph

    def chat(self, user_input: str = None, thread_id: str = 'default_thread') -> str:
        if self.verbose:
            print(f'User input: {user_input}')

        user_msg = HumanMessage(content=user_input)
        
        # History handling, if with_history is True, use thread_id to maintain separate conversation threads
        # The InMemorySaver will automatically handle state persistence
        # If with_history is False, start fresh each time - no history
        # Use a unique thread_id each time to avoid history
        if self.with_history:
            inputs = {
                'messages': [user_msg],
                'user_question': user_input,
                'persona_instructions': self.persona_instruction,
            }
        else:
            thread_id = f"no_history_{hash(user_input)}_{id(user_input)}"
            inputs = {
                'messages': [user_msg],
                'user_question': user_input,
                'persona_instructions': self.persona_instruction,
            }
        
        result_state = self.state_graph.invoke(
            inputs, 
            {'recursion_limit': 25, 'configurable': {'thread_id': thread_id}}
        )

        reply = 'Unable to generate a response.'
        for message in reversed(result_state['messages']):
            if isinstance(message, AIMessage) and message.content.strip():
                reply = message.content.strip()
                break

        return reply
    
    def forget(self, thread_id: str = 'default_thread') -> str:
        """Clear conversation history for a specific thread."""
        try:
            # Since InMemorySaver doesn't have clear/delete methods, we access its internal storage
            if hasattr(self.state_graph, 'checkpointer'):
                checkpointer = self.state_graph.checkpointer
                # For now, we return a success message if the clear history is successful, but we may choose to not return anything.
                if hasattr(checkpointer, '_storage') and thread_id in checkpointer._storage:
                    del checkpointer._storage[thread_id]
                    return "Conversation history cleared successfully."
                elif hasattr(checkpointer, 'storage') and thread_id in checkpointer.storage:
                    del checkpointer.storage[thread_id]
                    return "Conversation history cleared successfully."
                else:
                    return "Conversation history cleared successfully."
            else:
                return "Conversation history cleared successfully."
        except Exception as e:
            return f"Error clearing conversation history: {str(e)}"
