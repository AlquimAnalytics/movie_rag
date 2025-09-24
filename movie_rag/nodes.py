"""nodes
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

class QueryTranslatorNode:
    """Rewrites the user query clearly based on conversation history and context.
    """
    def __init__(self, llm, verbose=False):
        self.llm = llm
        self.verbose = verbose

        # Simple prompt for unambiguous queries
        self.simple_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query disambiguation specialist for a movie recommendation system. 
            Your job is to rewrite ambiguous queries to be explicit, or return the original unchanged.
            
            RULES:
            1. If the query is already clear and unambiguous, return it unchanged
            2. If it's a greeting, pleasantry, or non-query, return it unchanged
            3. If the query contains ambiguous references, rewrite it to be explicit
            4. Never generate answers or responses - only rewrite queries
            
            EXAMPLES:
            User: "Hello, how are you?"
            Rewritten: "Hello, how are you?"
            
            User: "How many movies are from 1999?"
            Rewritten: "How many movies are from 1999?"
            
            User: "Give me the plot of the first movie"
            Rewritten: "Give me the plot of the first movie"
            
            OUTPUT: Return only the rewritten query, nothing else."""),
            MessagesPlaceholder("history"),
            ("user", "{current_question}")
        ])

        # Complex prompt with structured reasoning for ambiguous queries
        self.complex_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query disambiguation specialist for a movie recommendation system. 
            Your job is to analyze user queries and rewrite them to be explicit and unambiguous.

            TASK: Rewrite ambiguous queries to be explicit, or return the original query if it's already clear.

            HISTORY HANDLING:
            - If conversation history is provided, use it to resolve ambiguous references
            - If no history exists, make reasonable assumptions based on movie recommendation patterns
            - Always preserve the original user intent while making references explicit

            RULES:
            1. If the query is unambiguous, return it unchanged
            2. If the query contains ambiguous references, resolve them using available context
            3. Always output ONLY the rewritten query - no explanations or additional text
            4. Preserve the original intent while making references explicit
            5. For greetings, pleasantries, or non-queries, return them unchanged
            6. Never generate answers or responses - only rewrite queries

            TYPES OF AMBIGUITY TO RESOLVE:
            - Positional references: "the first one", "the last movie", "item #3"
            - Comparative references: "the highest", "the most", "the best"
            - Temporal references: "this year", "last decade", "recently"
            - Entity references: "that actor", "the director", "the genre"
            - Follow-up questions: "what about that?", "tell me more", "and the others?", "which one is the highest rated?"

            EXAMPLES:

            Example 1 - Unambiguous Query:
            User: "How many movies are there in the database?"
            Rewritten: "How many movies are there in the database?"
             
            Example 2 - Unambiguous Query:
            User: "Which directors have made the most movies?"
            Rewritten: "Which directors have made the most movies?"

            Example 3 - Positional Reference (with history):
            History: User asked for sci-fi movies, got list with Blade Runner, The Matrix, Interstellar
            User: "Tell me about the first one"
            Rewritten: "Tell me about Blade Runner"

            Example 4 - Comparative Reference (with history):
            History: User asked for movie counts by genre, got Sci-fi: 45, Drama: 234, Comedy: 189
            User: "Which has the most?"
            Rewritten: "Which genre has the most movies among Sci-fi (45), Drama (234), and Comedy (189)?"
            
            Example 5 - Temporal Reference (with history):
            History: User asked about movies from 2020.
            User: "What about the previous year?"
            Rewritten: "What movies are from 2019?"
            
            Example 5b - Temporal Reference (with history):
            History: User asked about movies from the 1990s.
            User: "Show me an example from that decade"
            Rewritten: "Show me example movies from the 1990s"

            Example 6 - Entity Reference (with history):
            History: User asked about movies starring Leonardo DiCaprio
            User: "What about his frequent collaborator?"
            Rewritten: "What movies feature Leonardo DiCaprio's frequent collaborators?"

            Example 7 - No History Scenario:
            User: "Show me the latest movies"
            Rewritten: "Show me the most recently released movies"

            Example 8 - Follow-up Question (with history):
            History: User asked about Christopher Nolan movies, got 8 movies listed
            User: "How many are rated above 8?"
            Rewritten: "How many of the 8 Christopher Nolan movies are rated above 8?"

            Example 9 - Multiple References (with history):
            History: User asked about movies by rating, got 9+: 10 movies, 8-9: 25 movies, 7-8: 50 movies
            User: "What about the first two categories?"
            Rewritten: "What about movies rated 9+ (10 movies) and 8-9 (25 movies)?"

            Example 10 - Greeting/Non-Query:
            User: "Hello there!"
            Rewritten: "Hello there!"

            Example 10b - Greeting with Question:
            User: "Hi! Can you help me find movies?"
            Rewritten: "Hi! Can you help me find movies?"

            Example 11 - Complex Reference (with history):
            History: User asked about movies by decade, got 1990s: 100, 2000s: 145, 2010s: 200
            User: "Which decade has the highest count and what's the percentage?"
            Rewritten: "Which decade has the highest movie count among 1990s (100), 2000s (145), 2010s (200) and what's the percentage of the highest compared to total (445)?"

            Example 12 - Ambiguous Reference (no history):
            User: "Show me the best movies"
            Rewritten: "Show me movies with highest ratings (movies rated 8.5 and above)"

            OUTPUT FORMAT: Return only the rewritten query, nothing else
            """),
            MessagesPlaceholder("history"),
            ("user", "{current_question}")
        ])

    def __call__(self, state):
        history = state['messages'][:-1]
        current_question = state['messages'][-1].content

        formatted_history = "\n".join([
            f"{msg.type}: {msg.content}" for msg in history
        ])

        is_complex = self._is_complex_query(current_question, formatted_history)
        
        if is_complex:
            prompt_messages = self.complex_prompt.format_messages(
                history=[HumanMessage(content=formatted_history)],
                current_question=current_question
            )
        else:
            prompt_messages = self.simple_prompt.format_messages(
                history=[HumanMessage(content=formatted_history)],
                current_question=current_question
            )

        rewritten_query = self.llm.invoke(prompt_messages).content

        if self.verbose:
            print(f"Original Question: {current_question}")
            print(f"Query Complexity: {'Complex' if is_complex else 'Simple'}")
            print(f"Rewritten Query: {rewritten_query}")

        state['translated_query'] = rewritten_query

        return state

    def _is_complex_query(self, query, history):
        """Quick heuristic to determine if a query needs complex reasoning."""
        # Simple tests for complexity detection
        ambiguous_indicators = [
            'this', 'that', 'it', 'they', 'them', 'those', 'these',
            'first', 'last', 'next', 'previous', 'latest', 'earliest',
            'most', 'least', 'highest', 'lowest', 'best', 'worst',
            'recent', 'old', 'new', 'current', 'previous',
            'about', 'regarding', 'concerning'
        ]
        
        # Check for follow-up patterns
        follow_up_patterns = [
            'what about', 'how about', 'tell me more', 'and the',
            'what else', 'show me more', 'give me details', 'so', 'which',
            'among', 'amongst'
        ]
        
        # Check for simple greetings/pleasantries
        simple_greetings = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'thanks', 'thank you', 'bye', 'goodbye', 'see you',
            'nice to meet you', 'pleasure', 'have a good day'
        ]
        
        query_lower = query.lower()
        
        if any(greeting in query_lower for greeting in simple_greetings):
            return False
        
        has_ambiguous_indicators = any(indicator in query_lower for indicator in ambiguous_indicators)

        is_follow_up = any(pattern in query_lower for pattern in follow_up_patterns)
        
        has_history = len(history.strip()) > 0
        
        # Check query length (very short queries might be ambiguous)
        is_short = len(query.split()) <= 3
        
        complexity_score = sum([
            has_ambiguous_indicators,
            is_follow_up,
            has_history and is_short,
            has_history and has_ambiguous_indicators
        ])
        
        return complexity_score >= 2


class SupervisorNode:
    """Defines the graph node to route the user's query to the appropriate agent.
    """
    def __init__(self, supervisor_agent, verbose=False):
        self.supervisor_agent = supervisor_agent
        self.verbose = verbose

        return

    def __call__(self, state):
        command = self.supervisor_agent.run(state)
        if self.verbose:
            print(f'supervisor: command = "{command.goto}"')
        if command.goto != END:
            state['messages'].append(AIMessage(content=f"GOTO: {command.goto}"))

        return state


class SQLQueryNode:
    """Defines the graph node to execute SQL queries against the relational database using SQLAgent.
    """
    def __init__(self, sql_agent, verbose=False):
        self.sql_agent = sql_agent
        self.verbose = verbose

        return

    def __call__(self, state):
        query = state.get('translated_query', state['messages'][-1].content)
        print(f'sql_agent: query = "{query}"')
        message = self.sql_agent.run({'messages': [HumanMessage(content=query)]})['messages'][-1]
        state['messages'].append(message)
        if self.verbose:
            print(f'sql_agent: message = "{message.content}"')

        return state


class VectorEmbeddingsQueryNode:
    """Defines the graph node to perform similarity search using VectorAgent.
    """
    def __init__(self, vector_agent, verbose=False):
        self.vector_agent = vector_agent
        self.verbose = verbose

        return

    def __call__(self, state):
        query = state.get('translated_query', state['messages'][-1].content)
        message = self.vector_agent.run({'messages': [HumanMessage(content=query)]})['messages'][-1]
        state['messages'].append(message)
        if self.verbose:
            print(f'vector_agent: message = "{message.content}"')

        return state


class HybridQueryNode:
    """Defines the graph node to perform hybrid search (SQL prefilter + vector rerank)."""
    def __init__(self, hybrid_agent, verbose=False):
        self.hybrid_agent = hybrid_agent
        self.verbose = verbose

        return

    def __call__(self, state):
        query = state.get('translated_query', state['messages'][-1].content)
        message = self.hybrid_agent.run({'messages': [HumanMessage(content=query)]})['messages'][-1]
        state['messages'].append(message)
        if self.verbose:
            print(f'hybrid_agent: message = "{message.content}"')

        return state


class GraphQueryNode:
    """Defines the graph node to query a knowledge graph using GraphAgent.
    """
    def __init__(self, graph_agent, verbose=False):
        self.graph_agent = None
        self.verbose = verbose

        return

    def __call__(self, state):
        # Graph functionality removed
        message = AIMessage(content='Graph functionality is disabled in this demo.')
        state['messages'].append(message)

        return state


class FormatterNode:
    """Defines the graph node to format the response from the agents using FormatterAgent.
    """
    def __init__(self, formatter_agent, verbose=False):
        self.formatter_agent = formatter_agent
        self.verbose = verbose

    def __call__(self, state):
        result = self.formatter_agent.run(state)
        new_message = result['messages'][0]
        state['messages'].append(new_message)
        
        if self.verbose:
            print(f'formatter_agent: {new_message.content}')
            
        return state


class RefuserNode:
    """Politely refuses to answer the user's query using Refuser.
    """
    def __init__(self, llm, verbose=False):
        self.llm = llm
        self.verbose = verbose

        self.prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are a helpful assistant. Politely refuse to answer the user query.'),
            MessagesPlaceholder('messages'),
        ])

        return

    def __call__(self, state):
        message = AIMessage(content='Sorry, I know nothing about that.')
        state['messages'].append(message)
        if self.verbose:
            print(f'refuser: message = "{message.content}"')

        return state
