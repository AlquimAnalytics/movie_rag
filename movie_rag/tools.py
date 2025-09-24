"""tools.py"""

from typing import Annotated, List, TypedDict, Mapping, Literal
from pydantic import BaseModel, Field

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

class SQLTool:
    class SQLState(TypedDict):
        sql_phase: Annotated[dict, "Subgraph state storage"] # Namespaced state
        question: str
        query: str
        result: str
        answer: str

    class QueryOutput(TypedDict):
        query: Annotated[str, ..., "Syntactically valid SQL query."]

    def __init__(self, sql_db, query_prompt_template, llm, verbose=True):
        self.db = sql_db
        self.query_prompt_template = query_prompt_template
        self.verbose = verbose

        self.llm = llm
        self.llm_structured = llm.with_structured_output(self.QueryOutput)
        self.query_executor = QuerySQLDatabaseTool(db=self.db)

        # Build the StateGraph for SQL queries.
        graph_builder = StateGraph(self.SQLState)
        graph_builder.add_node('compose_query', self.compose_query)
        graph_builder.add_node('execute_query', self.execute_query)
        graph_builder.add_node('generate_answer', self.generate_answer)
        graph_builder.set_entry_point('compose_query')
        graph_builder.add_edge('compose_query', 'execute_query')
        graph_builder.add_edge('execute_query', 'generate_answer')
        self.graph = graph_builder.compile()

        return

    def compose_query(self, state: SQLState):
        prompt = self.query_prompt_template.invoke({
            'dialect': self.db.dialect,
            'top_k': 10,
            'table_info': self.db.get_table_info(),
            'input_question': state['question'],
        })
        result = self.llm_structured.invoke(prompt)
        if isinstance(result, Mapping):
            query = result.get('query', 'Unable to generate query.')
        else:
            query = 'Unable to generate query.'
        reply = {'sql_phase': {'query': query}}

        return reply

    def execute_query(self, state: SQLState):
        result = self.query_executor.invoke(state['sql_phase']['query'])
        reply = {'sql_phase': {'result': result}}

        return reply

    def generate_answer(self, state: SQLState):
        try:
            question = state.get('question', 'No question provided')
            sql_phase = state.get('sql_phase', {})
            query = sql_phase.get('query', 'No query generated')
            result = sql_phase.get('result', 'No results found')

            prompt = f"""
                Given the following user query and SQL result, provide an answer to the user query. Do not reinterpret the query, just answer it.
                Query: {question}
                SQL Result: {result}
            """
            response = self.llm.invoke(prompt)
            answer = {'sql_phase': {'answer': response.content}}
        except Exception as e:
            answer = {'sql_phase': {'answer': f'Error generating answer: {str(e)}'}}

        return answer

    def __call__(self, question: str):
        final_state = None
        for step in self.graph.stream(
            {'question': question},
            stream_mode='updates',
            output_keys=['sql_phase']  # Explicit output tracking
        ):
            if self.verbose:
                if 'sql_phase' in step:
                    print(f'{step["sql_phase"]}')
                else:
                    print(f'Error in sql_agent: "{step}".')
            final_state = step

        answer = (final_state.get('sql_phase', {})
                        .get('answer', 'Unable to generate answer.'))
        return answer


class SKLearnVectorStoreTool:
    """Perform similarity search on movie data using a vector store.
    """
    def __init__(self, vs):
        self.vs = vs

    def __call__(self, query_text):
        try:
            result = self.vs.similarity_search(query_text)
            reply = result
        except Exception as e:
            reply = str(e)

        return reply


class SupervisorTool:
    """Routes user queries to the appropriate agent based on the query type.
    """
    class Router(BaseModel):
        """Always use this tool to route the user's query to the appropriate agent.
        """
        next_agent: Literal['sql_agent', 'vector_agent', 'hybrid_agent', 'graph_agent', 'formatter_agent', 'refuser', 'FINISH']

    def __init__(self, llm, verbose=True):
        self.llm = llm
        self.verbose = verbose
        self.llm_structured = llm.with_structured_output(self.Router)

        self.prompt = ChatPromptTemplate.from_messages([
            ('system',
            """
            You are an assistant that answers user queries about movies, actors, directors, and genres.
            Answer ONLY questions related to film data.
            Route the user's query to the appropriate agent:
            1. sql_agent: for structured data, metrics, filters, or comparisons.
            2. vector_agent: for unstructured, semantic similarity questions.
            3. hybrid_agent: when the query combines filters (year/genre/director/rating/actor) with semantic similarity.
            3. formatter_agent: for greetings/pleasantries.
            4. refuser: when the question is outside of scope.

            Examples:
            - "How many movies were released in 1999?" -> sql_agent
            - "Find movies similar to The Matrix" -> vector_agent
            - "Sci-fi movies after 2005 like Interstellar" -> hybrid_agent

            Output: Return only the next_agent attribute as one of <"sql_agent"|"vector_agent"|"hybrid_agent"|"formatter_agent"|"refuser"|"FINISH">.
            """),
            MessagesPlaceholder(variable_name='messages'),
        ])

    def __call__(self, state):
        last_message_content = state['messages'][-1].content.lower()

        prompt_value = self.prompt.invoke({'messages': state['messages']})
        if self.verbose:
            for message in prompt_value.messages:
                if isinstance(message, SystemMessage):
                    pass
                elif isinstance(message, AIMessage):
                    print(f'AI message: {message.content}')
                elif isinstance(message, HumanMessage):
                    print(f'Human message: {message.content}')
                elif isinstance(message, BaseMessage):
                    print(f'Unknown message: {message.content}')
                else:
                    print(f'Unknown message: {message}')
        try:
            decision = self.llm_structured.invoke(prompt_value)
        except KeyError as e:
            if e.args[0] in ['sql_agent', 'vector_agent', 'hybrid_agent', 'formatter_agent', 'refuser']:
                decision = self.Router(next_agent=e.args[0])
            else:
                state['messages'].append(AIMessage(content='Encountered an Ollama bug! Cannot retrieve any information from "{e.args[0]}".'))
                state['messages'].pop(0)
                decision = self.Router(next_agent='FINISH')
        if decision:
            next_agent = decision.next_agent
            if next_agent == 'FINISH':
                next_agent = END
        else:
            state['messages'].append(AIMessage(content='Sorry, I am unable to decide how to answer your query.'))
            next_agent = END
            if self.verbose:
                print('No decision made by the LLM.')

        if self.verbose:
            print(f'Next agent: {next_agent}')

        return {'messages': state['messages'] + [AIMessage(content=f"GOTO: {next_agent}")]}
