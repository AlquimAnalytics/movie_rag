"""agents
"""

import os, json, sqlite3
from typing import Literal
from pydantic import BaseModel
from typing import Optional, List

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import Command
from langgraph.graph import END
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.utilities import SQLDatabase

from .tools import SQLTool, SKLearnVectorStoreTool, SupervisorTool


class MovieFilter(BaseModel):
    """Pydantic model for structured movie filter extraction."""
    title: Optional[str] = None
    director: Optional[str] = None
    actor: Optional[str] = None
    genre: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    rating_min: Optional[float] = None
    rating_max: Optional[float] = None
    limit: Optional[int] = None
    description_keywords: Optional[List[str]] = None

# Original source of this prompt template: langchain.hub.pull('langchain-ai/sql-agent-system-prompt') 
# Modified to include reasoning and CoT
SQL_SYSTEM_MESSAGE_TEMPLATE = """
Given an input query, create a syntactically correct {dialect} query to run to help find the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.

Whenever possible, refrain from querying all the columns from a specific table. Ask only for the relevant columns given the question.

Think step by step:
1. Analyze the user query and identify the required information
2. Examine the table schema to understand available data
3. Create a syntactically correct {dialect} query that retrieves the needed information
4. Ensure the query is efficient and returns relevant results

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

Only use the following tables and schema:
{table_info}
"""

SQL_HUMAN_MESSAGE_TEMPLATE = """
Question: {input_question}
"""


class SupervisorAgent:
    """Analyzes user input and determines the appropriate agent to handle the query.
    """
    def __init__(self, llm, verbose=False):
        self.supervisor_tool = SupervisorTool(llm, verbose)
        self.verbose = verbose

    def run(self, state):
        result = self.supervisor_tool(state)

        return Command(goto=result['messages'][-1].content.replace('GOTO:', '').strip())


class SQLAgent:
    """Queries a relational database.
    """
    def __init__(self, sql_db_uri, llm):
        self.sql_db_uri = sql_db_uri
        self.llm = llm

        # Load the SQL database.
        self.sql_db = SQLDatabase.from_uri(f'sqlite:///{self.sql_db_uri}')

        query_prompt_template = ChatPromptTemplate([
            ('system', SQL_SYSTEM_MESSAGE_TEMPLATE),
            ('human', SQL_HUMAN_MESSAGE_TEMPLATE),
        ])
        self.sql_tool = SQLTool(sql_db=self.sql_db, query_prompt_template=query_prompt_template, llm=self.llm)

        return

    def run(self, state):
        question = state['messages'][-1].content
        result = self.sql_tool(question)
        reply_message = f'{result}'
        reply = {'messages': state['messages'] + [AIMessage(content=reply_message)]}

        return reply


class VectorAgent:
    """Performs similarity search in a vector store.
    """
    def __init__(self, vs_store_path, embeddings):
        self.vs_store_path = vs_store_path
        self.embeddings = embeddings

        # Load the embeddings from the vector store.
        if os.path.exists(self.vs_store_path):
            try:
                self.vs = SKLearnVectorStore(
                    embedding=self.embeddings,
                    persist_path=self.vs_store_path,
                    serializer='json'
                )
            except json.JSONDecodeError as e:
                raise RuntimeError(f'Error parsing JSON data from vector store: {e}')

        self.vector_tool = SKLearnVectorStoreTool(self.vs)

        return

    def run(self, state):
        query = state['messages'][-1].content
        result = self.vector_tool(query)
        reply = {'messages': state['messages'] + [AIMessage(content=f'Vector Results:\n{result}')]} 

        return reply


class HybridAgent:
    """Combines SQL filtering with vector similarity re-ranking.
    """
    def __init__(self, sql_db_uri, vs_store_path, llm, embeddings):
        self.sql_db_uri = sql_db_uri
        self.vs_store_path = vs_store_path
        self.llm = llm
        self.embeddings = embeddings

        # SQL database for pre-filtering (use sqlite3 for parameterized queries)
        self.sqlite_path = self.sql_db_uri

        # Enhanced prompt to extract structured filters with better genre understanding
        self.filter_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """Extract structured movie filters from the user's request. Use null for missing values.
             Return a JSON object with keys: title, director, actor, genre, year_min, year_max, rating_min, rating_max, limit, description_keywords.
             
             For genres, be specific and use common movie genre terms:
             - Use "sci-fi" or "science fiction" for science fiction
             - Use "action", "comedy", "drama", "thriller", "horror", "romance", "adventure", "animation", etc.
             - For genre combinations like "cozy sci-fi", extract the main genre ("sci-fi") and put descriptors in description_keywords
             
             For description_keywords, extract mood/style descriptors that aren't genres:
             - "cozy", "dark", "bleak", "uplifting", "intense", "light-hearted", "thoughtful", etc.
             
             Examples:
             - 'cozy sci-fi movies, not bleak' -> {"genre":"sci-fi", "description_keywords":["cozy", "not bleak", "uplifting"]}
             - 'Christopher Nolan films rated over 8' -> {"director":"Christopher Nolan","rating_min":8}
             - 'dark thriller movies from the 90s' -> {"genre":"thriller", "year_min":1990, "year_max":1999, "description_keywords":["dark"]}
             - 'movies similar to Inception' -> {"title":"Inception", "description_keywords":["similar", "mind-bending"]}
             """),
            ("user", "{query}")
        ])

        self.llm_structured = self.llm.with_structured_output(MovieFilter)

        # Vector store for re-ranking
        if os.path.exists(self.vs_store_path):
            self.vs = SKLearnVectorStore(
                embedding=self.embeddings,
                persist_path=self.vs_store_path,
                serializer='json'
            )
        else:
            self.vs = None

    def _build_sql_where(self, filt: MovieFilter) -> str:
        clauses = []
        params = []
        
        if filt.title:
            clauses.append("\"name\" LIKE ?")
            params.append(f"%{filt.title}%")
        if filt.director:
            clauses.append("\"director\" LIKE ?")
            params.append(f"%{filt.director}%")
        if filt.actor:
            clauses.append("\"cast\" LIKE ?")
            params.append(f"%{filt.actor}%")
        
        # Enhanced genre filtering with case-insensitive matching
        if filt.genre:
            genre = filt.genre.lower()
            # Handle common genre variations
            genre_variations = {
                'sci-fi': ['sci-fi', 'science fiction', 'science-fiction'],
                'science fiction': ['sci-fi', 'science fiction', 'science-fiction'],
                'action': ['action'],
                'comedy': ['comedy'],
                'drama': ['drama'],
                'thriller': ['thriller'],
                'horror': ['horror'],
                'romance': ['romance', 'romantic'],
                'adventure': ['adventure'],
                'animation': ['animation', 'animated']
            }
            
            search_terms = genre_variations.get(genre, [genre])
            genre_clauses = []
            for term in search_terms:
                genre_clauses.append("LOWER(\"genres\") LIKE ?")
                params.append(f"%{term}%")
            
            if genre_clauses:
                clauses.append("(" + " OR ".join(genre_clauses) + ")")
        
        # Enhanced description filtering for mood/style keywords
        if filt.description_keywords:
            keywords = filt.description_keywords
            if isinstance(keywords, list):
                for keyword in keywords:
                    keyword = keyword.lower().strip()
                    if keyword and not keyword.startswith('not '):
                        clauses.append("(LOWER(\"description\") LIKE ? OR LOWER(\"genres\") LIKE ?)")
                        params.extend([f"%{keyword}%", f"%{keyword}%"])
        
        if filt.year_min is not None:
            clauses.append("\"year\" >= ?")
            params.append(int(filt.year_min))
        if filt.year_max is not None:
            clauses.append("\"year\" <= ?")
            params.append(int(filt.year_max))
        if filt.rating_min is not None:
            clauses.append("\"rating\" >= ?")
            params.append(float(filt.rating_min))
        if filt.rating_max is not None:
            clauses.append("\"rating\" <= ?")
            params.append(float(filt.rating_max))
        
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return where, params

    def _sql_prefilter(self, filt: MovieFilter, limit_default: int = 100):
        where, params = self._build_sql_where(filt)
        limit = int(filt.limit or limit_default)
        query = f"SELECT \"movie_id\", \"name\", \"year\", \"description\", \"director\", \"cast\", \"genres\", \"rating\" FROM movies{where} LIMIT {limit};"
        rows: list[dict] = []
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            fetched = cur.fetchall()
            for row in fetched:
                rows.append({k: row[k] for k in row.keys()})
        finally:
            conn.close()
        return rows

    def _rerank_with_vector(self, query_text: str, rows, filt: MovieFilter):
        if not self.vs or not rows:
            return rows
        
        try:
            # Get more candidates from vector store for better reranking
            k_candidates = min(len(rows) * 3, 100)  # Get 3x more for better filtering
            hits_with_scores = self.vs.similarity_search_with_score(query_text, k=k_candidates)
            
            # Create a mapping from movie identifiers to vector similarity scores
            vector_scores = {}
            for doc, score in hits_with_scores:
                # Try to match by movie name and year from metadata
                movie_name = doc.metadata.get('name', '')
                movie_year = doc.metadata.get('year', '')
                
                # Also try to extract from page_content if metadata is missing
                if not movie_name and doc.page_content:
                    try:
                        # Try to parse movie name from JSON content
                        import json
                        content_data = json.loads(doc.page_content)
                        movie_name = content_data.get('name', '')
                        movie_year = content_data.get('year', '')
                    except:
                        # Fallback to using first part of content
                        movie_name = doc.page_content[:50]
                
                if movie_name:
                    key = f"{movie_name}_{movie_year}".lower()
                    vector_scores[key] = 1.0 - score  # Convert distance to similarity
            
            # Apply negative filtering for "not" keywords
            negative_keywords = []
            if filt.description_keywords:
                for kw in filt.description_keywords:
                    if isinstance(kw, str) and kw.lower().startswith('not '):
                        negative_keywords.append(kw.lower().replace('not ', '').strip())
            
            # Filter out movies that match negative keywords
            if negative_keywords:
                filtered_rows = []
                for row in rows:
                    description = (row.get('description') or '').lower()
                    genres = (row.get('genres') or '').lower()
                    
                    # Check if any negative keyword appears in description or genres
                    has_negative = any(neg_kw in description or neg_kw in genres for neg_kw in negative_keywords)
                    if not has_negative:
                        filtered_rows.append(row)
                rows = filtered_rows
            
            # Sort by vector similarity score
            def get_similarity_score(row):
                movie_name = row.get('name', '')
                movie_year = row.get('year', '')
                key = f"{movie_name}_{movie_year}".lower()
                return vector_scores.get(key, 0.0)
            
            rows.sort(key=get_similarity_score, reverse=True)
            
        except Exception as e:
            # Fallback: basic keyword filtering for negative terms
            if filt.description_keywords:
                negative_keywords = [kw.lower().replace('not ', '').strip() 
                                   for kw in filt.description_keywords 
                                   if isinstance(kw, str) and kw.lower().startswith('not ')]
                if negative_keywords:
                    rows = [row for row in rows 
                           if not any(neg_kw in (row.get('description') or '').lower() or 
                                     neg_kw in (row.get('genres') or '').lower() 
                                     for neg_kw in negative_keywords)]
        
        return rows

    def run(self, state):
        query_text = state['messages'][-1].content
        try:
            filt = self.llm_structured.invoke(self.filter_prompt.format_messages(query=query_text))
            if not isinstance(filt, MovieFilter):
                filt = MovieFilter()
        except Exception:
            filt = MovieFilter()

        rows = self._sql_prefilter(filt)
        rows = self._rerank_with_vector(query_text, rows, filt)

        # Format concise list output
        formatted = []
        for r in rows[:10]:
            name = r.get('name') if isinstance(r, dict) else str(r)
            year = r.get('year') if isinstance(r, dict) else ''
            director = r.get('director') if isinstance(r, dict) else ''
            rating = r.get('rating') if isinstance(r, dict) else ''
            formatted.append(f"{name} ({year}) — Director: {director} — Rating: {rating}")
        message = AIMessage(content="\n".join(formatted) if formatted else "No results found")
        reply = {'messages': state['messages'] + [message]}
        return reply
class FormatterAgent:
    """Formats the response from the agents into a natural language response, and handles general conversation.
    """
    def __init__(self, llm, verbose=False, persona_directives=None):
        self.llm = llm
        self.verbose = verbose
        self.persona_directives = persona_directives or {}

        # Direct response prompt for greetings and general conversation
        self.direct_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             """You're a fun movie matchmaker friend who knows cinema inside and out! 
             Persona: {persona_instructions}
             
             Your vibe: Warm, witty, and genuinely excited about movies. Think "friend who always has the perfect rec."
             - Greet people like you're genuinely happy to help them find their next obsession
             - Use casual, conversational language ("What's your mood?" not "How may I assist?")
             - Drop in light movie references or playful observations when it feels natural
             - 1-2 emoji max, and only if they add genuine warmth
             - Always spoiler-free unless explicitly asked otherwise
             """),
            MessagesPlaceholder('messages'),
        ])

        # Comprehensive formatting prompt with full CoT reasoning
        self.formatting_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You're a fun movie matchmaker friend who takes raw movie data and turns it into perfect recommendations! "
             "Persona: {persona_instructions}\n\n"
             
             "Your job: Transform boring agent outputs into genuinely helpful, conversational responses that feel like talking to a movie-obsessed friend.\n\n"
             
             "Your style:\n"
             "- Casual, warm tone (\"Here's what I found\" not \"The results indicate\")\n"
             "- For movie recs: 'Title (Year) — Vibe description — Why it's perfect for them'\n"
             "- For counts/stats: Give the number with a fun observation (\"2016 was BUSY — 297 movies!\")\n"
             "- For no results: Stay encouraging, suggest tweaks (\"Hmm, no luck there. Want to try [similar thing]?\")\n"
             "- Use light humor when it feels natural, but never forced\n"
             "- 1-2 emoji max, only when they genuinely add warmth\n"
             "- Always spoiler-free unless they specifically ask for spoilers\n\n"
             
             "Examples of your voice:\n"
             "- Arrival (2016) — Thoughtful sci-fi that won't stress you out — Perfect for a cozy night when you want something smart but soothing.\n"
             "- Her (2013) — Gentle future romance — Soft colors, softer feelings, and Joaquin Phoenix being vulnerable.\n"
             "- Found 297 movies from 2016! That year was absolutely stacked. 🎬\n"
             ),
            ("user", 
             "Original question: {user_question}\n"
             "Translated query: {translated_query}\n"
             "Agent output: {agent_output}\n"
             "Conversation history: {conversation_history}\n\n"
             "Turn this into a warm, helpful response that sounds like you're genuinely excited to help them find their next favorite movie.")
        ])
        
    def run(self, state):
        last_routing_message = None
        for msg in reversed(state['messages']):
            if isinstance(msg, AIMessage) and msg.content.startswith("GOTO:"):
                last_routing_message = msg
                break
        
        # Check if this is a direct response (no previous agent output to format)
        # Look for recent agent outputs (non-routing messages from other agents)
        has_recent_agent_output = False
        for msg in reversed(state['messages']):
            if isinstance(msg, AIMessage) and not msg.content.startswith("GOTO:"):
                # Found a recent agent output that needs formatting
                has_recent_agent_output = True
                break
            elif isinstance(msg, HumanMessage):
                # Reached the user's message without finding agent output
                break
        
        is_direct_response = not has_recent_agent_output
        
        if is_direct_response:
            return self._handle_direct_response(state)
        else:
            return self._handle_agent_output(state)
    
    def _handle_direct_response(self, state):
        """Handle greetings and general conversation
        """
        user_message = None
        for msg in reversed(state['messages']):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            user_message = "Hello"
        
        persona_instructions = state.get('persona_instructions', '')
        
        # Include conversation history for better context
        conversation_messages = []
        for msg in state['messages'][-5:]:  # Last 5 messages for context
            if isinstance(msg, (HumanMessage, AIMessage)) and not msg.content.startswith("GOTO:"):
                conversation_messages.append(msg)
        
        prompt_messages = self.direct_prompt.format_messages(
            persona_instructions=persona_instructions,
            messages=conversation_messages if conversation_messages else [HumanMessage(content=user_message)]
        )
        
        response = self.llm.invoke(prompt_messages)
        
        if self.verbose:
            print(f"Direct Response - User: {user_message}")
            print(f"Response: {response.content}")
        
        return {'messages': [AIMessage(content=response.content)]}
    
    def _handle_agent_output(self, state):
        """Format output from agents with comprehensive formatting
        """
        agent_output = None
        for msg in reversed(state['messages']):
            if isinstance(msg, AIMessage) and not msg.content.startswith("GOTO:"):
                agent_output = msg.content
                break
        
        if not agent_output:
            agent_output = "No agent output found"
        
        user_question = state.get('user_question', '')
        translated_query = state.get('translated_query', '')
        
        # Get conversation history for context, this time the prompts are being formatted
        conversation_history = self._extract_conversation_context(state)
        
        persona_instructions = state.get('persona_instructions', '')
        prompt_messages = self.formatting_prompt.format_messages(
            persona_instructions=persona_instructions,
            user_question=user_question,
            translated_query=translated_query,
            agent_output=agent_output,
            conversation_history=conversation_history
        )
        
        response = self.llm.invoke(prompt_messages)
        
        if self.verbose:
            print(f"Formatting - User Question: {user_question}")
            print(f"Agent Output Length: {len(agent_output)} chars")
            print(f"Formatted Response: {response.content}")
        
        # Deterministic fallback if the model doesn't follow persona style
        text = response.content or ""
        if not self._looks_like_persona(text):
            text = self._fallback_format(user_question, agent_output)
        return {'messages': [AIMessage(content=text)]}
    
    def _extract_conversation_context(self, state):
        """Extract relevant conversation context for formatting decisions
        """
        context_parts = []
        
        recent_messages = state['messages'][-5:]  # Last 5 messages
        
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                context_parts.append(f"User: {content}")
            elif isinstance(msg, AIMessage) and not msg.content.startswith("GOTO:"):
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                context_parts.append(f"Assistant: {content}")
        
        return "\n".join(context_parts) if context_parts else "No recent context"

    def _looks_like_persona(self, text: str) -> bool:
        if not text:
            return False
        # Heuristic: contains at least one line with ' — ' (emdash spaced) and a vibe phrase
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        has_bullets = any(' — ' in l and '(' in l and ')' in l for l in lines)
        return has_bullets

    def _fallback_format(self, user_question: str, agent_output: str) -> str:
        # Parse simple lines like: Title (Year) — Director: X — Rating: Y
        recs = []
        for line in (agent_output or '').split('\n'):
            line = line.strip('- ').strip()
            if not line:
                continue
            # Try to extract Title (Year)
            title = None
            year = None
            try:
                if ')' in line and '(' in line and line.index('(') < line.index(')'):
                    title = line.split('(')[0].strip()
                    year = line.split('(')[1].split(')')[0].strip()
                else:
                    title = line
            except Exception:
                title = line
            if title:
                recs.append((title, year))
        
        # Derive a fun vibe from the user's phrasing
        if "cozy" in user_question.lower():
            vibe = "Cozy vibes guaranteed"
        elif "action" in user_question.lower():
            vibe = "Pure adrenaline fuel"
        elif "sci-fi" in user_question.lower() or "sci fi" in user_question.lower():
            vibe = "Mind-bending goodness"
        elif "comedy" in user_question.lower():
            vibe = "Laugh-out-loud material"
        else:
            vibe = "Right up your alley"
            
        bullets = []
        for (t, y) in recs[:3]:
            if y:
                bullets.append(f"{t} ({y}) — {vibe} — Trust me on this one!")
            else:
                bullets.append(f"{t} — {vibe} — You're gonna love it!")
        
        if bullets:
            return "\n".join(bullets)
        else:
            return agent_output or "Hmm, no matches there! Want to try something slightly different? 🎬"
    