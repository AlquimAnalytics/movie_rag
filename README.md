# MovieRAG 🎬

A **Retrieval-Augmented Generation (RAG)** system for intelligent movie recommendations. MovieRAG combines structured SQL queries, vector similarity search, and hybrid retrieval methods to provide personalized movie suggestions based on mood, genre, actors, directors, or any descriptive query.

## 🚀 Features

- **Multi-Modal Retrieval**: Combines SQL queries, vector embeddings, and hybrid search
- **Intelligent Agent System**: Uses LangGraph for coordinated multi-agent workflows  
- **Web Interface**: Beautiful, modern chat interface with real-time RAG pipeline visualization
- **Flexible Data Pipeline**: Supports IMDB-style movie datasets with automated staging and preparation
- **LLM Support**: Compatible with Ollama (local) and OpenAI models
- **Memory Management**: Conversation history with forget functionality
- **Real-time Visualization**: Watch the RAG pipeline in action with animated progress indicators

## 🏗️ Architecture

MovieRAG uses a sophisticated multi-agent architecture built with LangGraph:

```
User Query → Query Translator → Supervisor Agent → [SQL Agent | Vector Agent | Hybrid Agent] → Formatter Agent → Response
```

### Agent Components

- **Query Translator**: Analyzes and reformulates user queries for optimal retrieval
- **Supervisor Agent**: Routes queries to appropriate retrieval agents based on query type
- **SQL Agent**: Executes structured queries against the movie database
- **Vector Agent**: Performs semantic similarity search using embeddings
- **Hybrid Agent**: Combines SQL and vector results for comprehensive retrieval
- **Formatter Agent**: Generates natural, conversational responses with movie recommendations

## 📁 Data Folder Structure

The project uses a three-tier data pipeline:

```
data/
├── 01_landing/          # Raw input data
│   ├── archive/         # Processed files moved here
│   └── movies_imdb.json # Raw IMDB-style movie data
├── 02_staging/          # Normalized, cleaned data
│   ├── archive/         # Staged files moved here
│   └── *.json          # Normalized movie records
└── 03_prepared/         # Production-ready data stores
    ├── movies.db        # SQLite database with normalized tables
    ├── movie_embeddings.json # Vector store with embeddings
    └── movies_demo.db   # Demo database (smaller dataset)
```

### Data Pipeline Flow

1. **Landing**: Raw IMDB data with original schema
2. **Staging**: Normalized to canonical schema with data cleaning
3. **Prepared**: Split into SQL database and vector embeddings for retrieval

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- OpenAI API key (required for default configuration)
- [Ollama](https://ollama.ai) (optional, for local LLM inference)

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/ranacoskun/movie_rag.git
cd movie_rag
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key (Required)

**Option A: Set in Configuration File (Recommended)**
```bash
# After copying the example config, edit it to add your API key
nano movie_rag/chatbot.cfg

# Add your API key to the "openai_api_key" field:
# "openai_api_key": "sk-your-api-key-here"
```

**Option B: Use Environment Variable**
```bash
# Set your OpenAI API key as an environment variable
export OPENAI_API_KEY="sk-your-api-key-here"

# Or add it to your shell profile for persistence
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 2b. Optional: Set Up Ollama for Local Inference

```bash
# Install Ollama (optional - for local models)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull qwen2.5:7b              # Main chat model
ollama pull nomic-embed-text        # For local embeddings
```

### 3. Configure Environment

Copy the example configuration and customize it:

```bash
# Copy example config
cp movie_rag/chatbot.cfg.example movie_rag/chatbot.cfg

# Edit configuration as needed
nano movie_rag/chatbot.cfg  # or use your preferred editor
```

Key configuration options:

```json
{
    "host": "0.0.0.0",                    // Server host
    "port": 5005,                         // Server port
    "model_provider": "openai",           // "openai" or "ollama"
    "model_name": "gpt-4o-mini",         // LLM model name
    "embedding_model": "text-embedding-3-small", // Embedding model
    "openai_api_key": "",                 // Your OpenAI API key (or use env var)
    "sql_db_fname": "movies.db",         // Database filename
    "embedding_store_fname": "movie_embeddings.json", // Vector store filename
    "with_history": true,                 // Enable conversation memory
    "verbose": true                       // Debug logging
}
```

## 🗄️ Creating Data Stores

### Step 1: Prepare Your Movie Data

Place your movie data in JSON format in `data/01_landing/`. The system supports:

**IMDB Format** (recommended):
```json
[
    {
        "Title": "The Shawshank Redemption",
        "Year": 1994,
        "Director": "Frank Darabont",
        "Actors": ["Tim Robbins", "Morgan Freeman"],
        "Genre": ["Drama"],
        "Description": "Two imprisoned men bond over a number of years...",
        "Rating": 9.3
    }
]
```

### Step 2: Stage the Data

Convert raw data to normalized format:

```bash
# From project root
python -m movie_rag.stager -c movie_rag/chatbot.cfg
```

This processes files from `01_landing/` → `02_staging/` with schema normalization.

### Step 3: Create Vector Store & SQL Database

Generate embeddings and populate database:

```bash
# Create both SQL database and vector embeddings
python -m movie_rag.prepper -c movie_rag/chatbot.cfg
```

This creates:
- **SQLite Database**: `data/03_prepared/movies.db` with normalized tables
- **Vector Store**: `data/03_prepared/movie_embeddings.json` with semantic embeddings

### Database Schema

The SQL database includes these tables:

```sql
-- Main movies table
CREATE TABLE movies (
    movie_id TEXT PRIMARY KEY,
    name TEXT,
    year INTEGER, 
    description TEXT,
    director TEXT,
    cast TEXT,
    genres TEXT,
    rating REAL
);

-- Normalized cast relationships
CREATE TABLE movie_cast (
    movie_id TEXT,
    actor TEXT,
    PRIMARY KEY (movie_id, actor)
);

-- Normalized genre relationships  
CREATE TABLE movie_genres (
    movie_id TEXT,
    genre TEXT,
    PRIMARY KEY (movie_id, genre)
);
```

## 🚀 Running MovieRAG

### Start the Web Server

```bash
# From project root
python -m movie_rag.chat_server -c movie_rag/chatbot.cfg
```

Then visit: `http://localhost:5005`

### Command Line Usage

```bash
# Interactive chat mode
python -m movie_rag.chatter -c movie_rag/chatbot.cfg

# Process new data
python -m movie_rag.stager -c movie_rag/chatbot.cfg
python -m movie_rag.prepper -c movie_rag/chatbot.cfg
```

### Configuration Options

Edit `movie_rag/chatbot.cfg`:

```json
{
    "host": "0.0.0.0",                    // Server host
    "port": 5005,                         // Server port
    "model_provider": "openai",           // "openai" or "ollama"
    "model_name": "gpt-4o-mini",         // LLM model name
    "embedding_model": "text-embedding-3-small", // Embedding model
    "openai_api_key": "",                 // Your OpenAI API key (or use env var)
    "with_history": true,                 // Enable conversation memory
    "verbose": true,                      // Debug logging
    "delete_table": false,                // Reset database on startup
    "landing_area": "../data/01_landing", // Raw data location
    "staging_area": "../data/02_staging", // Staged data location
    "prepared_area": "../data/03_prepared" // Production data location
}
```

## 💬 Usage Examples

### Web Interface Queries

- **Mood-based**: "Cozy fantasy movies, not bleak"
- **Actor-specific**: "Emma Stone movies with high ratings"
- **Director-focused**: "Best drama movies from Ridley Scott"
- **Analytical**: "How many movies were released in 2016?"
- **Similarity**: "Movies similar to Inception"
- **Structured**: "Show me Christopher Nolan movies rated above 8"

### Query Types & Agent Routing

The Supervisor Agent automatically routes queries to optimal retrieval agents:

- **SQL Agent**: Structured queries (ratings, years, counts, specific actors/directors)
- **Vector Agent**: Semantic queries (mood, themes, similarity, vibes)
- **Hybrid Agent**: Complex queries requiring both structured and semantic matching

## 🔧 Development

### Project Structure

```
movie_rag/
├── __init__.py
├── agents.py           # LangGraph agent implementations
├── chat_server.py      # Flask web server
├── chatter.py          # Core chat orchestration
├── nodes.py            # LangGraph node implementations  
├── prepper.py          # Data preparation pipeline
├── stager.py           # Data staging and normalization
├── tools.py            # Agent tools and utilities
├── utils.py            # Configuration and utilities
├── chatbot.cfg         # Configuration file
├── templates/
│   └── index.html      # Web interface
└── static/
    └── favicon.svg     # Web assets
```

### Adding New Data Sources

1. Place raw data in `data/01_landing/`
2. Run staging: `python -m movie_rag.stager -c movie_rag/chatbot.cfg`
3. Run preparation: `python -m movie_rag.prepper -c movie_rag/chatbot.cfg`

### Customizing Agents

Edit `movie_rag/agents.py` to modify agent behavior:

- **SupervisorAgent**: Query routing logic
- **SQLAgent**: Database query generation
- **VectorAgent**: Embedding search parameters
- **FormatterAgent**: Response generation and persona

## 🔍 Troubleshooting

### Common Issues

**1. OpenAI API Key Error**
```bash
# Check if API key is set in config file
grep "openai_api_key" movie_rag/chatbot.cfg

# Or check environment variable
echo $OPENAI_API_KEY

# If neither is set, add to config file or export as environment variable
export OPENAI_API_KEY="sk-your-api-key-here"
```

**2. Ollama Connection Error (if using local models)**
```bash
# Check if Ollama is running
ollama list
# Start Ollama service if needed
ollama serve
```

**3. Empty Database**
```bash
# Recreate database with fresh data
python -m movie_rag.prepper -c movie_rag/chatbot.cfg
```

**4. Missing Embeddings**
```bash
# Ensure OpenAI API key is set for text-embedding-3-small (default)
echo $OPENAI_API_KEY

# For local embeddings with Ollama
ollama pull nomic-embed-text
```

**5. Port Already in Use**
```bash
# Change port in chatbot.cfg or kill existing process
lsof -ti:5005 | xargs kill -9
```

### Debug Mode

Enable verbose logging in `chatbot.cfg`:
```json
{
    "verbose": true
}
```

## 📊 Performance & Scaling

### Recommended Hardware

- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores, SSD storage
- **For large datasets**: 32GB+ RAM, GPU for embedding generation

### Optimization Tips

1. **Choose appropriate OpenAI models**: Use `gpt-4o-mini` for cost efficiency or `gpt-4o` for best quality
2. **Use local embeddings** for faster inference and cost savings (Ollama + nomic-embed-text)
3. **Index your database** for large datasets
4. **Batch process** large data imports
5. **Cache embeddings** to avoid regeneration and API costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain & LangGraph**: Agent orchestration framework
- **Ollama**: Local LLM inference
- **Flask**: Web framework
- **scikit-learn**: Vector storage and similarity search
- **SQLite**: Embedded database engine

---

**MovieRAG** - Powered by Alquim Analytics © 2025
