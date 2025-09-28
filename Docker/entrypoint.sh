#!/bin/bash
set -e

echo "🎬 Starting MovieRAG Setup..."

# Create chatbot.cfg with API key from environment variable
echo "📝 Creating configuration file..."
cat > movie_rag/chatbot.cfg << EOF
{
    "_comment": "MovieRAG Configuration - Auto-generated with API key",
    
    "host": "0.0.0.0",
    "port": "5005",
    
    "_comment_data_pipeline": "Data pipeline directories (relative to config file)",
    "landing_area": "${LANDING_AREA:-../data/01_landing}",
    "staging_area": "${STAGING_AREA:-../data/02_staging}", 
    "prepared_area": "${PREPARED_AREA:-../data/03_prepared}",
    
    "_comment_data_stores": "Database and vector store filenames",
    "sql_db_fname": "${SQL_DB_FNAME:-movies.db}",
    "embedding_store_fname": "${EMBEDDING_STORE_FNAME:-movie_embeddings.json}",
    
    "_comment_models": "LLM and embedding model configuration",
    "model_provider": "${MODEL_PROVIDER:-openai}",
    "model_name": "${MODEL_NAME:-gpt-4o-mini}",
    "embedding_model": "${EMBEDDING_MODEL:-text-embedding-3-small}",
    "openai_api_key": "${OPENAI_API_KEY}",
    
    "_comment_model_alternatives": "Alternative model configurations:",
    "_alt_ollama": "Set model_provider to 'ollama' and model_name to 'qwen2.5:7b' for local inference",
    "_alt_local_embeddings": "For local embeddings with Ollama, use 'nomic-embed-text'",
    "_alt_api_key": "Set your OpenAI API key above, or use environment variable OPENAI_API_KEY",
    
    "_comment_neo4j": "Neo4j graph database (optional - currently disabled)",
    "graph_url": "${GRAPH_URL:-}",
    "graph_username": "${GRAPH_USERNAME:-}",
    "graph_password": "${GRAPH_PASSWORD:-}",
    
    "_comment_behavior": "Application behavior settings",
    "with_history": ${WITH_HISTORY:-true},
    "verbose": ${VERBOSE:-true},
    "delete_table": ${DELETE_TABLE:-false},
    "delete_graph": ${DELETE_GRAPH:-false}
}
EOF

# Check if data preparation is needed
if [ ! -f "data/03_prepared/movies.db" ] || [ ! -f "data/03_prepared/movie_embeddings.json" ]; then
    echo "🔄 Data preparation needed. Running staging and prepping..."
    
    # Run staging
    echo "📊 Running data staging..."
    python -m movie_rag.stager -c movie_rag/chatbot.cfg
    
    # Run prepping
    echo "🧠 Running data prepping (creating embeddings and database)..."
    python -m movie_rag.prepper -c movie_rag/chatbot.cfg
    
    echo "✅ Data preparation completed!"
else
    echo "✅ Data already prepared, skipping staging and prepping."
fi

echo "🚀 Starting MovieRAG web server..."
python -m movie_rag.chat_server -c movie_rag/chatbot.cfg
