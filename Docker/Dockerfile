# MovieRAG Dockerfile for Flask Web Application
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install jq for JSON processing (required by MovieRAG)
RUN pip install jq

# Copy MovieRAG application code
COPY movie_rag/ ./movie_rag/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p data/01_landing/archive
RUN mkdir -p data/02_staging/archive  
RUN mkdir -p data/03_prepared

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=movie_rag.chat_server
ENV FLASK_ENV=production
ENV PORT=5005

# Expose port (MovieRAG runs on 5005 by default)
EXPOSE 5005

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5005/ || exit 1

# Copy entrypoint script
COPY Docker/entrypoint.sh /app/entrypoint.sh

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
