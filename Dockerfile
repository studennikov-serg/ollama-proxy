# Dockerfile
FROM ollama/ollama:latest

# Set environment variables for the application
ENV RAG_DATA_DIR="/app/rag_data"
ENV PROXY_PORT="5000"
ENV OLLAMA_BASE_URL="http://127.0.0.1:11434"
# Ollama running inside the container
ENV CHUNK_SIZE="700"
ENV CHUNK_OVERLAP="70"
ENV TOP_K_RAG="4"
# Default model to pull if needed
ENV OLLAMA_MODEL_NAME="llama3"
# High quality embedding model
ENV EMBEDDING_MODEL_NAME="BAAI/bge-large-en-v1.5"

# Create application directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install faiss-cpu prerequisites
# These are often linked to BLAS/LAPACK implementations. OpenBLAS is a common choice.
RUN apt-get update && apt-get install -y libopenblas-dev liblapack-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY entrypoint.sh .

# Ensure entrypoint.sh is executable
RUN chmod +x /app/entrypoint.sh

# Create data directories and set permissions
RUN mkdir -p /app/rag_data
RUN mkdir -p /root/.ollama/models # Ollama stores models here

# Expose ports
EXPOSE 5000
EXPOSE 11434

# Set entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
