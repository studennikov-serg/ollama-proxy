#!/bin/bash
set -e

echo "--- Entrypoint Script Started (PRODUCTION MODE - Ollama RAG) ---"

# Configuration variables (passed as environment variables from Dockerfile now)
# OLLAMA_HOST="0.0.0.0:11434" # This is for Ollama's internal binding, not needed as an env var for app
# OLLAMA_MODEL_NAME is already an ENV var
# RAG_DATA_DIR is already an ENV var
# EMBEDDING_MODEL_NAME is already an ENV var
# PROXY_PORT is already an ENV var
# OLLAMA_BASE_URL is already an ENV var
# CHUNK_SIZE is already an ENV var
# CHUNK_OVERLAP is already an ENV var
# TOP_K_RAG is already an ENV var


echo "Ollama Host: 0.0.0.0:11434" # Hardcoded for clarity based on Ollama's default
echo "Ollama Model Name: $OLLAMA_MODEL_NAME"
echo "RAG Data Directory: $RAG_DATA_DIR"
echo "Embedding Model: $EMBEDDING_MODEL_NAME"
echo "Proxy Port: $PROXY_PORT"
echo "OLLAMA_BASE_URL: $OLLAMA_BASE_URL"
echo "CHUNK_SIZE: $CHUNK_SIZE"
echo "CHUNK_OVERLAP: $CHUNK_OVERLAP"
echo "TOP_K_RAG: $TOP_K_RAG"

# Ensure RAG data and Ollama models directories exist
mkdir -p "$RAG_DATA_DIR"
echo "Ensured RAG data directory exists: $RAG_DATA_DIR"
mkdir -p "/root/.ollama/models" # Ollama stores models here
echo "Ensured Ollama models directory exists: /root/.ollama/models"

# Start Ollama server in the background
echo "Starting Ollama server on 0.0.0.0:11434..."
ollama serve &

OLLAMA_PID=$!
echo "Ollama server started with PID: $OLLAMA_PID"

# Wait for Ollama server to become available
echo "Waiting for Ollama server to become available at $OLLAMA_BASE_URL/api/tags..."
for i in $(seq 1 60); do # Wait up to 5 minutes (60 * 5 seconds)
    if curl --silent --fail "$OLLAMA_BASE_URL/api/tags"; then
        echo "Ollama server is up!"
        # Pull the default model if it's not already downloaded
        echo "Attempting to pull default model: $OLLAMA_MODEL_NAME..."
        ollama pull "$OLLAMA_MODEL_NAME" || { echo "Failed to pull model $OLLAMA_MODEL_NAME. Ensure network access or pre-download." && exit 1; }
        echo "Model $OLLAMA_MODEL_NAME is ready."
        break
    else
        echo "Waiting for Ollama... ($((i*5))/300 seconds)"
        sleep 5
    fi
    if [ "$i" -eq 60 ]; then
        echo "Error: Ollama server did not become available within 5 minutes."
        exit 1 # Exit if Ollama doesn't start
    fi
done

# Start Flask proxy application with Gunicorn
echo "Starting Flask proxy application with Gunicorn..."
# Gunicorn binds to 0.0.0.0 to be accessible from outside the container
# Use 4 workers for better concurrency (adjust based on CPU cores)
exec gunicorn -w 4 -b 0.0.0.0:"$PROXY_PORT" app:app
