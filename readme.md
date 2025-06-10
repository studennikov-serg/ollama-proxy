# Ollama RAG Proxy System

This project provides an easy and transparent way to integrate a Retrieval-Augmented Generation (RAG) system with Ollama's Large Language Models (LLMs). It leverages a Flask-based proxy server to handle RAG functionalities (document ingestion, embedding, retrieval) and then forwards augmented queries to an Ollama LLM, returning OpenAI-compatible chat completions.

The entire setup is containerized using Docker, ensuring portability and ease of deployment. Ollama runs directly within the same Docker container, simplifying local development and production environments.

## Features

* **Integrated Ollama LLM:** Ollama server runs directly within the Docker container.
* **RAG Capabilities:** Incorporates document embedding (using `sentence-transformers` and `FAISS`) to provide contextual information to the LLM.
* **OpenAI API Compatibility (Partial):** The chat completion endpoint (`/v1/chat/completions`) is designed to mimic the OpenAI Chat Completions API for easier integration with existing tools.
* **Easy Document Ingestion:** Simple endpoint for uploading text files to build the RAG knowledge base.
* **Containerized Deployment:** Uses Docker to encapsulate all dependencies, making setup straightforward.

## Getting Started

### Prerequisites

* [Docker](https://docs.docker.com/get-docker/) installed and running.
* `curl` (for testing API endpoints from your terminal).
* `jq` (optional, for pretty-printing JSON responses in the terminal).

### Project Structure

```
.
├── Dockerfile                  # Defines the Docker image
├── app.py                      # Flask application with RAG and Ollama proxy logic
├── entrypoint.sh               # Script to set up environment and start services in the container
├── requirements.txt            # Python dependencies
└── run_production.sh           # Script to build, run, and test the Docker container
```

### Setup and Running

1.  **Clone the repository** (or ensure you have all the project files in a directory).

2.  **Make `run_production.sh` executable** (if it's not already):
    ```bash
    chmod +x run_production.sh
    ```

3.  **Run the production script:** This script will build the Docker image, start the container, and perform initial health checks and API tests.
    ```bash
    ./run_production.sh
    ```
    The script will log its output to `debug_run.log` in your current directory. It takes some time on the first run as Ollama pulls the default model (`llama3`) and the embedding model is downloaded.

    You should see `--- FINAL STATUS SUMMARY ---` indicating `SUCCESS` for all tests if everything is working correctly.

### Main Commands

* **Build & Run (and Test):**
    ```bash
    ./run_production.sh
    ```
    This script handles stopping/removing old containers/images, building the new image (with `--no-cache` for a fresh build), running the container, and executing initial tests.

* **Stop the running container:**
    ```bash
    docker stop ollama_rag_proxy_container
    ```

* **Remove the container:**
    ```bash
    docker rm ollama_rag_proxy_container
    ```

* **Remove the Docker image (to force a full rebuild next time):**
    ```bash
    docker rmi ollama_rag_proxy_image
    ```

* **View live container logs:**
    ```bash
    docker logs -f ollama_rag_proxy_container
    ```

* **Copy application logs from container to host:**
    ```bash
    docker cp ollama_rag_proxy_container:/app/rag_data/app.log ./app_container.log
    ```
    The `app.log` file contains detailed application-level logs, including Python tracebacks for errors.

### API Endpoints and `curl` Examples

The Flask application exposes endpoints on port `5000` of your host machine (mapped from the container).

#### 1. Health Check

Checks the status of the Flask application and the RAG index.

```bash
curl [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health) | jq .
```

Expected successful response:
```json
{
  "faiss_index_total_vectors": 0,
  "status": "healthy",
  "total_documents": 0
}
```
(The `total_vectors` and `total_documents` will be `0` initially, increasing after documents are uploaded).

#### 2. Upload Text for RAG

Uploads a plain text file to be processed, chunked, embedded, and added to the RAG knowledge base.

First, create a sample text file (e.g., `my_document.txt`):
```text
This is a sample document about artificial intelligence.
AI is transforming various industries.
Machine learning is a subset of AI.
Deep learning is a subset of machine learning.
```

Then, upload it:
```bash
curl -X POST -F "file=@my_document.txt" [http://127.0.0.1:5000/upload_text](http://127.0.0.1:5000/upload_text) | jq .
```

Expected successful response:
```json
{
  "message": "File my_document.txt uploaded and processed successfully."
}
```

#### 3. Chat Completions (OpenAI-like)

Sends a user query to the RAG-enabled Ollama LLM. The system will first retrieve relevant context from the uploaded documents and then pass it to Ollama along with the user's question.

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{ "messages": [{"role": "user", "content": "What is deep learning?"}], "max_tokens": 100, "temperature": 0.7 }' \
  [http://127.0.0.1:5000/v1/chat/completions](http://127.0.0.1:5000/v1/chat/completions) | jq .
```

Expected successful response (content will vary based on model and context):
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Deep learning is a subset of machine learning. It's a type of artificial intelligence.",
        "role": "assistant"
      }
    }
  ],
  "created": 1234567890,
  "id": "chatcmpl-xxxx",
  "model": "ollama-llama3-rag",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 17,
    "prompt_tokens": 50,
    "total_tokens": 67
  }
}
```

### Removing RAG Binary Data

The RAG system stores its FAISS index and document content in the `rag_data/` directory, which is mounted as a Docker volume. Ollama models are stored in `ollama_models/`, also a mounted volume.

To completely remove the RAG data and Ollama models from your host machine (and thus from future container runs):

```bash
# Stop the container first if it's running
docker stop ollama_rag_proxy_container

# Remove the container
docker rm ollama_rag_proxy_container

# Remove the local data directories
rm -rf ./rag_data
rm -rf ./ollama_models
```

This will clear all processed documents and downloaded Ollama models.

### Environment Variables

You can customize the application's behavior by modifying the environment variables defined in the `Dockerfile`.

* `RAG_DATA_DIR`: Directory inside the container for RAG data.
* `PROXY_PORT`: Port on which the Flask proxy listens inside the container.
* `OLLAMA_BASE_URL`: URL for the Ollama server inside the container.
* `CHUNK_SIZE`: Size of text chunks for RAG processing.
* `CHUNK_OVERLAP`: Overlap between text chunks.
* `TOP_K_RAG`: Number of top-k documents to retrieve for RAG.
* `OLLAMA_MODEL_NAME`: Name of the Ollama model to use (e.g., `llama3`, `mistral`).
* `EMBEDDING_MODEL_NAME`: Name of the Sentence Transformer model for embeddings.

