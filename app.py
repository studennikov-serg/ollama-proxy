import os
import requests
import asyncio
from flask import Flask, request, jsonify, send_from_directory, Response
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import logging
from threading import Lock
import time

# Corrected import for Ollama
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

# Define the path for the application log file within the RAG_DATA_DIR
RAG_DATA_DIR = os.environ.get("RAG_DATA_DIR","/app/rag_data")

# Create the RAG data directory if it doesn't exist. This ensures the log file path is valid.
os.makedirs(RAG_DATA_DIR, exist_ok=True)

# Configure logging to both a file and the console
LOG_FILE_PATH = os.path.join(RAG_DATA_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH), # Log to a file
        logging.StreamHandler() # Also log to console (stdout/stderr) for docker logs
    ]
)

app = Flask(__name__)

# --- Configuration ---
# OLLAMA_BASE_URL: This will point to the Ollama server running inside the same container.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434") # Default Ollama port
FAISS_INDEX_PATH = os.path.join(RAG_DATA_DIR, "faiss_index.bin")
DOC_STORE_PATH = os.path.join(RAG_DATA_DIR, "documents.json")

# RAG parameters
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "70"))
TOP_K_RAG = int(os.environ.get("TOP_K_RAG", "4"))
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "llama3")

# Global variables for RAG components
model = None
faiss_index = None
documents = []
index_lock = Lock() # For thread-safe index operations

# --- Helper Functions ---
def initialize_embedding_model():
    global model
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Embedding model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}", exc_info=True)
        # Re-raise to prevent app from starting without model
        raise

def load_rag_data():
    global faiss_index, documents
    logging.info(f"Attempting to load RAG data from {RAG_DATA_DIR}...")
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOC_STORE_PATH):
        try:
            with index_lock:
                faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                with open(DOC_STORE_PATH, 'r') as f:
                    documents = json.load(f)
            logging.info(f"Loaded {len(documents)} documents and FAISS index with {faiss_index.ntotal} vectors.")
        except Exception as e:
            logging.error(f"Error loading RAG data: {e}", exc_info=True)
            faiss_index = None # Ensure index is reset if loading fails
            documents = []
    else:
        logging.info("No existing RAG data found. Starting with empty index.")
        faiss_index = None # Will be initialized as empty later if no data

def add_documents_to_index(new_documents_content):
    global faiss_index, documents
    if not new_documents_content:
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs_for_embeddings = []
    for doc_content in new_documents_content:
        chunks = text_splitter.split_text(doc_content)
        for chunk in chunks:
            docs_for_embeddings.append(LangchainDocument(page_content=chunk))

    if not docs_for_embeddings:
        logging.info("No chunks generated from new documents for embedding.")
        return

    logging.info(f"Generating embeddings for {len(docs_for_embeddings)} chunks...")
    try:
        # SentenceTransformer directly takes a list of strings
        texts_to_embed = [doc.page_content for doc in docs_for_embeddings]
        new_embeddings = model.encode(texts_to_embed, convert_to_numpy=True).astype('float32')

        with index_lock:
            if faiss_index is None:
                logging.info(f"Initializing new FAISS index with dimension {new_embeddings.shape[1]}.")
                faiss_index = faiss.IndexFlatL2(new_embeddings.shape[1])
            faiss_index.add(new_embeddings)
            documents.extend([{"content": doc.page_content} for doc in docs_for_embeddings]) # Store original content
        logging.info(f"Added {len(new_embeddings)} embeddings to FAISS index. Total vectors: {faiss_index.ntotal}")

        # Save updated index and documents
        save_rag_data()
    except Exception as e:
        logging.error(f"Error adding documents to index: {e}", exc_info=True)

def save_rag_data():
    global faiss_index, documents
    logging.info(f"Saving RAG data to {RAG_DATA_DIR}...")
    try:
        with index_lock:
            faiss.write_index(faiss_index, FAISS_INDEX_PATH)
            with open(DOC_STORE_PATH, 'w') as f:
                json.dump(documents, f, indent=2)
        logging.info("RAG data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving RAG data: {e}", exc_info=True)


# --- API Endpoints ---

@app.route('/upload_text', methods=['POST'])
async def upload_text():
    """Endpoint to upload text files for RAG."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            content = file.read().decode('utf-8')
            add_documents_to_index([content])
            logging.info(f"File {file.filename} uploaded and processed.")
            return jsonify({"message": f"File {file.filename} uploaded and processed successfully."}), 200
        except Exception as e:
            logging.error(f"Error processing uploaded file: {e}", exc_info=True)
            return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

@app.route('/v1/chat/completions', methods=['POST'])
async def chat_completions():
    """Handles chat completion requests with RAG."""
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({"error": "Invalid request: 'messages' field is required."}), 400

        messages = data['messages']
        user_message_content = None
        for message in messages:
            if message.get('role') == 'user' and message.get('content'):
                user_message_content = message['content']
                break

        if not user_message_content:
            return jsonify({"error": "No user message found in the request."}), 400

        logging.info(f"Received chat completion request: {user_message_content}")

        context = ""
        if faiss_index and documents:
            try:
                # Generate embedding for the user query
                query_embedding = model.encode([user_message_content], convert_to_numpy=True).astype('float32')

                # Perform similarity search
                with index_lock:
                    D, I = faiss_index.search(query_embedding, TOP_K_RAG) # D is distances, I is indices
                
                retrieved_documents = [documents[int(i)]['content'] for i in I[0] if i != -1 and i < len(documents)]
                context = "\n\n".join(retrieved_documents)
                logging.info(f"Retrieved {len(retrieved_documents)} documents for RAG.")
            except Exception as e:
                logging.warning(f"Error during RAG retrieval, proceeding without context: {e}", exc_info=True)
                context = "" # Continue without RAG if retrieval fails

        # 2. Construct prompt for Ollama
        if context:
            prompt = (
                f"Based on the following context, answer the question:\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_message_content}\n\n"
                f"Answer:"
            )
        else:
            prompt = user_message_content

        # 3. Call Ollama (llama.cpp)
        ollama_model = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL_NAME) # Changed LLAMA_CPP_BASE_URL to OLLAMA_BASE_URL

        # Langchain's invoke method can be synchronous
        llm_response_content = ollama_model.invoke(prompt)

        # 4. Return response in OpenAI-like format
        response_payload = {
            "id": "chatcmpl-xxxx", # Dummy ID
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"ollama-{OLLAMA_MODEL_NAME}-rag",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": llm_response_content.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()), # Basic token estimation
                "completion_tokens": len(llm_response_content.split()),
                "total_tokens": len(prompt.split()) + len(llm_response_content.split())
            }
        }
        return jsonify(response_payload), 200

    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error to Ollama server: {e}", exc_info=True)
        return jsonify({"error": "Failed to connect to Ollama server. Please ensure it's running and accessible."}), 500
    except Exception as e:
        # This will now log the full traceback to app.log AND console
        logging.error(f"Error processing chat completion: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    total_vectors = 0
    if faiss_index:
        try:
            total_vectors = faiss_index.ntotal
        except Exception as e:
            logging.error(f"Error getting total vectors from FAISS index: {e}", exc_info=True)
            total_vectors = -1 # Indicate error
    return jsonify({"status": "healthy", "faiss_index_total_vectors": total_vectors, "total_documents": len(documents)}), 200

# --- Initial Setup ---
# This will be called when the Flask app starts (e.g., by gunicorn)
with app.app_context():
    try:
        initialize_embedding_model()
        load_rag_data()
        # If after loading, index is still None (no data or error), initialize an empty one
        if faiss_index is None:
            logging.info("FAISS index not loaded or initialized. Creating an empty one.")
            # We need to know the embedding dimension from the model to create an empty index
            if model:
                sample_embedding_dim = model.get_sentence_embedding_dimension()
                faiss_index = faiss.IndexFlatL2(sample_embedding_dim)
                logging.info(f"Empty FAISS index created with dimension {sample_embedding_dim}.")
            else:
                logging.error("Embedding model not loaded, cannot initialize empty FAISS index.")
    except Exception as e:
        logging.critical(f"Fatal error during app initialization: {e}", exc_info=True)
        # Depending on criticality, you might want to exit here or return an error state
        # For now, let gunicorn handle the crash if it's truly unrecoverable.

if __name__ == '__main__':
    # This block is for direct Python execution (e.g., during development),
    # gunicorn will handle execution in production.
    app.run(host='0.0.0.0', port=5000, debug=True)
