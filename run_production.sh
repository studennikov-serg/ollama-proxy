# run_production.sh
#!/bin/bash
set -e

# Define the log file where all output will be stored
LOG_FILE="debug_run.log"

# Clear the log file from previous runs
> "$LOG_FILE"

echo "--- Script Started: $(date) ---" | tee -a "$LOG_FILE"
echo "Output will be logged to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# --- 0. STOP AND REMOVE ALL CONTAINERS AND THE SPECIFIC IMAGE (Clean Slate for this image) ---
echo "--- Stopping and removing all existing containers ---" | tee -a "$LOG_FILE"
docker stop $(docker ps -aq) 2>> "$LOG_FILE" || true | tee -a "$LOG_FILE"
docker rm $(docker ps -aq) 2>> "$LOG_FILE" || true | tee -a "$LOG_FILE"
echo "--- All containers cleared ---" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "--- Removing existing Docker image 'ollama_rag_proxy_image' to force a fresh build ---" | tee -a "$LOG_FILE"
docker rmi ollama_rag_proxy_image 2>> "$LOG_FILE" || true | tee -a "$LOG_FILE"
echo "--- Existing Docker image removed (if it existed) ---" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"


# --- 1. BUILD THE DOCKER IMAGE ---
echo "--- Building Docker image with --no-cache (forces fresh build of all layers) ---" | tee -a "$LOG_FILE"
# Capture build output, errors to log file
docker build --no-cache -t ollama_rag_proxy_image . > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)
if [ $? -ne 0 ]; then
    echo "ERROR: Docker image build failed. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
fi
echo "--- Docker image built successfully ---" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# --- 2. RUN THE CONTAINER ---
echo "--- Running container 'ollama_rag_proxy_container' ---" | tee -a "$LOG_FILE"
# The -d flag runs the container in detached (background) mode
# Redirect docker run output to log file
docker run -d \
    --name ollama_rag_proxy_container \
    -p 5000:5000 \
    -p 11434:11434 \
    -v "$(pwd)/rag_data:/app/rag_data" \
    -v "$(pwd)/ollama_models:/root/.ollama/models" \
    ollama_rag_proxy_image > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)

if [ $? -ne 0 ]; then
    echo "ERROR: Docker container failed to start. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
fi

echo "--- Container started in detached mode. ---" | tee -a "$LOG_FILE"
echo "--- Waiting 30 seconds for Ollama server and Flask proxy to fully initialize... ---" | tee -a "$LOG_FILE"
sleep 30 # Increased initial wait time for Ollama to start and potentially pull model

# --- 3. RETRIEVE INITIAL CONTAINER LOGS ---
echo "" | tee -a "$LOG_FILE"
echo "--- Initial Container Logs: ---" | tee -a "$LOG_FILE"
docker logs ollama_rag_proxy_container > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)
echo "------------------------------" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# --- 4. PERFORM HEALTH CHECKS AND API TESTS ---

# Health Check for Flask Proxy
echo "--- Waiting for Flask proxy to become available at http://127.0.0.1:5000/health... ---" | tee -a "$LOG_FILE"
CURL_HEALTH_STATUS=""
for i in $(seq 1 60); do # Wait up to 5 minutes (60 * 5 seconds)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/health 2>> "$LOG_FILE")
    if [ "$HTTP_CODE" -eq "200" ]; then
        echo "Flask proxy is up!" | tee -a "$LOG_FILE"
        CURL_HEALTH_STATUS="200"
        break
    else
        echo "Waiting for proxy... (Attempt $((i*5))/300 seconds, HTTP Code: $HTTP_CODE)" | tee -a "$LOG_FILE"
        sleep 5
    fi
    if [ "$i" -eq 60 ]; then
        echo "Error: Flask proxy did not become available within 5 minutes." | tee -a "$LOG_FILE"
        CURL_HEALTH_STATUS="TIMEOUT"
    fi
done

if [ "$CURL_HEALTH_STATUS" != "200" ]; then
    echo "Attempting to retrieve full container logs for more details (this will show why the Flask app failed):" | tee -a "$LOG_FILE"
    docker logs ollama_rag_proxy_container > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)
    echo "Flask proxy health check FAILED. Exiting." | tee -a "$LOG_FILE"
    exit 1 # Exit if health check fails
fi

echo "" | tee -a "$LOG_FILE"
echo "--- Performing Health Check Curl Test ---" | tee -a "$LOG_FILE"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/health 2>> "$LOG_FILE")
echo "Health Check HTTP Status Code: $HTTP_STATUS" | tee -a "$LOG_FILE"
echo "Health Check Response Body:" | tee -a "$LOG_FILE"
# Capture the raw response body without piping to jq immediately
HEALTH_RESPONSE_BODY=$(curl -s http://127.0.0.1:5000/health 2>> "$LOG_FILE")
echo "$HEALTH_RESPONSE_BODY" | tee -a "$LOG_FILE" # Print raw body

if [ "$HTTP_STATUS" -eq "200" ]; then
    echo "Health Check: SUCCESS" | tee -a "$LOG_FILE"
    # If successful, try to parse with jq, but don't fail the script if jq fails
    echo "$HEALTH_RESPONSE_BODY" | jq . 2> /dev/null | tee -a "$LOG_FILE" || echo "Note: Health Check response body is not valid JSON, or jq not available." | tee -a "$LOG_FILE"
else
    echo "Health Check: FAILED" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "--- Full Container Logs for Debugging Health Check Error: ---" | tee -a "$LOG_FILE"
    docker logs ollama_rag_proxy_container > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)
    echo "-------------------------------------------------------------" | tee -a "$LOG_FILE"
fi
echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"


# RAG File Upload Test
echo "--- Performing RAG File Upload Curl Test ---" | tee -a "$LOG_FILE"
DOCUMENT_CONTENT="This is a test document for RAG processing. It contains sample information that will be used for indexing and retrieval."
# Save content to a temporary file for curl to upload
echo "$DOCUMENT_CONTENT" > /tmp/test_rag_document.txt

HTTP_STATUS_UPLOAD=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
-F "file=@/tmp/test_rag_document.txt" \
http://127.0.0.1:5000/upload_text 2>> "$LOG_FILE")
echo "File Upload HTTP Status Code: $HTTP_STATUS_UPLOAD" | tee -a "$LOG_FILE"
echo "File Upload Response Body:" | tee -a "$LOG_FILE"
# Capture the raw response body without piping to jq immediately
UPLOAD_RESPONSE_BODY=$(curl -s -X POST -F "file=@/tmp/test_rag_document.txt" http://127.0.0.1:5000/upload_text 2>> "$LOG_FILE")
echo "$UPLOAD_RESPONSE_BODY" | tee -a "$LOG_FILE" # Print raw body

if [ "$HTTP_STATUS_UPLOAD" -eq "200" ]; then
    echo "File Upload: SUCCESS" | tee -a "$LOG_FILE"
    # If successful, try to parse with jq, but don't fail the script if jq fails
    echo "$UPLOAD_RESPONSE_BODY" | jq . 2> /dev/null | tee -a "$LOG_FILE" || echo "Note: Response body is not valid JSON, or jq not available." | tee -a "$LOG_FILE"
else
    echo "File Upload: FAILED (HTTP Status: $HTTP_STATUS_UPLOAD)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "--- Full Container Logs for Debugging File Upload Error: ---" | tee -a "$LOG_FILE"
    docker logs ollama_rag_proxy_container > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)
    echo "----------------------------------------------------------" | tee -a "$LOG_FILE"
fi
rm /tmp/test_rag_document.txt # Clean up temporary file
echo "------------------------------------------" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"


# Ollama Chat Completion Test via Flask Proxy
echo "--- Performing Ollama Chat Completion Curl Test via Flask Proxy ---" | tee -a "$LOG_FILE"
HTTP_STATUS_CHAT=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{ "messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 100, "temperature": 0.7 }' http://127.0.0.1:5000/v1/chat/completions 2>> "$LOG_FILE")
echo "Ollama Chat Completion HTTP Status Code: $HTTP_STATUS_CHAT" | tee -a "$LOG_FILE"
echo "Ollama Chat Completion Response Body:" | tee -a "$LOG_FILE"
# Capture the raw response body without piping to jq immediately
CHAT_RESPONSE_BODY=$(curl -s -X POST -H "Content-Type: application/json" -d '{ "messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 100, "temperature": 0.7 }' http://127.0.0.1:5000/v1/chat/completions 2>> "$LOG_FILE")
echo "$CHAT_RESPONSE_BODY" | tee -a "$LOG_FILE" # Print raw body (will be HTML on 500)

if [ "$HTTP_STATUS_CHAT" -eq "200" ]; then
    echo "Ollama Chat Completion: SUCCESS" | tee -a "$LOG_FILE"
    # If successful, try to parse with jq, but don't fail the script if jq fails
    echo "$CHAT_RESPONSE_BODY" | jq . 2> /dev/null | tee -a "$LOG_FILE" || echo "Note: Response body is not valid JSON, or jq not available." | tee -a "$LOG_FILE"
else
    echo "Ollama Chat Completion: FAILED (HTTP Status: $HTTP_STATUS_CHAT)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "--- Full Container Logs for Debugging Ollama Chat Completion Error: ---" | tee -a "$LOG_FILE"
    docker logs ollama_rag_proxy_container > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)
    echo "---------------------------------------------------------------------" | tee -a "$LOG_FILE"
fi
echo "--------------------------------------------------------" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# --- FINAL STATUS SUMMARY ---
echo "--- FINAL STATUS SUMMARY ---" | tee -a "$LOG_FILE"
echo "Health Check Result: $([ "$CURL_HEALTH_STATUS" -eq "200" ] && echo "SUCCESS" || echo "FAILED")" | tee -a "$LOG_FILE"
echo "File Upload Result: $([ "$HTTP_STATUS_UPLOAD" -eq "200" ] && echo "SUCCESS" || echo "FAILED")" | tee -a "$LOG_FILE"
echo "Ollama Chat Completion Result: $([ "$HTTP_STATUS_CHAT" -eq "200" ] && echo "SUCCESS" || echo "FAILED")" | tee -a "$LOG_FILE"
echo "----------------------------" | tee -a "$LOG_FILE"
