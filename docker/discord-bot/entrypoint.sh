#!/bin/bash
set -e

# 1. Function to check if a service is ready
wait_for_service() {
    local url="$1"
    local service_name="$2"
    
    echo "Waiting for $service_name to be ready..."
    
    # Loop until the HTTP response code is 200
    until curl --output /dev/null --silent --head --fail "$url"; do
        printf '.'
        sleep 5
    done
    
    echo ""
    echo "$service_name is ready!"
}

# 2. Wait for ChromaDB (Using the heartbeat endpoint)
wait_for_service "http://chromadb:8000/api/v1/heartbeat" "ChromaDB"

# 3. Wait for vLLM (Using the models endpoint)
# Crucial: This ensures the model is actually loaded in GPU memory
wait_for_service "http://vllm:8000/v1/models" "vLLM Inference Engine"

echo "All systems operational. Launching Discord RAG Bot..."

# 4. Execute the main command
exec "$@"