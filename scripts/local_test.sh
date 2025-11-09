#!/bin/bash

# Local Docker Testing Script
# Tests the Docker container locally before Railway deployment

set -e

echo "🐳 Building Docker image..."
docker build -t fracture-detection:latest .

echo ""
echo "✅ Build successful!"
echo ""
echo "🚀 Starting Docker container..."
echo ""
echo "⏳ Container starting at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the container"
echo ""

# Run with HF API key from environment or prompt
if [ -z "$HUGGINGFACE_API_KEY" ]; then
    echo "⚠️  HUGGINGFACE_API_KEY not set in environment"
    echo "You can set it before running:"
    echo "export HUGGINGFACE_API_KEY=\"hf_your_token_here\""
    echo ""
    read -p "Enter your HF API key (or press Enter to skip): " hf_key
    if [ -z "$hf_key" ]; then
        echo "⚠️  Running without HF API key - HF features will not work"
        docker run -p 8501:8501 \
            -v "$(pwd)/outputs:/app/outputs" \
            fracture-detection:latest
    else
        docker run -p 8501:8501 \
            -e HUGGINGFACE_API_KEY="$hf_key" \
            -v "$(pwd)/outputs:/app/outputs" \
            fracture-detection:latest
    fi
else
    docker run -p 8501:8501 \
        -e HUGGINGFACE_API_KEY="$HUGGINGFACE_API_KEY" \
        -v "$(pwd)/outputs:/app/outputs" \
        fracture-detection:latest
fi
