#!/bin/bash
set -e

echo "🚀 Starting Fracture Detection Streamlit App..."

# Check if models exist
if [ ! -d "outputs" ]; then
    echo "❌ ERROR: ./outputs directory not found!"
    echo "   Please ensure all model checkpoints are in ./outputs/"
    exit 1
fi

# Check for at least one model
MODEL_COUNT=$(ls -1 outputs/best_*.pth 2>/dev/null | wc -l)
echo "✅ Found $MODEL_COUNT model checkpoints"

# Check Ollama availability (optional, but recommended)
echo "⚠️  Checking Ollama availability..."
if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "⚠️  WARNING: Ollama not running on localhost:11434"
    echo "   Chat feature will be unavailable"
    echo "   Start Ollama with: ollama serve"
else
    echo "✅ Ollama is running"
fi

# Start Streamlit
echo "📱 Launching Streamlit on http://localhost:8501"
streamlit run apps/patient_chat_app.py --logger.level=info
