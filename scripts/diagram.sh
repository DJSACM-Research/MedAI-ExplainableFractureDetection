#!/bin/bash
# Generate workflow and architecture diagrams

echo "🎨 Generating MedAI system diagrams..."

# Create diagram directory if it doesn't exist
mkdir -p diagram

# Generate diagrams
python diagram/generate_workflow.py
python diagram/generate_architecture.py

echo "✅ Diagrams generated successfully!"
echo "   📄 diagram/workflow_diagram.png"
echo "   📄 diagram/architecture_diagram.png"
