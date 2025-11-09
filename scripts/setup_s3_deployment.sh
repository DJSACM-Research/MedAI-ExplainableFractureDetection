#!/bin/bash

# AWS S3 Quick Setup Script for Streamlit Cloud Deployment
# This script helps upload models to AWS S3

set -e

echo "🚀 AWS S3 Setup for Streamlit Cloud Deployment"
echo "================================================"
echo ""

# Check if models exist
if [ ! -d "outputs" ] || [ -z "$(ls -1 outputs/best_*.pth 2>/dev/null)" ]; then
    echo "❌ ERROR: No model files found in ./outputs/"
    echo "   Please ensure all model checkpoints are present:"
    echo "   - best_swin.pth"
    echo "   - best_mobilenetv2.pth"
    echo "   - best_densenet169.pth"
    echo "   - best_efficientnetv2.pth"
    echo "   - best_maxvit.pth"
    exit 1
fi

echo "✅ Found model files:"
du -h outputs/best_*.pth

# Check AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo ""
    echo "⚠️ AWS CLI not found. Installing..."
    pip install awscli
fi

echo ""
echo "📋 AWS S3 Setup Steps:"
echo "1. Configure AWS credentials (if not already done)"
echo "   Run: aws configure"
echo ""
echo "2. Enter your AWS credentials when prompted"
echo ""

# Ask for bucket name
echo "Enter AWS S3 bucket name (e.g., fracture-detection-models):"
read BUCKET_NAME

if [ -z "$BUCKET_NAME" ]; then
    echo "❌ Bucket name cannot be empty"
    exit 1
fi

# Ask for region
echo ""
echo "Enter AWS region (default: us-east-1):"
read REGION
REGION=${REGION:-us-east-1}

echo ""
echo "Creating S3 bucket: $BUCKET_NAME in region $REGION..."
aws s3 mb s3://$BUCKET_NAME --region $REGION 2>/dev/null || echo "⚠️ Bucket may already exist"

echo ""
echo "📤 Uploading models to S3..."
aws s3 sync outputs/ s3://$BUCKET_NAME/models/ --include "best_*.pth" --region $REGION

echo ""
echo "✅ Models uploaded successfully!"
echo ""
echo "🔗 Generating pre-signed URLs for Streamlit Secrets..."
echo ""

cat > /tmp/generate_urls.py << 'EOF'
import subprocess
import json
import sys

bucket_name = sys.argv[1]
region = sys.argv[2]

urls = {}
models = [
    "best_swin.pth",
    "best_mobilenetv2.pth",
    "best_densenet169.pth",
    "best_efficientnetv2.pth",
    "best_maxvit.pth",
]

for model in models:
    key = f"models/{model}"
    try:
        result = subprocess.run(
            ["aws", "s3", "presign", f"s3://{bucket_name}/{key}", "--region", region],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            urls[model] = result.stdout.strip()
    except Exception as e:
        print(f"Error generating URL for {model}: {e}", file=sys.stderr)

print(json.dumps(urls, indent=2))
EOF

python /tmp/generate_urls.py $BUCKET_NAME $REGION > /tmp/urls.json

echo ""
echo "📝 Add these environment variables to Streamlit Cloud Secrets:"
echo "=================================================="

python << 'EOF'
import json
with open('/tmp/urls.json', 'r') as f:
    urls = json.load(f)

for model, url in urls.items():
    env_var = model.replace("best_", "").upper() + "_MODEL_URL"
    print(f'{env_var} = "{url}"')

print("")
print("Plus these:")
print('STREAMLIT_DEPLOYMENT = "true"')
print('OLLAMA_HOST = "http://your-ollama-server:11434"  # optional')
EOF

echo ""
echo "=================================================="
echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy the environment variables above"
echo "2. Go to https://share.streamlit.io"
echo "3. Find your app > Settings > Secrets"
echo "4. Paste the environment variables"
echo "5. Your app will automatically download models on startup"
echo ""
