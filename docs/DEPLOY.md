# STREAMLIT CLOUD DEPLOYMENT GUIDE

## Overview

This guide walks you through deploying the Fracture Detection AI app on **Streamlit Cloud** (https://share.streamlit.io).

## Key Challenges & Solutions

### Challenge 1: Large Model Files (~1GB)

**Problem:** Models (5 x ~200MB each) cannot be stored in GitHub and exceed Streamlit Cloud's limits.

**Solutions:**

1. **AWS S3 + Download on Startup** (Recommended)
2. **Google Cloud Storage + Download on Startup**
3. **Manual Upload via Streamlit Cloud**

### Challenge 2: Ollama (Local LLM)

**Problem:** Ollama cannot run in Streamlit Cloud environment.

**Solutions:**

1. Run Ollama on separate VM and provide URL
2. Use cloud APIs (OpenAI, Anthropic, etc.)
3. Disable chat feature for cloud deployment

---

## Step-by-Step Deployment

### Step 1: Prepare Your GitHub Repository

```bash
# 1. Ensure models are in .gitignore
git status  # Verify *.pth files are NOT listed

# 2. Push all code to GitHub
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Choose Model Storage Method

#### Option A: AWS S3 (Recommended)

**2A.1: Set up AWS S3 bucket**

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region

# Create bucket
aws s3 mb s3://fracture-detection-models --region us-east-1

# Upload models
aws s3 sync ./outputs s3://fracture-detection-models/models --include "best_*.pth"
```

**2A.2: Generate pre-signed URLs**

```bash
python -c "
from src.config.cloud_deployment import generate_s3_presigned_urls
import json
urls = generate_s3_presigned_urls()
print(json.dumps(urls, indent=2))
"
```

**2A.3: Copy the URLs and save them for Step 4**

#### Option B: Google Cloud Storage

```bash
# Install GCS tools
pip install google-cloud-storage

# Upload to GCS
gsutil -m cp outputs/best_*.pth gs://your-bucket/models/
```

#### Option C: Manual Upload

You'll upload models directly via Streamlit Cloud dashboard (we'll cover this in Step 4).

---

### Step 3: Configure Environment Variables

Create `.streamlit/secrets.toml` in Streamlit Cloud with:

```toml
# Model download URLs (from Step 2 Option A/B)
swin_model_url = "https://..."
mobilenetv2_model_url = "https://..."
densenet_model_url = "https://..."
efficientnet_model_url = "https://..."
maxvit_model_url = "https://..."

# Ollama Configuration (optional)
# Option 1: If running Ollama on separate server
ollama_host = "http://your-server:11434"

# Option 2: Use cloud LLM API instead
use_cloud_llm = true
cloud_llm_provider = "openai"  # or "anthropic"
cloud_llm_api_key = "your-api-key"

# Deployment flag
streamlit_deployment = true
```

---

### Step 4: Deploy on Streamlit Cloud

**4.1: Go to Streamlit Cloud**

- Visit https://share.streamlit.io
- Click "New app"

**4.2: Connect to GitHub**

- Select your repository: `DJSACM-Research/MedAI-ExplainableFractureDetection`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Click "Deploy"

**4.3: Add Secrets**

- Once deployed, go to Settings (⚙️) > Secrets
- Paste the contents from `.streamlit/secrets.toml`
- Click "Save"

**4.4: If using Manual Upload**

- Wait for initial app load (it will fail looking for models)
- See error message for what's needed
- Use Streamlit's file system to upload models (contact support if needed)

---

### Step 5: Test Deployment

Once deployed (URL will be like `https://your-app-name.streamlit.app/`):

1. **Check Model Availability**

   - Open the app
   - Look for model status messages
   - If "Models not found": Check environment variables

2. **Test Image Upload**

   - Go to "Complete Workflow" tab
   - Upload a sample X-ray
   - Click "Run Complete Workflow"

3. **Test Chat (if Ollama configured)**
   - Go to "Patient Chat" tab
   - Verify Ollama connection status
   - Ask a question about diagnosis

---

## Troubleshooting

### Problem: "Models not found"

**Solution:**

```bash
# 1. Verify environment variables in Streamlit Cloud Secrets
# 2. Check S3/GCS URLs are correct and accessible
# 3. Run locally to test:
python -c "from src.utils.model_manager import initialize_models_for_deployment; initialize_models_for_deployment()"
```

### Problem: "Ollama not connected"

**Solution:**

1. Verify `OLLAMA_HOST` is correct in Secrets
2. Check Ollama server is running and accessible
3. Consider using cloud API instead

### Problem: App times out during startup

**Solution:**

- Model download takes time
- Increase Streamlit timeout in Secrets:

```toml
[client]
maxMessageSize = 200

[server]
maxUploadSize = 500
timeout = 300  # 5 minutes
```

---

## Performance Tips

1. **Cache model downloads**

   - Models persist in Streamlit Cloud's temporary storage
   - First load takes ~2-3 minutes
   - Subsequent loads are instant

2. **Optimize image sizes**

   - Streamlit Cloud has bandwidth limits
   - Compress X-ray images before uploading

3. **Use regional storage**
   - Place S3/GCS bucket in same region as Streamlit Cloud
   - Reduces latency

---

## Cost Considerations

- **Streamlit Cloud**: Free tier available
- **AWS S3**: ~$0.01/GB/month for storage + bandwidth
- **Cloud LLM API**: ~$0.0015/1K tokens (OpenAI GPT-3.5)
- **Ollama Server VM**: $5-20/month depending on hosting

---

## Next Steps

1. Choose model storage method (S3 recommended)
2. Upload models to cloud storage
3. Generate pre-signed URLs or credentials
4. Deploy on Streamlit Cloud
5. Add environment variables/secrets
6. Test all features

Good luck with your deployment! 🚀
