# HUGGING FACE INFERENCE API SETUP

## Overview

The app now uses **Hugging Face Inference API** instead of Ollama for the chat feature. This provides:

- ✅ No local LLM server needed
- ✅ Free tier available (with rate limits)
- ✅ Works seamlessly in Streamlit Cloud
- ✅ Powered by Mistral 7B (open-source, performant)

## Getting Your Hugging Face API Key

### Step 1: Create Hugging Face Account
- Go to https://huggingface.co
- Sign up or log in

### Step 2: Generate API Token
1. Click your profile icon (top right) → Settings
2. Click "Access Tokens" in left sidebar
3. Click "New token"
4. Name it (e.g., "streamlit-fracture-app")
5. Select "Read" access
6. Click "Generate"
7. **Copy the token** (starts with `hf_...`)

## Local Development Setup

### Add to `.streamlit/secrets.toml`:

```toml
huggingface_api_key = "hf_your_actual_token_here"
```

Then run locally:
```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud Deployment Setup

### Step 1: Deploy Your App
- Go to https://share.streamlit.io
- Click "New app"
- Select repo and branch
- Main file: `streamlit_app.py`
- Click "Deploy"

### Step 2: Add Secrets
1. Once deployed, click Settings ⚙️
2. Click "Secrets"
3. Add your Hugging Face token:

```toml
HUGGINGFACE_API_KEY = "hf_your_actual_token_here"
```

**Note:** The key must be `HUGGINGFACE_API_KEY` (uppercase, with underscore) in Streamlit Cloud Secrets.

4. Click "Save"

### Step 3: Refresh App
- The app will automatically use your token
- The chat feature should now work!

## Testing the Setup

1. Run the app
2. Go to "⚙️ Complete Workflow" tab
3. Upload an X-ray image
4. Click "Run Complete Workflow"
5. Go to "💬 Patient Chat" tab
6. Ask a question

## App Versions

### For Local Development (with Ollama):
```bash
streamlit run apps/patient_chat_app_local.py
```

### For Streamlit Cloud (with Hugging Face):
```bash
streamlit run streamlit_app.py
# or
streamlit run apps/patient_chat_app_cloud.py
```

## Troubleshooting

### Issue: "HUGGINGFACE_API_KEY not found"
**Solution:**
- In Streamlit Cloud, go to Settings > Secrets
- Add your token: `HUGGINGFACE_API_KEY = "hf_..."`
- Refresh the app

### Issue: "Error from API: Model is loading"
**Solution:**
- The model is starting up for the first time
- Wait a few seconds and try again
- Subsequent requests will be faster

### Issue: "API rate limit reached"
**Solution:**
- You've exceeded the free tier rate limit
- Either wait a few hours or upgrade your Hugging Face account
- Free tier: ~1000 requests/hour

### Issue: "Request timed out"
**Solution:**
- The model took too long to respond
- Try a shorter prompt
- Or use a different Hugging Face model

## Available Models

Current: **Mistral 7B Instruct** (`mistralai/Mistral-7B-Instruct-v0.2`)

Alternative options you can use by modifying `apps/patient_chat_app_cloud.py`:

1. **Zephyr 7B** (faster, smaller)
   ```python
   HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
   ```

2. **OpenHermes 2.5** (good for medical context)
   ```python
   HF_API_URL = "https://api-inference.huggingface.co/models/teknium/OpenHermes-2.5-Mistral-7B"
   ```

3. **Neural Chat** (optimized for conversation)
   ```python
   HF_API_URL = "https://api-inference.huggingface.co/models/Intel/neural-chat-7b-v3-1"
   ```

## Cost Estimation

| Tier | Cost | Monthly Requests |
|------|------|------------------|
| Free | $0 | ~1000/hour |
| Pro | $9/month | Unlimited |
| Enterprise | Custom | Custom |

For production with high usage, consider upgrading to Pro or self-hosting.

## Files Modified

- ✅ `apps/patient_chat_app_cloud.py` - Cloud version (NEW, uses Hugging Face)
- ✅ `apps/patient_chat_app_local.py` - Local version (renamed, uses Ollama)
- ✅ `streamlit_app.py` - Entry point (now uses cloud version)
- ✅ `.streamlit/secrets.toml` - Added HF API key template

## Next Steps

1. Get your Hugging Face token from https://huggingface.co/settings/tokens
2. Add it to `.streamlit/secrets.toml` for local testing
3. Deploy to Streamlit Cloud and add it to Secrets there
4. Test the chat feature with a sample X-ray!

---

**Questions?** Check the Hugging Face documentation:
- https://huggingface.co/docs/api-inference/index
- https://huggingface.co/models?pipeline_tag=text-generation

Good luck! 🚀
