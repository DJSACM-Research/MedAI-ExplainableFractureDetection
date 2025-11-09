# 🚀 Railway Deployment Guide

## Prerequisites

- GitHub account with access to private repo
- Hugging Face API token (free)
- Railway account (free)

## Step-by-Step Deployment

### 1. Prepare Your Repository

Ensure everything is committed:

```bash
git status
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

### 2. Create Railway Account

1. Visit [railway.app](https://railway.app)
2. Click **"Login"** and choose **"Login with GitHub"**
3. Authorize Railway to access your GitHub account

### 3. Connect Repository

1. In Railway dashboard, click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Click **"Configure GitHub App"** if prompted
4. Select **`DJSACM-Research/MedAI-ExplainableFractureDetection`**
5. Choose **"main"** branch
6. Click **"Deploy"**

### 4. Add Environment Variables

While it's building:

1. Click on your project
2. Go to **"Variables"** tab (top right)
3. Click **"Add New Variable"**
4. Add your secrets:

```
HUGGINGFACE_API_KEY = hf_your_token_here
STREAMLIT_SERVER_HEADLESS = true
STREAMLIT_CLIENT_SHOWERRORDETAILS = false
```

### 5. Wait for Deployment

Railway will:

- ✓ Clone your repo
- ✓ Build Docker image (~3-5 min)
- ✓ Deploy to container (~1 min)

### 6. Access Your App

Once deployed, you'll get a URL like:

```
https://fracture-detection-production.up.railway.app
```

Click it to view your live app!

## Troubleshooting

### App not loading?

- Check **"Logs"** tab in Railway
- Verify `HUGGINGFACE_API_KEY` is set correctly
- Ensure model files are in `outputs/` directory

### Port errors?

Railway auto-detects port 8501 from Dockerfile. If issues:

- Add `PORT=8501` in Variables

### Container crashing?

- Check logs for errors
- Ensure all required Python packages are in `requirements-prod.txt`
- Verify model files exist in outputs/

## Cost

- **Completely FREE** for the free tier
- 500 hours/month included
- Generous for hobby projects

## Domain Setup (Optional)

To use custom domain:

1. Railway → Settings → Domains
2. Add your domain and configure DNS

---

**✅ You're deployed!** Share your app URL with colleagues!
