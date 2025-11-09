"""
Cloud deployment configuration for model storage and management.
Supports AWS S3, Google Cloud Storage, and other cloud providers.
"""

import os
import json
from typing import Optional

# ============================================================================
# AWS S3 Configuration (if using S3 for model storage)
# ============================================================================

AWS_S3_CONFIG = {
    "bucket": os.getenv("AWS_S3_BUCKET", "your-bucket-name"),
    "region": os.getenv("AWS_REGION", "us-east-1"),
    "access_key": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
}

# ============================================================================
# Google Cloud Storage Configuration
# ============================================================================

GCS_CONFIG = {
    "project_id": os.getenv("GCP_PROJECT_ID", ""),
    "bucket": os.getenv("GCP_BUCKET", ""),
    "credentials_json": os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
}

# ============================================================================
# Model Download URLs
# ============================================================================

# These should be set as environment variables for security
# Example for AWS S3 pre-signed URLs:
# export SWIN_MODEL_URL="https://your-bucket.s3.amazonaws.com/best_swin.pth?..."

MODEL_DOWNLOAD_URLS = {
    "best_swin.pth": os.getenv("SWIN_MODEL_URL", ""),
    "best_mobilenetv2.pth": os.getenv("MOBILENETV2_MODEL_URL", ""),
    "best_densenet169.pth": os.getenv("DENSENET_MODEL_URL", ""),
    "best_efficientnetv2.pth": os.getenv("EFFICIENTNET_MODEL_URL", ""),
    "best_maxvit.pth": os.getenv("MAXVIT_MODEL_URL", ""),
}

# ============================================================================
# Ollama Configuration for Cloud Deployment
# ============================================================================

OLLAMA_CONFIG = {
    # For local deployment
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "model": os.getenv("OLLAMA_MODEL", "llama3"),
    
    # Alternative: Use cloud-hosted LLM API instead
    "use_cloud_api": os.getenv("USE_CLOUD_LLM", "False").lower() == "true",
    "cloud_api_provider": os.getenv("CLOUD_LLM_PROVIDER", "openai"),  # openai, anthropic, etc
    "cloud_api_key": os.getenv("CLOUD_LLM_API_KEY", ""),
}

# ============================================================================
# Streamlit Cloud Configuration
# ============================================================================

STREAMLIT_CLOUD_CONFIG = {
    "deployment_mode": os.getenv("STREAMLIT_DEPLOYMENT", "False").lower() == "true",
    "enable_model_download": os.getenv("ENABLE_MODEL_DOWNLOAD", "True").lower() == "true",
    "model_cache_size_mb": int(os.getenv("MODEL_CACHE_SIZE_MB", "1000")),
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_s3_client():
    """Create AWS S3 client."""
    try:
        import boto3
        return boto3.client(
            's3',
            region_name=AWS_S3_CONFIG["region"],
            aws_access_key_id=AWS_S3_CONFIG["access_key"],
            aws_secret_access_key=AWS_S3_CONFIG["secret_key"],
        )
    except ImportError:
        raise ImportError("boto3 not installed. Run: pip install boto3")


def get_gcs_client():
    """Create Google Cloud Storage client."""
    try:
        from google.cloud import storage
        return storage.Client(project=GCS_CONFIG["project_id"])
    except ImportError:
        raise ImportError("google-cloud-storage not installed. Run: pip install google-cloud-storage")


def upload_models_to_s3(local_model_dir: str = "./outputs") -> dict:
    """
    Upload local models to AWS S3.
    
    Args:
        local_model_dir: Directory containing model files
        
    Returns:
        Dictionary with upload results
    """
    from pathlib import Path
    
    client = get_s3_client()
    results = {}
    
    for model_file in Path(local_model_dir).glob("best_*.pth"):
        try:
            key = f"models/{model_file.name}"
            print(f"Uploading {model_file.name} to S3...")
            client.upload_file(
                str(model_file),
                AWS_S3_CONFIG["bucket"],
                key,
                Callback=None
            )
            results[model_file.name] = {"status": "success", "s3_key": key}
            print(f"✅ Uploaded {model_file.name}")
        except Exception as e:
            results[model_file.name] = {"status": "failed", "error": str(e)}
            print(f"❌ Failed to upload {model_file.name}: {e}")
    
    return results


def upload_models_to_gcs(local_model_dir: str = "./outputs") -> dict:
    """
    Upload local models to Google Cloud Storage.
    
    Args:
        local_model_dir: Directory containing model files
        
    Returns:
        Dictionary with upload results
    """
    from pathlib import Path
    
    client = get_gcs_client()
    bucket = client.bucket(GCS_CONFIG["bucket"])
    results = {}
    
    for model_file in Path(local_model_dir).glob("best_*.pth"):
        try:
            blob = bucket.blob(f"models/{model_file.name}")
            print(f"Uploading {model_file.name} to GCS...")
            blob.upload_from_filename(str(model_file))
            results[model_file.name] = {"status": "success", "gs_url": blob.public_url}
            print(f"✅ Uploaded {model_file.name}")
        except Exception as e:
            results[model_file.name] = {"status": "failed", "error": str(e)}
            print(f"❌ Failed to upload {model_file.name}: {e}")
    
    return results


def generate_s3_presigned_urls() -> dict:
    """Generate S3 pre-signed URLs for models."""
    client = get_s3_client()
    urls = {}
    
    for model_name in MODEL_DOWNLOAD_URLS.keys():
        key = f"models/{model_name}"
        try:
            url = client.generate_presigned_url(
                'get_object',
                Params={'Bucket': AWS_S3_CONFIG["bucket"], 'Key': key},
                ExpiresIn=3600 * 24 * 7  # 7 days
            )
            urls[model_name] = url
        except Exception as e:
            print(f"Error generating URL for {model_name}: {e}")
    
    return urls


def print_deployment_checklist():
    """Print deployment checklist."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    STREAMLIT CLOUD DEPLOYMENT CHECKLIST                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. GITHUB SETUP
   ☐ Repository pushed to GitHub
   ☐ .gitignore excludes *.pth files
   ☐ README.md describes the project
   ☐ requirements-prod.txt is in root

2. MODEL STORAGE (Choose one)
   ☐ AWS S3 Setup:
     - Created S3 bucket
     - Uploaded models
     - Generated pre-signed URLs
     - Set environment variables (SWIN_MODEL_URL, etc.)
   
   OR
   
   ☐ Google Cloud Storage Setup:
     - Created GCS bucket
     - Uploaded models
     - Set environment variables
   
   OR
   
   ☐ Manual Upload:
     - Will upload models manually to Streamlit Cloud

3. ENVIRONMENT VARIABLES (in Streamlit Cloud Secrets)
   ☐ OLLAMA_HOST (if using external Ollama server)
   ☐ OLLAMA_MODEL (default: llama3)
   ☐ Model download URLs or credentials
   ☐ Cloud provider credentials (if applicable)

4. STREAMLIT CLOUD DEPLOYMENT
   ☐ Created account at share.streamlit.io
   ☐ Connected GitHub repository
   ☐ Configured Secrets
   ☐ Deployed app

5. TESTING
   ☐ App loads successfully
   ☐ Models are available
   ☐ Chat feature works (if Ollama is configured)
   ☐ Workflow can run end-to-end

═══════════════════════════════════════════════════════════════════════════════

IMPORTANT NOTES:
- Each model is ~200MB, total ~1GB
- Streamlit Cloud max storage is ~1GB
- Models must be downloaded/cached on startup
- Ollama requires external server (not available in Streamlit Cloud)
- For chat feature, consider using cloud APIs (OpenAI, Anthropic)

═══════════════════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    print("Cloud Deployment Configuration")
    print_deployment_checklist()
    
    print("\n📋 Current Configuration:")
    print(f"  Deployment Mode: {STREAMLIT_CLOUD_CONFIG['deployment_mode']}")
    print(f"  Ollama Host: {OLLAMA_CONFIG['host']}")
    print(f"  Use Cloud API: {OLLAMA_CONFIG['use_cloud_api']}")
