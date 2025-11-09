"""
Model management utility for cloud deployments.
Handles downloading and caching models from cloud storage.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
import requests

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Model registry - Update these URLs with your cloud storage
MODEL_REGISTRY = {
    "best_swin.pth": {
        "size_mb": 200,
        # Replace with your actual cloud storage URL
        "url": os.getenv("SWIN_MODEL_URL", ""),
        "hash": "",  # Optional: SHA256 hash for verification
    },
    "best_mobilenetv2.pth": {
        "size_mb": 100,
        "url": os.getenv("MOBILENETV2_MODEL_URL", ""),
        "hash": "",
    },
    "best_densenet169.pth": {
        "size_mb": 200,
        "url": os.getenv("DENSENET_MODEL_URL", ""),
        "hash": "",
    },
    "best_efficientnetv2.pth": {
        "size_mb": 180,
        "url": os.getenv("EFFICIENTNET_MODEL_URL", ""),
        "hash": "",
    },
    "best_maxvit.pth": {
        "size_mb": 220,
        "url": os.getenv("MAXVIT_MODEL_URL", ""),
        "hash": "",
    },
}

MODELS_DIR = Path("./outputs")
MODELS_DIR.mkdir(exist_ok=True)


def check_model_exists(model_name: str) -> bool:
    """Check if a model file exists locally."""
    model_path = MODELS_DIR / model_name
    return model_path.exists()


def get_all_models_status() -> Dict[str, Dict]:
    """Get status of all models."""
    status = {}
    for model_name, config in MODEL_REGISTRY.items():
        exists = check_model_exists(model_name)
        status[model_name] = {
            "exists": exists,
            "size_mb": config["size_mb"],
            "url": config["url"],
        }
    return status


def download_model(model_name: str, force: bool = False) -> bool:
    """
    Download a model from cloud storage.
    
    Args:
        model_name: Name of the model file
        force: Force download even if file exists
        
    Returns:
        True if successful, False otherwise
    """
    if not force and check_model_exists(model_name):
        print(f"✅ {model_name} already exists locally")
        return True
    
    if model_name not in MODEL_REGISTRY:
        print(f"❌ {model_name} not found in registry")
        return False
    
    config = MODEL_REGISTRY[model_name]
    url = config.get("url")
    
    if not url:
        print(f"⚠️ No download URL configured for {model_name}")
        print(f"   Set environment variable or update MODEL_REGISTRY")
        return False
    
    try:
        print(f"📥 Downloading {model_name} from cloud storage...")
        response = requests.get(url, timeout=300, stream=True)
        response.raise_for_status()
        
        model_path = MODELS_DIR / model_name
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"  Progress: {percent:.1f}%", end='\r')
        
        print(f"\n✅ Successfully downloaded {model_name}")
        return True
    
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False


def download_all_models() -> Dict[str, bool]:
    """Download all models that have URLs configured."""
    results = {}
    for model_name in MODEL_REGISTRY:
        results[model_name] = download_model(model_name)
    return results


def initialize_models_for_deployment() -> bool:
    """
    Initialize models for deployment.
    Checks if models exist, attempts download if needed.
    
    Returns:
        True if all models are available, False otherwise
    """
    print("\n🔍 Checking model availability...")
    status = get_all_models_status()
    
    all_available = True
    for model_name, info in status.items():
        if info["exists"]:
            print(f"  ✅ {model_name}")
        else:
            print(f"  ❌ {model_name} - NOT FOUND")
            if info["url"]:
                print(f"     URL configured: {info['url'][:50]}...")
            else:
                print(f"     No download URL - configure via environment variables")
            all_available = False
    
    if not all_available:
        print("\n⚠️ Some models are missing!")
        print("   Option 1: Configure cloud storage URLs and run: python -c 'from src.utils.model_manager import download_all_models; download_all_models()'")
        print("   Option 2: Upload models manually to ./outputs/")
        return False
    
    print("\n✅ All models are available!")
    return True


if __name__ == "__main__":
    print("Model Manager - Cloud Deployment Utility")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            status = get_all_models_status()
            print(json.dumps(status, indent=2))
        
        elif command == "download-all":
            results = download_all_models()
            print("\nDownload Results:")
            print(json.dumps(results, indent=2))
        
        elif command == "check":
            success = initialize_models_for_deployment()
            sys.exit(0 if success else 1)
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: status, download-all, check")
    
    else:
        # Default: check status
        initialize_models_for_deployment()
