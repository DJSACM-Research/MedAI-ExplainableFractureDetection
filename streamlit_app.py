"""
Main entry point for Streamlit Cloud deployment.
Streamlit Cloud looks for streamlit_app.py or app.py in the root directory.

Uses the cloud-optimized version with Hugging Face Inference API.
For local development with Ollama, use: streamlit run apps/patient_chat_app_local.py
"""

import os
import sys
import streamlit as st

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Pre-initialize models check (runs once at app startup)
@st.cache_resource
def initialize_deployment():
    """Initialize deployment environment and models."""
    from src.utils.model_manager import initialize_models_for_deployment
    
    try:
        models_ready = initialize_models_for_deployment()
        return models_ready
    except Exception as e:
        st.error(f"Error checking models: {e}")
        return False

if __name__ == "__main__":
    # Check model availability
    # models_ready = initialize_deployment()
    
    # Import and run the cloud version with Hugging Face
    from apps.patient_chat_app_cloud import main
    main()

# import os
# import sys
# import streamlit as st

# # Add src directory to Python path
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# # Check if we're in deployment mode
# IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_DEPLOYMENT", "False").lower() == "true"

# # Pre-initialize models check (runs once at app startup)
# @st.cache_resource
# def initialize_deployment():
#     """Initialize deployment environment and models."""
#     from src.utils.model_manager import initialize_models_for_deployment
    
#     try:
#         models_ready = initialize_models_for_deployment()
#         return models_ready
#     except Exception as e:
#         st.error(f"Error checking models: {e}")
#         return False

# if __name__ == "__main__":
#     # Check model availability
#     # models_ready = initialize_deployment()
    
#     # Import and run the main app
#     from apps.patient_chat_app_local import main
#     main()
