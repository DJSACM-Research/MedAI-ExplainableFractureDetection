"""
MedAI Streamlit Application Entry Point

This is the main entry point for Streamlit Cloud deployment.
Streamlit Cloud looks for streamlit_app.py or app.py in the root directory.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    # Import and run the unified app
    from apps.app import main
    main()
else:
    # For streamlit run, execute main directly
    from apps.app import main
    main()
