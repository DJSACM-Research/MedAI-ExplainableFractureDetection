"""
MedAI Streamlit Application

Unified web interface for fracture detection, diagnosis, and patient education.
Supports both local (Ollama) and cloud (Hugging Face) LLM backends.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import requests

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from new medai package
from medai.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    IMG_SIZE,
    MODELS_DIR,
    MEDICAL_KNOWLEDGE_BASE,
    OLLAMA_ENDPOINT,
    OLLAMA_MODEL,
    STREAMLIT_CONFIG,
)
from medai.agents.diagnostic import DiagnosticAgent
from medai.agents.ensemble import EnsembleAgent
from medai.agents.educational import EducationalAgent
from medai.agents.explainability import ExplainabilityAgent, generate_random_heatmap
from medai.agents.knowledge import KnowledgeAgent

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"],
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: bold;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LLM Backend
# ============================================================================

class PatientInteractionAgent:
    """Handles LLM-based patient Q&A using Ollama."""
    
    def __init__(self, medical_summary: Dict[str, Any], patient_history: Dict[str, Any]):
        self.medical_summary = medical_summary
        self.patient_history = patient_history
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Create RAG context for LLM."""
        guidelines = "\n- ".join(
            self.medical_summary.get("Guidelines", ["No specific guidelines available."])
        )
        
        return f"""
You are a compassionate medical assistant. Answer patient questions based ONLY on this diagnostic information.

RULES:
1. Use reassuring, non-technical language suitable for patients.
2. Keep answers concise and address the patient's concern.
3. ALWAYS advise consulting their orthopedic specialist.

--- DIAGNOSTIC INFORMATION ---
Diagnosis: {self.medical_summary.get('Diagnosis')} (Confidence: {self.medical_summary.get('Ensemble_Confidence')})
Definition: {self.medical_summary.get('Type')}
Severity: {self.medical_summary.get('Severity')}
Treatment Guidelines:
{guidelines}

--- PATIENT HISTORY ---
Age: {self.patient_history.get('age')}
Gender: {self.patient_history.get('gender')}
Medical History: {self.patient_history.get('history')}
"""
    
    def get_response(self, query: str) -> str:
        """Query Ollama LLM."""
        full_prompt = f"{self.system_prompt}\n\nPATIENT QUERY: {query}"
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        
        try:
            response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("response", "Error: No response from LLM.")
        except requests.exceptions.RequestException as e:
            return f"Error: Could not connect to Ollama. Make sure it's running. ({e})"


# ============================================================================
# Helper Functions
# ============================================================================

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to temp location."""
    if uploaded_file is None:
        return None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def run_ensemble_agent(image_path: str) -> Dict[str, Any]:
    """Run ensemble prediction."""
    try:
        agent = EnsembleAgent(
            model_names=["swin", "densenet169", "efficientnetv2", "maxvit", "mobilenetv2"],
            checkpoints_dir=str(MODELS_DIR),
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES,
        )
        return agent.run_ensemble(image_path)
    except Exception as e:
        return {"error": str(e)}


def run_complete_workflow(image_path: str) -> Dict[str, Any]:
    """Run the complete analysis workflow."""
    result = {
        "ensemble": None,
        "educational": None,
        "knowledge": None,
        "explanation": None,
    }
    
    try:
        # 1. Ensemble prediction
        ensemble_result = run_ensemble_agent(image_path)
        if "error" in ensemble_result:
            return {"error": ensemble_result["error"]}
        result["ensemble"] = ensemble_result
        
        # 2. Generate explanation
        explain_agent = ExplainabilityAgent(class_names=CLASS_NAMES)
        mapped_diag = {
            "predicted_class": ensemble_result["ensemble_prediction"],
            "confidence_score": ensemble_result["ensemble_confidence"],
            "fracture_detected": ensemble_result["fracture_detected"],
        }
        heatmap = generate_random_heatmap()
        explanation = explain_agent.generate_explanation(mapped_diag, heatmap)
        result["explanation"] = explanation
        
        # 3. Educational translation
        edu_agent = EducationalAgent()
        educational = edu_agent.translate_to_layman_terms(mapped_diag, explanation)
        result["educational"] = educational
        
        # 4. Medical knowledge
        knowledge_agent = KnowledgeAgent()
        knowledge = knowledge_agent.get_medical_summary(
            ensemble_result["ensemble_prediction"],
            ensemble_result["ensemble_confidence"],
        )
        result["knowledge"] = knowledge
        
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application."""
    st.title("🦴 MedAI Fracture Detection System")
    st.info(
        "⚠️ **Research/Educational Use Only** - "
        "This system is not approved for clinical use without professional oversight."
    )
    st.markdown("---")
    
    # Initialize session state
    if "patient_context" not in st.session_state:
        st.session_state.patient_context = {
            "age": 45,
            "gender": "Female",
            "history": "No major past issues.",
        }
    if "workflow_result" not in st.session_state:
        st.session_state.workflow_result = None
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 Diagnosis",
        "💬 Patient Chat",
        "📊 Details",
        "ℹ️ About",
    ])
    
    # ========================================================================
    # Tab 1: Diagnosis
    # ========================================================================
    with tab1:
        st.header("Upload X-ray for Analysis")
        
        image_file = st.file_uploader(
            "Upload X-ray image",
            type=["jpg", "png", "jpeg"],
            help="Upload a bone X-ray image for fracture detection.",
        )
        
        if image_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image_file, caption="Uploaded X-ray", use_container_width=True)
            
            with col2:
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Running AI analysis..."):
                        image_path = save_uploaded_file(image_file)
                        result = run_complete_workflow(image_path)
                        st.session_state.workflow_result = result
        
        # Display results
        if st.session_state.workflow_result:
            result = st.session_state.workflow_result
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("Analysis Complete!")
                
                # Main diagnosis
                ensemble = result.get("ensemble", {})
                st.subheader("Diagnosis Result")
                
                pred = ensemble.get("ensemble_prediction", "Unknown")
                conf = ensemble.get("ensemble_confidence", 0)
                fracture = ensemble.get("fracture_detected", False)
                
                if fracture:
                    st.warning(f"**Fracture Detected:** {pred}")
                else:
                    st.info(f"**Result:** {pred}")
                
                st.metric("Confidence", f"{conf:.1%}")
                
                # Educational summary
                educational = result.get("educational", {})
                if educational:
                    st.subheader("Patient Summary")
                    st.markdown(educational.get("patient_summary", ""))
                    
                    with st.expander("📋 Next Steps"):
                        st.markdown(educational.get("next_steps_action_plan", ""))
    
    # ========================================================================
    # Tab 2: Patient Chat
    # ========================================================================
    with tab2:
        st.header("Ask Questions About Your Diagnosis")
        
        if not st.session_state.workflow_result or "error" in st.session_state.workflow_result:
            st.warning("Please complete an analysis first in the Diagnosis tab.")
        else:
            result = st.session_state.workflow_result
            knowledge = result.get("knowledge", {})
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your diagnosis..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            agent = PatientInteractionAgent(
                                medical_summary=knowledge,
                                patient_history=st.session_state.patient_context,
                            )
                            response = agent.get_response(prompt)
                        except Exception as e:
                            response = f"Error: {e}. Make sure Ollama is running."
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # ========================================================================
    # Tab 3: Details
    # ========================================================================
    with tab3:
        st.header("Detailed Analysis")
        
        if st.session_state.workflow_result and "error" not in st.session_state.workflow_result:
            result = st.session_state.workflow_result
            
            with st.expander("🎯 Ensemble Predictions", expanded=True):
                ensemble = result.get("ensemble", {})
                if "individual_predictions" in ensemble:
                    for model, pred in ensemble["individual_predictions"].items():
                        st.write(f"**{model}:** {pred['class']} ({pred['confidence']:.2%})")
            
            with st.expander("🔬 Technical Explanation"):
                st.markdown(result.get("explanation", "No explanation available."))
            
            with st.expander("📚 Medical Knowledge"):
                knowledge = result.get("knowledge", {})
                st.json(knowledge)
        else:
            st.info("Complete an analysis to see detailed results.")
    
    # ========================================================================
    # Tab 4: About
    # ========================================================================
    with tab4:
        st.header("About MedAI")
        st.markdown("""
        ### MedAI Fracture Detection System
        
        This system uses a multi-model ensemble of deep learning models to detect
        and classify bone fractures from X-ray images.
        
        #### Models Used
        - **MaxViT** - Vision Transformer with multi-axis attention
        - **DenseNet-169** - Dense convolutional network
        - **HyperColumn-CBAM** - Custom architecture with attention
        - **EfficientNet-B0** - Efficient scaling network
        - **Swin Transformer** - Shifted window transformer
        - **MobileNetV2** - Efficient mobile architecture
        
        #### Fracture Types Detected
        - Comminuted (severe fragmentation)
        - Greenstick (partial break)
        - Oblique (diagonal break)
        - Oblique Displaced
        - Spiral (twisting break)
        - Transverse (straight break)
        - Transverse Displaced
        - Healthy (no fracture)
        
        #### Disclaimer
        This system is for research and educational purposes only.
        All diagnoses should be confirmed by qualified medical professionals.
        """)


if __name__ == "__main__":
    main()
