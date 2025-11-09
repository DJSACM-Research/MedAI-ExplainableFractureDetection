"""
Streamlit-based Patient Chat Application for Fracture Detection and Diagnosis.
CLOUD VERSION - Uses Hugging Face Inference API instead of Ollama

Supports:
1. Running individual agents (Diagnostic, Educational, Explainability, Knowledge)
2. Running the complete workflow
3. LLM-based Q&A via Hugging Face Inference API for patient education
"""

import os
import sys
import streamlit as st
import requests
import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import the Agents ---
from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.educational_agent import EducationalAgent
from src.agents.explain_agent import ExplainabilityAgent, generate_random_heatmap, calculate_heatmap_centroid
from src.agents.knowledge_agent import KnowledgeAgent, MEDICAL_KNOWLEDGE_BASE
from src.agents.cross_validation_agent import ModelEnsembleAgent
from src.utils import get_device

# --- Hugging Face Inference API Configuration ---
HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# --- Constants ---
CLASS_NAMES = ["Comminuted", "Greenstick", "Healthy", "Oblique", 
               "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

# --- Page Configuration ---
st.set_page_config(
    page_title="🦴 Fracture Detection AI System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better UI ---
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
# --- 1. Hugging Face Inference API Patient Interaction Agent ---
# ============================================================================

class PatientInteractionAgent:
    """
    Handles communication with Mistral 7B model via Hugging Face Inference API.
    Free tier available with rate limiting.
    """
    def __init__(self, medical_summary: Dict[str, Any], patient_history: Dict[str, Any]):
        """Initialize the agent with medical context."""
        # --- Configuration Check ---
        if not HF_API_KEY:
            raise ValueError(
                "⚠️ HUGGINGFACE_API_KEY not found in Streamlit Secrets. "
                "Please set your Hugging Face API token in Settings > Secrets."
            )

        self.medical_summary = medical_summary
        self.patient_history = patient_history
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Creates a detailed instruction set for the LLM (RAG Context)."""
        guidelines = "\n- ".join(self.medical_summary.get('Guidelines', ["No specific guidelines available."]))

        return f"""You are a highly compassionate, clear, and professional medical assistant. Your goal is to answer patient questions in natural language based ONLY on the following diagnostic information and patient history.

RULES:
1. Maintain a reassuring, non-technical, and empathetic tone suitable for a patient.
2. Keep answers concise and address the patient's underlying concern.
3. ALWAYS conclude your answer by advising the patient to consult their orthopedic specialist or doctor.

--- DIAGNOSTIC INFORMATION ---
Diagnosis: {self.medical_summary.get('Diagnosis')} (Confidence: {self.medical_summary.get('Ensemble_Confidence')})
Definition: {self.medical_summary.get('Type')}
Severity: {self.medical_summary.get('Severity')}
Treatment Guidelines:
{guidelines}

--- PATIENT HISTORY ---
Age: {self.patient_history.get('age')}
Gender: {self.patient_history.get('gender')}
Medical History: {self.patient_history.get('history')}"""

    def get_response(self, query: str) -> str:
        """Queries the Hugging Face Inference API with the patient's question."""
        try:
            # Format prompt for Mistral using [INST] tags
            full_prompt = f"{self.system_prompt}\n\nPATIENT QUERY: {query}"
            
            payload = {
                "inputs": f"[INST] {full_prompt} [/INST]",
                "parameters": {
                    "max_new_tokens": 512,
                    "return_full_text": False,
                    "temperature": 0.7,
                }
            }

            response = requests.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()

            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "Error: Unexpected API response format.")
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            elif "error" in result:
                # Handle API errors (e.g., model loading, rate limiting)
                error_msg = result.get("error", "Unknown error")
                if "rate_limit" in str(error_msg).lower():
                    return "⚠️ API rate limit reached. Please wait a moment and try again."
                return f"⚠️ API Error: {error_msg}"
            else:
                return "Error: Unknown API response format."

        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. The model may be loading. Please try again."
        except requests.exceptions.RequestException as e:
            return f"❌ Network error: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"


# ============================================================================
# --- 2. Helper Functions ---
# ============================================================================

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to a temporary location."""
    try:
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        tmp_file = temp_dir / uploaded_file.name
        with open(tmp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


# ============================================================================
# --- 3. Workflow Functions ---
# ============================================================================

def run_diagnostic_agent(image_path: str) -> Dict[str, Any]:
    """Run the diagnostic agent on an image."""
    try:
        # Placeholder checkpoint path - in production, use actual model checkpoint
        checkpoint_path = "./outputs/best_swin.pth"
        
        if not os.path.exists(checkpoint_path):
            return {"error": f"Checkpoint not found at {checkpoint_path}"}
        
        agent = DiagnosticAgent(
            checkpoint_path=checkpoint_path,
            model_name='swin',
            num_classes=NUM_CLASSES,
            img_size=IMG_SIZE,
            class_names=CLASS_NAMES
        )
        
        result = agent.run_diagnosis(image_path)
        return result
    except Exception as e:
        return {"error": str(e)}


def run_ensemble_agent(image_path: str) -> Dict[str, Any]:
    """Run the ensemble agent on an image."""
    try:
        checkpoints_dir = "./outputs"
        
        if not os.path.exists(checkpoints_dir):
            return {"error": f"Checkpoints directory not found at {checkpoints_dir}"}
        
        agent = ModelEnsembleAgent(
            model_names=['swin', 'mobilenetv2', 'densenet169', 'efficientnetv2', 'maxvit'],
            checkpoints_dir=checkpoints_dir,
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES
        )
        
        result = agent.run_ensemble(image_path)
        return result
    except Exception as e:
        return {"error": str(e)}


def run_educational_agent(diagnosis_result: Dict[str, Any], explanation_text: str = "") -> Dict[str, str]:
    """Run the educational agent to translate diagnosis."""
    try:
        agent = EducationalAgent(doctor_name="your treating doctor")
        
        # Map ensemble result format to educational agent format
        # Ensemble uses: ensemble_prediction, ensemble_confidence
        # EducationalAgent expects: predicted_class, confidence_score
        mapped_result = {
            "predicted_class": diagnosis_result.get("ensemble_prediction", "Unknown"),
            "confidence_score": diagnosis_result.get("ensemble_confidence", 0.0),
            "fracture_detected": diagnosis_result.get("fracture_detected", True)
        }
        
        result = agent.translate_to_layman_terms(mapped_result, explanation_text)
        return result
    except Exception as e:
        return {"error": str(e)}


def run_explainability_agent(diagnosis_result: Dict[str, Any]) -> str:
    """Run the explainability agent to generate explanations."""
    try:
        agent = ExplainabilityAgent(class_names=CLASS_NAMES, body_part="bone")
        
        # Map ensemble result format to explainability agent format
        # Ensemble uses: ensemble_prediction, ensemble_confidence
        # ExplainabilityAgent expects: predicted_class, confidence_score
        mapped_result = {
            "predicted_class": diagnosis_result.get("ensemble_prediction", "Unknown"),
            "confidence_score": diagnosis_result.get("ensemble_confidence", 0.0),
            "fracture_detected": diagnosis_result.get("fracture_detected", True)
        }
        
        # Generate a random heatmap for demonstration
        heatmap = generate_random_heatmap()
        
        # Call with correct parameters
        explanation = agent.generate_explanation(
            diagnosis_result=mapped_result,
            cam_array=heatmap
        )
        return explanation
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


def run_knowledge_agent(diagnosis: str, confidence: float) -> Dict[str, Any]:
    """Run the knowledge agent to retrieve medical information."""
    try:
        agent = KnowledgeAgent(knowledge_base=MEDICAL_KNOWLEDGE_BASE)
        result = agent.get_medical_summary(diagnosis, confidence)
        return result
    except Exception as e:
        return {"error": str(e)}


def run_complete_workflow(image_path: str) -> Dict[str, Any]:
    """Run the complete workflow: Ensemble -> Education -> Knowledge."""
    workflow_result = {
        "ensemble_result": None,
        "educational_result": None,
        "knowledge_result": None,
        "explanation_result": None
    }
    
    try:
        # 1. Run Ensemble Agent
        ensemble_result = run_ensemble_agent(image_path)
        if "error" in ensemble_result:
            return {"error": f"Ensemble failed: {ensemble_result['error']}"}
        
        workflow_result["ensemble_result"] = ensemble_result
        
        # 2. Run Educational Agent
        educational_result = run_educational_agent(ensemble_result)
        workflow_result["educational_result"] = educational_result
        
        # 3. Run Explainability Agent
        explanation = run_explainability_agent(ensemble_result)
        workflow_result["explanation_result"] = explanation
        
        # 4. Run Knowledge Agent
        diagnosis = ensemble_result.get("ensemble_prediction", "Unknown")
        confidence = ensemble_result.get("ensemble_confidence", 0.0)
        knowledge_result = run_knowledge_agent(diagnosis, confidence)
        workflow_result["knowledge_result"] = knowledge_result
        
        return workflow_result
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# --- 4. Streamlit UI ---
# ============================================================================

def main():
    """Main Streamlit application."""
    st.title("🦴 AI Medical Assistant for Fracture Detection & Diagnosis")
    st.info("⚠️ **Research/Educational Use Only** - This system is not approved for clinical use without professional oversight.")
    st.markdown("---")
    
    # Initialize session state
    if "patient_context" not in st.session_state:
        st.session_state.patient_context = {
            "age": 45,
            "gender": "Female",
            "history": "No major past issues, but has mild osteoporosis."
        }
    
    # Initialize workflow results storage
    if "workflow_result" not in st.session_state:
        st.session_state.workflow_result = None
    
    # --- Create Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🏥 Single Agents", "⚙️ Complete Workflow", "💬 Patient Chat", "📋 Workflow Details", "ℹ️ About"]
    )
    
    # ========================================================================
    # --- TAB 1: Individual Agents ---
    # ========================================================================
    with tab1:
        st.header("Run Individual Agents")
        st.markdown("Test each agent independently with sample diagnosis data.")
        
        agent_choice = st.selectbox(
            "Select an Agent",
            ["Diagnostic Agent", "Ensemble Agent", "Educational Agent", "Explainability Agent", "Knowledge Agent"]
        )
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if agent_choice == "Diagnostic Agent":
                st.subheader("🔍 Diagnostic Agent")
                st.write("Runs a single model on an X-ray image to detect fractures.")
                
                image_file = st.file_uploader("Upload X-ray image", type=["jpg", "png", "jpeg"])
                if image_file and st.button("Run Diagnostic Agent"):
                    st.info("Note: Running this requires a valid model checkpoint at ./outputs/best_swin.pth")
                    with st.spinner("Running diagnostic agent..."):
                        image_path = save_uploaded_file(image_file)
                        result = run_diagnostic_agent(image_path)
                        st.json(result)
            
            elif agent_choice == "Ensemble Agent":
                st.subheader("🎯 Ensemble Agent (5 Models)")
                st.write("Combines predictions from multiple models for robust diagnosis.")
                
                image_file = st.file_uploader("Upload X-ray image", type=["jpg", "png", "jpeg"])
                if image_file and st.button("Run Ensemble Agent"):
                    st.info("Note: Running this requires model checkpoints in ./outputs/")
                    with st.spinner("Running ensemble agent..."):
                        image_path = save_uploaded_file(image_file)
                        result = run_ensemble_agent(image_path)
                        st.json(result)
            
            elif agent_choice == "Educational Agent":
                st.subheader("📚 Educational Agent")
                st.write("Translates technical diagnosis into patient-friendly language.")
                
                # Sample diagnosis for demo
                sample_diagnosis = {
                    "fracture_detected": True,
                    "predicted_class": "Transverse",
                    "confidence_score": 0.85,
                    "severity_type": "Transverse"
                }
                
                sample_explanation = "The bone shows a clear transverse break pattern."
                
                if st.button("Run Educational Agent (Demo)"):
                    with st.spinner("Translating diagnosis..."):
                        result = run_educational_agent(sample_diagnosis, sample_explanation)
                        if isinstance(result, dict):
                            for key, value in result.items():
                                st.write(f"**{key}:**\n{value}")
                        else:
                            st.error(result)
            
            elif agent_choice == "Explainability Agent":
                st.subheader("🎨 Explainability Agent")
                st.write("Generates human-readable explanations of model predictions.")
                
                sample_diagnosis = {
                    "predicted_class": "Greenstick",
                    "confidence_score": 0.92,
                    "fracture_detected": True
                }
                
                if st.button("Run Explainability Agent (Demo)"):
                    with st.spinner("Generating explanation..."):
                        explanation = run_explainability_agent(sample_diagnosis)
                        st.write(explanation)
            
            elif agent_choice == "Knowledge Agent":
                st.subheader("🧠 Knowledge Agent")
                st.write("Retrieves medical knowledge and guidelines for a diagnosis.")
                
                diagnosis_input = st.selectbox("Select Diagnosis", CLASS_NAMES)
                confidence_input = st.slider("Confidence Score", 0.0, 1.0, 0.85)
                
                if st.button("Run Knowledge Agent"):
                    with st.spinner("Retrieving medical knowledge..."):
                        result = run_knowledge_agent(diagnosis_input, confidence_input)
                        if isinstance(result, dict):
                            st.json(result)
                        else:
                            st.error(result)
    
    # ========================================================================
    # --- TAB 2: Complete Workflow ---
    # ========================================================================
    with tab2:
        st.header("Complete Diagnosis Workflow")
        st.markdown("Upload an X-ray image and run the complete diagnostic pipeline.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📤 Upload X-ray Image")
            image_file = st.file_uploader("Upload X-ray image for full diagnosis", type=["jpg", "png", "jpeg"])
            
            if image_file:
                st.image(image_file, caption="Uploaded Image", width='stretch')
        
        with col2:
            st.subheader("👤 Patient Information")
            age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.patient_context["age"])
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                 index=0 if st.session_state.patient_context["gender"] == "Male" else 
                                       1 if st.session_state.patient_context["gender"] == "Female" else 2)
            history = st.text_area("Medical History", value=st.session_state.patient_context["history"])
            
            st.session_state.patient_context = {"age": age, "gender": gender, "history": history}
        
        if image_file and st.button("🚀 Run Complete Workflow", key="workflow"):
            st.info("Note: Running this requires all model checkpoints in ./outputs/")
            with st.spinner("Running complete diagnostic workflow..."):
                image_path = save_uploaded_file(image_file)
                workflow_result = run_complete_workflow(image_path)
                
                # Store workflow result in session state for use in other tabs
                st.session_state.workflow_result = workflow_result
                
                if "error" in workflow_result:
                    st.error(f"❌ Error: {workflow_result['error']}")
                else:
                    # Display results
                    st.success("✅ Workflow completed successfully!")
                    
                    # Ensemble Results
                    if workflow_result["ensemble_result"]:
                        st.subheader("1️⃣ Ensemble Agent Results")
                        ensemble = workflow_result["ensemble_result"]
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Prediction", ensemble.get("ensemble_prediction", "N/A"))
                        col2.metric("Confidence", f"{ensemble.get('ensemble_confidence', 0):.2%}")
                        col3.metric("Fracture Detected", "Yes" if ensemble.get("fracture_detected") else "No")
                    
                    # Educational Results
                    if workflow_result["educational_result"]:
                        st.subheader("2️⃣ Patient-Friendly Summary")
                        educational = workflow_result["educational_result"]
                        for key, value in educational.items():
                            st.write(f"**{key}:**\n{value}")
                    
                    # Explainability Results
                    if workflow_result["explanation_result"]:
                        st.subheader("3️⃣ Technical Explanation")
                        st.write(workflow_result["explanation_result"])
                    
                    # Knowledge Results
                    if workflow_result["knowledge_result"]:
                        st.subheader("4️⃣ Medical Knowledge Base")
                        st.json(workflow_result["knowledge_result"])
    
    # ========================================================================
    # --- TAB 3: Patient Chat (Hugging Face) ---
    # ========================================================================
    with tab3:
        st.header("💬 Patient Q&A with AI Assistant")
        st.markdown("Ask questions about your fracture diagnosis using Hugging Face Inference API")
        
        # Check if workflow has been run
        if st.session_state.workflow_result is None or "error" in st.session_state.workflow_result:
            st.info("ℹ️ Please run the 'Complete Workflow' first to generate a diagnosis for the chat feature.")
        else:
            # Check HF API configuration
            if not HF_API_KEY:
                st.error(
                    "❌ Hugging Face API key not configured. "
                    "Please add your HUGGINGFACE_API_KEY to Streamlit Secrets."
                )
                st.markdown("""
                ### How to set up Hugging Face API:
                1. Get your API key from https://huggingface.co/settings/tokens
                2. In Streamlit Cloud, go to Settings > Secrets
                3. Add: `HUGGINGFACE_API_KEY = "hf_your_token_here"`
                4. Refresh the app
                """)
            else:
                # Build medical summary from workflow results
                ensemble_result = st.session_state.workflow_result.get("ensemble_result", {})
                knowledge_result = st.session_state.workflow_result.get("knowledge_result", {})
                
                diagnosis = ensemble_result.get("ensemble_prediction", "Unknown")
                confidence = ensemble_result.get("ensemble_confidence", 0.0)
                
                # Create medical summary from knowledge base
                medical_summary = {
                    "Diagnosis": diagnosis,
                    "Ensemble_Confidence": f"{confidence:.2f}",
                    "Type": knowledge_result.get("Type", "Unknown fracture type"),
                    "Severity": knowledge_result.get("Severity", "Unknown"),
                    "Guidelines": knowledge_result.get("Guidelines", [])
                }
                
                try:
                    agent = PatientInteractionAgent(medical_summary, st.session_state.patient_context)
                    
                    # Initialize chat history with diagnosis info
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Hello! I'm your AI medical assistant. I've reviewed your diagnosis: **{medical_summary['Diagnosis']}** (Confidence: {medical_summary['Ensemble_Confidence']}). How can I help answer your questions?"
                        })
                    
                    # Display chat messages
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    
                    # Accept user input
                    if prompt := st.chat_input("Ask a question about your diagnosis..."):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        
                        with st.chat_message("assistant"):
                            with st.spinner("🤖 Consulting Mistral 7B via Hugging Face..."):
                                response = agent.get_response(prompt)
                                st.markdown(response)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"❌ Error initializing chat agent: {str(e)}")
    
    # ========================================================================
    # --- TAB 4: Workflow Details ---
    # ========================================================================
    with tab4:
        st.header("📋 Workflow Execution Details")
        
        if st.session_state.workflow_result is None:
            st.info("ℹ️ No workflow results available. Please run a workflow first.")
        else:
            if "error" in st.session_state.workflow_result:
                st.error(f"Workflow Error: {st.session_state.workflow_result['error']}")
            else:
                st.success("Workflow executed successfully!")
                st.json(st.session_state.workflow_result)
    
    # ========================================================================
    # --- TAB 5: About ---
    # ========================================================================
    with tab5:
        st.header("ℹ️ About This Application")
        st.markdown("""
        ### 🦴 AI-Powered Fracture Detection System
        
        This application uses advanced deep learning models to detect and classify fractures from X-ray images.
        
        **Features:**
        - **Multi-Model Ensemble:** Combines 5 different architectures (Swin, MobileNetV2, DenseNet, EfficientNet, MaxViT)
        - **Explainability:** Generates human-readable explanations for predictions
        - **Patient Education:** Translates medical terminology into patient-friendly language
        - **AI Chatbot:** Ask questions about your diagnosis powered by Mistral 7B via Hugging Face
        
        **Models Used:**
        - Swin Transformer
        - MobileNetV2
        - DenseNet169
        - EfficientNetV2
        - MaxViT
        
        **Fracture Types Detected:**
        """)
        
        for i, fracture_type in enumerate(CLASS_NAMES, 1):
            st.write(f"{i}. {fracture_type}")
        
        st.markdown("""
        ### ⚠️ Important Disclaimer
        This system is for **research and educational purposes only**. 
        It is **NOT approved for clinical use** without professional medical oversight.
        Always consult with a qualified healthcare professional for medical diagnosis.
        
        ### 🔧 Technology Stack
        - **Frontend:** Streamlit
        - **ML Models:** PyTorch
        - **AI Assistant:** Hugging Face Inference API (Mistral 7B)
        - **Deployment:** Streamlit Cloud
        
        ### 📞 Contact & Support
        For issues or questions, please contact the development team.
        """)


if __name__ == "__main__":
    main()
