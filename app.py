"""
MedAI - Integrated Multi-Agent Fracture Detection System
=========================================================
A Streamlit application integrating all six agents:
1. DiagnosticAgent - Single model inference
2. ModelEnsembleAgent - Cross-validation ensemble
3. ExplainabilityAgent - Grad-CAM explanations
4. EducationalAgent - Patient-friendly translations
5. KnowledgeAgent - RAG-based knowledge retrieval
6. PatientInteractionAgent - LLM-powered chat
"""

import os
import io
import tempfile
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Attempt to import optional dependencies
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique", 
    "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

# Model configurations
MODEL_CONFIGS = {
    "swin": "swin_base_patch4_window7_224",
    "convnext": "convnext_base",
    "densenet169": "densenet169",
    "efficientnetv2": "tf_efficientnetv2_s",
    "mobilenetv2": "mobilenetv2_100",
    "maxvit": "maxvit_tiny_tf_224"
}

# Ollama configuration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# ChromaDB configuration
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================================
# MEDICAL KNOWLEDGE BASE
# ============================================================================

MEDICAL_KNOWLEDGE_BASE = {
    "Comminuted": {
        "definition": "A fracture where the bone is shattered into three or more fragments.",
        "icd_code": "S42.35",
        "severity": "Severe",
        "treatment_guidelines": [
            "Immediate orthopedic consultation required",
            "Surgical intervention often necessary (ORIF)",
            "Extended immobilization period (8-12 weeks)",
            "Physical therapy post-healing"
        ],
        "prognosis": "Recovery typically 3-6 months with proper surgical management."
    },
    "Greenstick": {
        "definition": "An incomplete fracture where the bone bends and cracks but does not break completely.",
        "icd_code": "S42.31",
        "severity": "Mild to Moderate",
        "treatment_guidelines": [
            "Often treated with casting or splinting",
            "Immobilization for 4-6 weeks",
            "Common in children due to bone flexibility",
            "Follow-up X-rays to monitor healing"
        ],
        "prognosis": "Excellent prognosis, typically heals within 4-8 weeks."
    },
    "Healthy": {
        "definition": "No fracture detected. Bone structure appears normal.",
        "icd_code": "Z03.89",
        "severity": "None",
        "treatment_guidelines": [
            "No treatment required for fracture",
            "Address any other symptoms if present",
            "Follow up if pain persists"
        ],
        "prognosis": "N/A - No fracture present."
    },
    "Oblique": {
        "definition": "A fracture with an angled break across the bone shaft.",
        "icd_code": "S42.33",
        "severity": "Moderate",
        "treatment_guidelines": [
            "May require reduction if displaced",
            "Casting for 6-8 weeks typical",
            "Monitor for displacement during healing",
            "Physical therapy may be beneficial"
        ],
        "prognosis": "Good prognosis with proper alignment, 6-10 weeks healing."
    },
    "Oblique Displaced": {
        "definition": "An angled fracture where bone fragments have shifted from normal alignment.",
        "icd_code": "S42.33",
        "severity": "Moderate to Severe",
        "treatment_guidelines": [
            "Closed or open reduction typically required",
            "May need internal fixation (pins, plates)",
            "Extended immobilization (8-12 weeks)",
            "Regular imaging to monitor alignment"
        ],
        "prognosis": "Good with surgical correction, 8-12 weeks healing."
    },
    "Spiral": {
        "definition": "A fracture caused by a twisting force, creating a helical break pattern.",
        "icd_code": "S42.34",
        "severity": "Moderate to Severe",
        "treatment_guidelines": [
            "Often requires surgical stabilization",
            "Evaluate for associated soft tissue injury",
            "Cast or brace after stabilization",
            "Rotational alignment must be maintained"
        ],
        "prognosis": "Good with proper stabilization, 8-12 weeks healing."
    },
    "Transverse": {
        "definition": "A horizontal fracture perpendicular to the long axis of the bone.",
        "icd_code": "S42.32",
        "severity": "Moderate",
        "treatment_guidelines": [
            "Often stable and amenable to casting",
            "Reduction if significantly displaced",
            "Immobilization for 6-8 weeks",
            "Monitor for angulation"
        ],
        "prognosis": "Good prognosis, typically 6-8 weeks healing."
    },
    "Transverse Displaced": {
        "definition": "A horizontal fracture with bone fragments out of alignment.",
        "icd_code": "S42.32",
        "severity": "Moderate to Severe",
        "treatment_guidelines": [
            "Reduction required (closed or open)",
            "Internal fixation often recommended",
            "Extended monitoring for healing",
            "Physical therapy post-healing"
        ],
        "prognosis": "Good with proper reduction, 8-10 weeks healing."
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device():
    """Detects and returns the appropriate torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transforms(img_size: int = 224):
    """Returns standard image transforms for inference."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_model(name: str, num_classes: int, pretrained: bool = False):
    """Loads a model architecture from timm."""
    if not TIMM_AVAILABLE:
        return None
    
    model_name = MODEL_CONFIGS.get(name, name)
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def load_model_from_checkpoint(model_name: str, checkpoint_path: str, num_classes: int, device):
    """Loads a model with weights from a checkpoint."""
    model = get_model(model_name, num_classes, pretrained=False)
    if model is None:
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load {model_name}: {e}")
        return None


# ============================================================================
# AGENT 1: DIAGNOSTIC AGENT
# ============================================================================

class DiagnosticAgent:
    """Runs inference on a single model to diagnose fractures."""
    
    def __init__(self, model, class_names: List[str], device, img_size: int = 224):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.transforms = get_transforms(img_size)
    
    @torch.no_grad()
    def diagnose(self, image: Image.Image) -> Dict[str, Any]:
        """Runs diagnosis on a PIL image."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        outputs = self.model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_class = self.class_names[pred_idx]
        
        return {
            "predicted_class": predicted_class,
            "confidence_score": confidence,
            "fracture_detected": predicted_class != "Healthy",
            "all_probabilities": {self.class_names[i]: float(probs[i]) for i in range(len(probs))},
            "severity_type": predicted_class
        }


# ============================================================================
# AGENT 2: MODEL ENSEMBLE AGENT (Cross-Validation)
# ============================================================================

class ModelEnsembleAgent:
    """Runs inference across multiple models and combines predictions."""
    
    def __init__(self, models: Dict[str, nn.Module], class_names: List[str], device, img_size: int = 224):
        self.models = models
        self.class_names = class_names
        self.device = device
        self.transforms = get_transforms(img_size)
    
    @torch.no_grad()
    def run_ensemble(self, image: Image.Image) -> Dict[str, Any]:
        """Runs ensemble inference on a PIL image."""
        if not self.models:
            return {"error": "No models loaded"}
        
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        all_probs = []
        individual_predictions = {}
        
        for name, model in self.models.items():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            
            pred_idx = np.argmax(probs)
            individual_predictions[name] = {
                "class": self.class_names[pred_idx],
                "confidence": float(probs[pred_idx])
            }
        
        # Soft voting (average probabilities)
        avg_probs = np.mean(all_probs, axis=0)
        ensemble_idx = np.argmax(avg_probs)
        ensemble_class = self.class_names[ensemble_idx]
        ensemble_confidence = float(avg_probs[ensemble_idx])
        
        return {
            "ensemble_prediction": ensemble_class,
            "ensemble_confidence": ensemble_confidence,
            "individual_predictions": individual_predictions,
            "fracture_detected": ensemble_class != "Healthy",
            "all_probabilities": {self.class_names[i]: float(avg_probs[i]) for i in range(len(avg_probs))}
        }


# ============================================================================
# AGENT 3: EXPLAINABILITY AGENT (Grad-CAM)
# ============================================================================

class ExplainabilityAgent:
    """Generates Grad-CAM visualizations and textual explanations."""
    
    def __init__(self, model, class_names: List[str], device, body_part: str = "bone"):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.body_part = body_part
        self.transforms = get_transforms()
        self.target_layer = self._get_target_layer()
    
    def _get_target_layer(self):
        """Gets the appropriate target layer for Grad-CAM."""
        if self.model is None:
            return None
        
        # Try common layer names
        for attr in ['layer4', 'features', 'stages', 'blocks']:
            if hasattr(self.model, attr):
                layer = getattr(self.model, attr)
                if isinstance(layer, nn.Sequential) and len(layer) > 0:
                    return [layer[-1]]
                return [layer]
        
        # Fallback: get last conv layer
        layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                layers.append(module)
        return [layers[-1]] if layers else None
    
    def generate_gradcam(self, image: Image.Image, target_class: int = None) -> Optional[np.ndarray]:
        """Generates Grad-CAM heatmap."""
        if not GRADCAM_AVAILABLE or self.model is None or self.target_layer is None:
            return None
        
        try:
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            with GradCAM(model=self.model, target_layers=self.target_layer) as cam:
                targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                return grayscale_cam[0]
        except Exception as e:
            st.warning(f"Grad-CAM generation failed: {e}")
            return None
    
    def visualize_gradcam(self, image: Image.Image, cam_array: np.ndarray) -> Image.Image:
        """Overlays Grad-CAM on the original image."""
        if cam_array is None:
            return image
        
        # Normalize image to 0-1
        img_array = np.array(image.resize((224, 224))) / 255.0
        
        # Create heatmap overlay
        visualization = show_cam_on_image(img_array.astype(np.float32), cam_array, use_rgb=True)
        return Image.fromarray(visualization)
    
    def generate_explanation(self, diagnosis_result: Dict[str, Any], cam_array: np.ndarray = None) -> str:
        """Generates textual explanation based on diagnosis and Grad-CAM."""
        predicted_class = diagnosis_result.get("predicted_class", diagnosis_result.get("ensemble_prediction", "Unknown"))
        confidence = diagnosis_result.get("confidence_score", diagnosis_result.get("ensemble_confidence", 0.0))
        
        if predicted_class == "Healthy":
            if confidence > 0.90:
                return f"The {self.body_part} appears **healthy** with high confidence ({confidence:.2f}). No fracture pattern was detected."
            else:
                return f"The {self.body_part} is likely **healthy** ({confidence:.2f}), though some areas warrant closer examination."
        
        # Analyze heatmap if available
        location_text = ""
        if cam_array is not None:
            norm_cam = cam_array / (cam_array.max() + 1e-8)
            y_indices, x_indices = np.where(norm_cam > 0.5)
            if len(y_indices) > 0 and len(x_indices) > 0:
                avg_x = np.mean(x_indices) / cam_array.shape[1]
                avg_y = np.mean(y_indices) / cam_array.shape[0]
                
                x_loc = "right side" if avg_x > 0.65 else ("left side" if avg_x < 0.35 else "center")
                y_loc = "distal end" if avg_y > 0.65 else ("proximal end" if avg_y < 0.35 else "middle region")
                location_text = f" The model's attention is focused on the **{y_loc}** of the **{x_loc}**."
        
        # Confidence description
        if confidence > 0.9:
            conf_desc = "high"
        elif confidence > 0.7:
            conf_desc = "moderate"
        else:
            conf_desc = "low"
        
        explanation = (
            f"A fracture pattern consistent with **{predicted_class}** is detected with {conf_desc} "
            f"confidence ({confidence:.2f}).{location_text}"
        )
        
        return explanation


# ============================================================================
# AGENT 4: EDUCATIONAL AGENT
# ============================================================================

class EducationalAgent:
    """Translates technical diagnoses into patient-friendly explanations."""
    
    def __init__(self, doctor_name: str = "Your Doctor"):
        self.doctor_name = doctor_name
        
        self.severity_map = {
            "Healthy": "None",
            "Greenstick": "Mild (The bone is cracked but not completely broken through.)",
            "Transverse": "Moderate (A straight break across the bone.)",
            "Oblique": "Moderate (An angled break across the bone.)",
            "Oblique Displaced": "Moderate-Severe (The bone pieces have shifted out of place.)",
            "Transverse Displaced": "Moderate-Severe (The bone pieces have shifted out of place.)",
            "Spiral": "Moderate-Severe (A twisting break that spirals around the bone.)",
            "Comminuted": "Severe (The bone has broken into multiple pieces.)"
        }
    
    def translate(self, diagnosis_result: Dict[str, Any], explanation_text: str) -> Dict[str, str]:
        """Generates patient-friendly summary and action plan."""
        fracture_detected = diagnosis_result.get("fracture_detected", False)
        predicted_class = diagnosis_result.get("predicted_class", diagnosis_result.get("ensemble_prediction", "Unknown"))
        confidence = diagnosis_result.get("confidence_score", diagnosis_result.get("ensemble_confidence", 0.0))
        
        severity_layman = self.severity_map.get(predicted_class, "Unknown")
        
        if not fracture_detected:
            summary = (
                f"Great news! The AI analysis suggests your bone looks healthy. "
                f"The system is {confidence*100:.0f}% confident in this assessment."
            )
            action_plan = (
                "📋 **Recommended Actions:**\n"
                "1. If you're still experiencing pain, please discuss with your doctor.\n"
                "2. This AI result should be confirmed by a medical professional.\n"
                "3. No immediate treatment appears necessary based on this analysis."
            )
        else:
            summary = (
                f"The AI analysis has detected what appears to be a **{predicted_class}** fracture. "
                f"This is classified as **{severity_layman}**. "
                f"The system is {confidence*100:.0f}% confident in this finding."
            )
            
            kb_info = MEDICAL_KNOWLEDGE_BASE.get(predicted_class, {})
            guidelines = kb_info.get("treatment_guidelines", ["Consult with an orthopedic specialist."])
            
            action_plan = (
                "📋 **Recommended Actions:**\n"
                + "\n".join([f"{i+1}. {g}" for i, g in enumerate(guidelines)])
                + f"\n\n⚠️ **Important:** This is an AI-assisted analysis. "
                f"Please consult with {self.doctor_name} for definitive diagnosis and treatment."
            )
        
        return {
            "patient_summary": summary,
            "severity_layman": severity_layman,
            "next_steps_action_plan": action_plan
        }


# ============================================================================
# AGENT 5: KNOWLEDGE AGENT
# ============================================================================

class KnowledgeAgent:
    """Provides structured medical knowledge and RAG capabilities."""
    
    def __init__(self):
        self.knowledge_base = MEDICAL_KNOWLEDGE_BASE
        self.chroma_client = None
        self.collection = None
        
        if CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL_NAME
                )
                self._setup_collection()
            except Exception as e:
                st.warning(f"ChromaDB initialization failed: {e}")
    
    def _setup_collection(self):
        """Sets up the ChromaDB collection."""
        if self.chroma_client is None:
            return
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="medical_diagnoses",
            embedding_function=self.embedding_fn
        )
        
        diagnoses = list(self.knowledge_base.keys())
        ids = [d.lower().replace(" ", "-") for d in diagnoses]
        
        if self.collection.count() != len(diagnoses):
            try:
                self.chroma_client.delete_collection("medical_diagnoses")
            except:
                pass
            self.collection = self.chroma_client.get_or_create_collection(
                name="medical_diagnoses",
                embedding_function=self.embedding_fn
            )
            self.collection.add(documents=diagnoses, ids=ids)
    
    def get_medical_summary(self, diagnosis: str, confidence: float) -> Dict[str, Any]:
        """Gets structured medical information for a diagnosis."""
        diagnosis = diagnosis.strip()
        
        # Try exact match first
        if diagnosis in self.knowledge_base:
            raw = self.knowledge_base[diagnosis]
        else:
            # Fall back to vector search if available
            if self.collection and self.collection.count() > 0:
                results = self.collection.query(query_texts=[diagnosis], n_results=1)
                if results and results["documents"] and results["documents"][0]:
                    retrieved = results["documents"][0][0]
                    raw = self.knowledge_base.get(retrieved, {})
                else:
                    raw = {}
            else:
                raw = {}
        
        if not raw:
            return {"error": f"No information found for '{diagnosis}'"}
        
        return {
            "Diagnosis": diagnosis,
            "Ensemble_Confidence": f"{confidence:.2f}",
            "Type_Definition": raw.get("definition", "N/A"),
            "ICD_Code": raw.get("icd_code", "N/A"),
            "Severity_Rating": raw.get("severity", "N/A"),
            "Treatment_Guidelines": raw.get("treatment_guidelines", []),
            "Long_Term_Prognosis": raw.get("prognosis", "N/A")
        }


# ============================================================================
# AGENT 6: PATIENT INTERACTION AGENT
# ============================================================================

class PatientInteractionAgent:
    """Handles patient chat using RAG and LLM."""
    
    def __init__(self, medical_summary: Dict[str, Any], patient_history: Dict[str, Any]):
        self.medical_summary = medical_summary
        self.patient_history = patient_history
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Creates the system prompt with medical context."""
        guidelines = "\n- ".join(self.medical_summary.get('Treatment_Guidelines', ["No specific guidelines available."]))
        
        return f"""
You are a knowledgeable and compassionate medical assistant specializing in fracture care. Your goal is to provide 
helpful, accurate information about fractures based on the context provided. 

IMPORTANT RULES:
1. ONLY use the information provided in the context below
2. Do NOT give specific medical advice or treatment plans
3. Always recommend consulting with a healthcare professional
4. Be empathetic and use clear, simple language
5. If unsure about something, acknowledge the limitation

--- DIAGNOSIS CONTEXT ---
Diagnosis: {self.medical_summary.get('Diagnosis')} (Confidence: {self.medical_summary.get('Ensemble_Confidence')})
ICD Code: {self.medical_summary.get('ICD_Code', 'N/A')}
Definition: {self.medical_summary.get('Type_Definition')}
Severity: {self.medical_summary.get('Severity_Rating')}
General Treatment Guidelines: 
- {guidelines}
Prognosis Note: {self.medical_summary.get('Long_Term_Prognosis', 'N/A')}

--- PATIENT INFORMATION ---
Age: {self.patient_history.get('age', 'Unknown')}
Gender: {self.patient_history.get('gender', 'Unknown')}
Medical History: {self.patient_history.get('history', 'None provided')}
"""
    
    def get_response(self, query: str) -> str:
        """Gets LLM response for a patient query."""
        if not REQUESTS_AVAILABLE:
            return "Chat functionality requires the requests library."
        
        full_prompt = f"{self.system_prompt}\n\nPATIENT QUERY: {query}"
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        try:
            response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "Could not get response from LLM.")
        except requests.exceptions.ConnectionError:
            return "⚠️ Cannot connect to Ollama. Please ensure Ollama is running with `ollama serve` and the llama3 model is pulled."
        except Exception as e:
            return f"⚠️ Error: {e}"


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def initialize_session_state():
    """Initializes session state variables."""
    defaults = {
        "diagnosis_result": None,
        "ensemble_result": None,
        "gradcam_image": None,
        "explanation_text": None,
        "educational_output": None,
        "medical_summary": None,
        "chat_messages": [],
        "patient_agent": None,
        "models_loaded": False,
        "uploaded_image": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_models(checkpoint_dir: str, selected_models: List[str], device):
    """Loads selected models from checkpoint directory."""
    models = {}
    
    for model_name in selected_models:
        checkpoint_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
        
        if os.path.exists(checkpoint_path):
            model = load_model_from_checkpoint(model_name, checkpoint_path, NUM_CLASSES, device)
            if model is not None:
                models[model_name] = model
    
    return models


def render_sidebar():
    """Renders the sidebar configuration."""
    st.sidebar.title("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    checkpoint_dir = st.sidebar.text_input(
        "Checkpoint Directory",
        value="./models",
        help="Directory containing model checkpoint files"
    )
    
    available_models = list(MODEL_CONFIGS.keys())
    selected_models = st.sidebar.multiselect(
        "Models to Load",
        options=available_models,
        default=["swin"] if not st.session_state.models_loaded else available_models[:1],
        help="Select models for ensemble inference"
    )
    
    # Patient info
    st.sidebar.subheader("Patient Information")
    patient_age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
    patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    patient_history = st.sidebar.text_area(
        "Medical History",
        value="No significant medical history.",
        height=100
    )
    
    return {
        "checkpoint_dir": checkpoint_dir,
        "selected_models": selected_models,
        "patient_info": {
            "age": patient_age,
            "gender": patient_gender,
            "history": patient_history
        }
    }


def render_image_upload():
    """Renders the image upload section."""
    st.subheader("📤 Upload X-Ray Image")
    
    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a bone X-ray image for analysis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
        
        return image
    
    return None


def render_diagnosis_results():
    """Renders the diagnosis results section."""
    if st.session_state.diagnosis_result is None and st.session_state.ensemble_result is None:
        return
    
    st.subheader("🔬 Diagnosis Results")
    
    col1, col2 = st.columns(2)
    
    # Single model result
    with col1:
        st.markdown("**Primary Model Diagnosis**")
        if st.session_state.diagnosis_result:
            result = st.session_state.diagnosis_result
            if "error" not in result:
                status = "🔴 Fracture Detected" if result["fracture_detected"] else "🟢 No Fracture"
                st.metric("Status", status)
                st.metric("Classification", result["predicted_class"])
                st.metric("Confidence", f"{result['confidence_score']:.2%}")
            else:
                st.error(result["error"])
    
    # Ensemble result
    with col2:
        st.markdown("**Ensemble Prediction**")
        if st.session_state.ensemble_result:
            result = st.session_state.ensemble_result
            if "error" not in result:
                status = "🔴 Fracture Detected" if result["fracture_detected"] else "🟢 No Fracture"
                st.metric("Status", status)
                st.metric("Classification", result["ensemble_prediction"])
                st.metric("Confidence", f"{result['ensemble_confidence']:.2%}")
                
                # Show individual predictions
                with st.expander("Individual Model Predictions"):
                    for name, pred in result["individual_predictions"].items():
                        st.write(f"**{name}**: {pred['class']} ({pred['confidence']:.2%})")
            else:
                st.error(result["error"])
    
    # Probability distribution
    if st.session_state.ensemble_result and "all_probabilities" in st.session_state.ensemble_result:
        st.markdown("**Class Probabilities**")
        probs = st.session_state.ensemble_result["all_probabilities"]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        classes = list(probs.keys())
        values = list(probs.values())
        colors = ['#2ecc71' if c == 'Healthy' else '#e74c3c' for c in classes]
        
        bars = ax.barh(classes, values, color=colors)
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2%}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def render_explainability():
    """Renders the explainability section."""
    if st.session_state.gradcam_image is None and st.session_state.explanation_text is None:
        return
    
    st.subheader("🔍 AI Explanation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.gradcam_image:
            st.image(st.session_state.gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)
        else:
            st.info("Grad-CAM visualization not available.")
    
    with col2:
        if st.session_state.explanation_text:
            st.markdown("**Model Explanation:**")
            st.markdown(st.session_state.explanation_text)


def render_educational_output():
    """Renders the educational/patient-friendly section."""
    if st.session_state.educational_output is None:
        return
    
    st.subheader("📚 Patient Information")
    
    output = st.session_state.educational_output
    
    st.info(output["patient_summary"])
    
    st.markdown(f"**Severity Level:** {output['severity_layman']}")
    
    st.markdown(output["next_steps_action_plan"])


def render_knowledge_base():
    """Renders the knowledge base section."""
    if st.session_state.medical_summary is None:
        return
    
    st.subheader("📖 Medical Knowledge Base")
    
    summary = st.session_state.medical_summary
    
    if "error" in summary:
        st.error(summary["error"])
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Diagnosis:** {summary.get('Diagnosis', 'N/A')}")
        st.markdown(f"**ICD Code:** {summary.get('ICD_Code', 'N/A')}")
        st.markdown(f"**Severity:** {summary.get('Severity_Rating', 'N/A')}")
    
    with col2:
        st.markdown(f"**Definition:** {summary.get('Type_Definition', 'N/A')}")
        st.markdown(f"**Prognosis:** {summary.get('Long_Term_Prognosis', 'N/A')}")
    
    with st.expander("Treatment Guidelines"):
        for guideline in summary.get("Treatment_Guidelines", []):
            st.markdown(f"• {guideline}")


def render_chat_interface():
    """Renders the patient chat interface."""
    st.subheader("💬 Ask Questions")
    
    if st.session_state.patient_agent is None:
        st.info("Complete the analysis above to enable the chat feature.")
        return
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your diagnosis, treatment, or recovery..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.patient_agent.get_response(prompt)
                st.markdown(response)
        
        st.session_state.chat_messages.append({"role": "assistant", "content": response})


def run_analysis(image: Image.Image, config: dict, device):
    """Runs the full analysis pipeline."""
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models(config["checkpoint_dir"], config["selected_models"], device)
    
    if not models:
        st.error("No models could be loaded. Please check your checkpoint directory.")
        return
    
    # Get primary model for single diagnosis
    primary_model_name = list(models.keys())[0]
    primary_model = models[primary_model_name]
    
    # Agent 1: Diagnostic Agent
    with st.spinner("Running primary diagnosis..."):
        diagnostic_agent = DiagnosticAgent(primary_model, CLASS_NAMES, device)
        st.session_state.diagnosis_result = diagnostic_agent.diagnose(image)
    
    # Agent 2: Ensemble Agent
    if len(models) > 1:
        with st.spinner("Running ensemble analysis..."):
            ensemble_agent = ModelEnsembleAgent(models, CLASS_NAMES, device)
            st.session_state.ensemble_result = ensemble_agent.run_ensemble(image)
    else:
        # Use single model result as ensemble result
        st.session_state.ensemble_result = {
            "ensemble_prediction": st.session_state.diagnosis_result["predicted_class"],
            "ensemble_confidence": st.session_state.diagnosis_result["confidence_score"],
            "individual_predictions": {primary_model_name: {
                "class": st.session_state.diagnosis_result["predicted_class"],
                "confidence": st.session_state.diagnosis_result["confidence_score"]
            }},
            "fracture_detected": st.session_state.diagnosis_result["fracture_detected"],
            "all_probabilities": st.session_state.diagnosis_result["all_probabilities"]
        }
    
    # Agent 3: Explainability Agent
    with st.spinner("Generating explanation..."):
        explain_agent = ExplainabilityAgent(primary_model, CLASS_NAMES, device, body_part="bone")
        
        # Get predicted class index
        pred_class = st.session_state.ensemble_result["ensemble_prediction"]
        pred_idx = CLASS_NAMES.index(pred_class) if pred_class in CLASS_NAMES else None
        
        cam_array = explain_agent.generate_gradcam(image, pred_idx)
        
        if cam_array is not None:
            st.session_state.gradcam_image = explain_agent.visualize_gradcam(image, cam_array)
        
        st.session_state.explanation_text = explain_agent.generate_explanation(
            st.session_state.ensemble_result, cam_array
        )
    
    # Agent 4: Educational Agent
    with st.spinner("Preparing patient information..."):
        edu_agent = EducationalAgent(doctor_name="your healthcare provider")
        st.session_state.educational_output = edu_agent.translate(
            st.session_state.ensemble_result,
            st.session_state.explanation_text or ""
        )
    
    # Agent 5: Knowledge Agent
    with st.spinner("Retrieving medical knowledge..."):
        knowledge_agent = KnowledgeAgent()
        st.session_state.medical_summary = knowledge_agent.get_medical_summary(
            st.session_state.ensemble_result["ensemble_prediction"],
            st.session_state.ensemble_result["ensemble_confidence"]
        )
    
    # Agent 6: Patient Interaction Agent
    if "error" not in st.session_state.medical_summary:
        st.session_state.patient_agent = PatientInteractionAgent(
            st.session_state.medical_summary,
            config["patient_info"]
        )
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": f"Hello! I've analyzed your X-ray and found: **{st.session_state.ensemble_result['ensemble_prediction']}** "
                       f"(Confidence: {st.session_state.ensemble_result['ensemble_confidence']:.1%}). "
                       f"How can I help answer your questions about this diagnosis?"
        }]


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="MedAI - Fracture Detection System",
        page_icon="🦴",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🦴 MedAI - Multi-Agent Fracture Detection System")
    st.markdown(
        "An AI-powered system for detecting and explaining bone fractures using "
        "ensemble deep learning and explainable AI techniques."
    )
    
    # Check dependencies
    missing_deps = []
    if not TIMM_AVAILABLE:
        missing_deps.append("timm")
    if not GRADCAM_AVAILABLE:
        missing_deps.append("pytorch-grad-cam")
    if not CHROMADB_AVAILABLE:
        missing_deps.append("chromadb")
    
    if missing_deps:
        st.warning(f"Some features may be limited. Missing optional dependencies: {', '.join(missing_deps)}")
    
    # Device info
    device = get_device()
    st.sidebar.info(f"🖥️ Device: {device}")
    
    # Sidebar configuration
    config = render_sidebar()
    
    st.markdown("---")
    
    # Main content
    col_upload, col_results = st.columns([1, 2])
    
    with col_upload:
        image = render_image_upload()
        
        if image is not None:
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                run_analysis(image, config, device)
                st.rerun()
    
    with col_results:
        render_diagnosis_results()
    
    st.markdown("---")
    
    # Explainability and Education
    col_explain, col_edu = st.columns(2)
    
    with col_explain:
        render_explainability()
    
    with col_edu:
        render_educational_output()
    
    st.markdown("---")
    
    # Knowledge Base
    render_knowledge_base()
    
    st.markdown("---")
    
    # Chat Interface
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This is an AI-assisted tool for educational purposes only. "
        "It is not intended to replace professional medical advice, diagnosis, or treatment. "
        "Always consult with a qualified healthcare provider for medical decisions."
    )


if __name__ == "__main__":
    main()
