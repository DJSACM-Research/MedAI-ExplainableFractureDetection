import streamlit as st
import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import tempfile
import cv2

# Add src to path so package imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Import project modules
from medai.agents.educational_agent import EducationalAgent
import visualize_xgradcam as vxc
from medai.agents.diagnostic_agent import get_model, get_transforms, DiagnosticAgent

# Page Check
st.set_page_config(page_title="MedAI Fracture Detection", layout="wide")

# --- CSS / Aesthetics ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
    .report-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
CHECKPOINT_PATH_SWIN = 'outputs/swin_mps/best.pth'
CHECKPOINT_PATH_DENSE = 'outputs/cross_validation/best_hypercolumn_cbam_densenet169.pth' # The requested model
CLASS_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Oblique', 'Oblique Displaced', 'Spiral', 'Transverse', 'Transverse Displaced']

@st.cache_resource
def load_fracture_model():
    """
    Loads the best available model. Priority: Hypercolumn DenseNet -> Swin.
    """
    device = vxc.detect_device()
    
    # Try loading the requested DenseNet first
    model_path = CHECKPOINT_PATH_DENSE
    model_name = 'densenet169'
    
    # Verify file existence and validity (basic check)
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        st.warning(f"⚠️ Requested model '{os.path.basename(model_path)}' not found or invalid. Falling back to Swin Transformer.")
        model_path = CHECKPOINT_PATH_SWIN
        model_name = 'swin'
    
    try:
        # Load architecture
        # We use the visualize_xgradcam get_model which is adapted for this repo
        model = vxc.get_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False)
        
        # Load weights
        ck = torch.load(model_path, map_location='cpu')
        state_dict = ck.get('model_state_dict', ck)
        
        # Determine strictness: if we are loading the "Hypercolumn" model into standard DenseNet, we might need strict=False
        strict_load = True
        if 'hypercolumn' in model_path:
            strict_load = False # Flexible loading for custom architecture
            
        model.load_state_dict(state_dict, strict=strict_load)
        model.to(device)
        model.eval()
        
        return model, device, model_name
        
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        # Final Fallback to Swin if DenseNet failed mid-load
        if model_name != 'swin' and os.path.exists(CHECKPOINT_PATH_SWIN):
             st.info("Attempting fallback to Swin Transformer...")
             try:
                model = vxc.get_model('swin', num_classes=len(CLASS_NAMES), pretrained=False)
                ck = torch.load(CHECKPOINT_PATH_SWIN, map_location='cpu')
                model.load_state_dict(ck['model_state_dict'])
                model.to(device)
                model.eval()
                return model, device, 'swin'
             except Exception as e2:
                 st.error(f"Fallback failed: {e2}")
                 return None, None, None
        return None, None, None

# --- Main App ---

st.title("🏥 MedAI Fracture Detection & Education System")
st.markdown("### Powered by Hypercolumn-CBAM DenseNet / MedGemma")

# Sidebar
st.sidebar.header("System Status")
model, device, loaded_model_name = load_fracture_model()
if model:
    st.sidebar.success(f"Model Loaded: {loaded_model_name.upper()}")
    st.sidebar.info(f"Device: {device}")
else:
    st.sidebar.error("Model Failed to Load")

@st.cache_resource
def get_educational_agent():
    return EducationalAgent(doctor_name="Dr. Automated")

# Educational Agent Init
edu_agent = get_educational_agent()

# Upload
uploaded_file = st.file_uploader("Upload X-Ray Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file and model:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    tfile.write(uploaded_file.getvalue())
    tfile_path = tfile.name
    tfile.close() # Close so other processes can read it

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)
    
    if st.button("Analyze Fracture"):
        with st.spinner("Running global diagnosis & generating XGrad-CAM heatmap..."):
            # 1. Prediction
            try:
                transform = get_transforms(224)
                pil_img = Image.open(tfile_path).convert('RGB')
                inp = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = model(inp)
                    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx])
                    
                pred_class = CLASS_NAMES[pred_idx]
                is_fracture = pred_class != "Healthy"
                
                # 2. XGrad-CAM (Disabled per user request to revert to pre-4b model state)
                # heatmap_img, cam_mask, _ = vxc.generate_xgradcam_overlay(...)
                
                # with col2:
                #     if heatmap_img:
                #         st.image(heatmap_img, caption=f"XGrad-CAM Attention (Pred: {pred_class})", use_container_width=True)
                #     else:
                #         st.warning("Could not generate heatmap.")
                        
                # 3. Diagnosis Display
                st.divider()
                st.markdown(f"## Diagnosis: **{pred_class}**")
                st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.text(traceback.format_exc())

        # 4. Educational Agent (MedGemma)
        if 'pred_class' in locals():
            st.divider()
            st.subheader("🤖 MedGemma Patient Education")
            
            with st.spinner("Consulting MedGemma Educational Agent..."):
                # Prepare data for agent
                diagnosis_data = {
                    "image_path": tfile_path,
                    "predicted_class": pred_class,
                    "confidence_score": confidence,
                    "fracture_detected": is_fracture
                }
                
                # Use a basic technical explanation since XGrad-CAM is disabled
                raw_text = f"The model detected features consistent with a {pred_class} fracture with {confidence:.2%} confidence."
                
                try:
                    # Run Edu Agent
                    report = edu_agent.translate_to_layman_terms(diagnosis_data, raw_text)
                    
                    # Display
                    st.info(f"**Patient Summary:**\n\n{report['patient_summary']}")
                    st.markdown(f"**Action Plan:**\n{report['next_steps_action_plan']}")
                    
                    with st.expander("Show Technical Details (Doctor View)"):
                        st.write(raw_text)
                        
                except Exception as e:
                    st.error(f"Educational Agent failed: {e}")

    # Cleanup
    # os.remove(tfile_path) # optional, keep for debugging
