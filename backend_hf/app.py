

import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import io
import timm
import requests
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try optional imports
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not installed. Explainability features will be disabled.")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed. Knowledge features will be disabled.")

app = FastAPI()

# ============================================================================
# CUSTOM MODEL ARCHITECTURES
# ============================================================================

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0.0):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        return out

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0.0):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'denselayer{i + 1}', layer)
    
    def forward(self, x):
        features = [x]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class HypercolumnCBAMDenseNet(nn.Module):
    def __init__(self, num_classes=8, growth_rate=32, bn_size=4, drop_rate=0.0):
        super(HypercolumnCBAMDenseNet, self).__init__()
        import torchvision.models as models
        densenet = models.densenet169(weights=None)
        self.features = densenet.features
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64)
        )
        self.db1 = self.features.denseblock1
        self.db2 = self.features.denseblock2
        self.db3 = self.features.denseblock3
        self.db4 = self.features.denseblock4
        self.t1 = self.features.transition1
        self.t2 = self.features.transition2
        self.t3 = self.features.transition3
        self.norm_final = self.features.norm5
        self.fusion_conv = nn.Conv2d(2688, 1024, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(1024)
        self.cbam = CBAM(1024)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.init_conv(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.db1(x)
        t1_out = self.t1(x)
        x = self.db2(t1_out)
        t2_out = self.t2(x)
        x = self.db3(t2_out)
        t3_out = self.t3(x)
        x = self.db4(t3_out)
        x_final = self.norm_final(x)
        target_size = x_final.shape[2:]
        t1_resized = nn.functional.interpolate(t1_out, size=target_size, mode='bilinear', align_corners=False)
        t2_resized = nn.functional.interpolate(t2_out, size=target_size, mode='bilinear', align_corners=False)
        t3_resized = nn.functional.interpolate(t3_out, size=target_size, mode='bilinear', align_corners=False)
        hypercolumn = torch.cat([x_final, t3_resized, t2_resized, t1_resized], dim=1)
        x = self.fusion_conv(hypercolumn)
        x = self.bn_fusion(x)
        x = nn.functional.relu(x)
        x = self.cbam(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================================
# CONSTANTS & CONFIG
# ============================================================================

CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique", 
    "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

MODEL_FILES = {
    "best_swin.pth": "swin",
    "best_densenet169.pth": "densenet169",
    "best_efficientnetv2.pth": "efficientnetv2",
    "best_mobilenetv2.pth": "mobilenetv2",
    "best_maxvit.pth": "maxvit",
    "best_hypercolumn_cbam_densenet169.pth": "hypercolumn_cbam_densenet169",
    "best_hypercolumn_cbam_densenet169_focal.pth": "hypercolumn_cbam_densenet169_focal",
    "best_hypercolumn_cbam_densenet169_old.pth": "hypercolumn_cbam_densenet169_old",
    "best_hypercolumn_densenet169.pth": "hypercolumn_densenet169",
    "best_hypercolumn_densenet169_old.pth": "hypercolumn_densenet169_old",
}

MODEL_CONFIGS = {
    "swin": "swin_small_patch4_window7_224",
    "densenet169": "densenet169",
    "efficientnetv2": "efficientnet_b0",
    "mobilenetv2": "mobilenetv2_100",
    "maxvit": "maxvit_tiny_tf_224",
    "hypercolumn_cbam_densenet169": "custom",
    "hypercolumn_cbam_densenet169_focal": "custom",
    "hypercolumn_cbam_densenet169_old": "custom",
    "hypercolumn_densenet169": "custom",
    "hypercolumn_densenet169_old": "custom",
}

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

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================================
# AGENTS
# ============================================================================

def get_transforms(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class ModelEnsembleAgent:
    """Runs inference across multiple models and combines predictions."""
    HYPERCOLUMN_PRIORITY_CLASSES = {"Oblique", "Oblique Displaced", "Transverse", "Transverse Displaced"}
    HYPERCOLUMN_WEIGHT = 3.0
    DEFAULT_WEIGHT = 1.0
    
    def __init__(self, models: Dict[str, nn.Module], class_names: List[str], device, img_size: int = 224):
        self.models = models
        self.class_names = class_names
        self.device = device
        self.transforms = get_transforms(img_size)
    
    def _is_hypercolumn_model(self, model_name: str) -> bool:
        return "hypercolumn" in model_name.lower() or "cbam" in model_name.lower()
    
    def _get_weighted_average(self, all_probs: List[np.ndarray], model_names: List[str], 
                               use_hypercolumn_priority: bool) -> np.ndarray:
        weights = []
        for name in model_names:
            if use_hypercolumn_priority and self._is_hypercolumn_model(name):
                weights.append(self.HYPERCOLUMN_WEIGHT)
            else:
                weights.append(self.DEFAULT_WEIGHT)
        weights = np.array(weights)
        weights = weights / weights.sum()
        weighted_probs = np.zeros_like(all_probs[0])
        for prob, weight in zip(all_probs, weights):
            weighted_probs += prob * weight
        return weighted_probs
    
    @torch.no_grad()
    def run_ensemble(self, image: Image.Image) -> Dict[str, Any]:
        if not self.models:
            return {"error": "No models loaded"}
        
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        all_probs = []
        model_names = []
        individual_predictions = {}
        
        for name, model in self.models.items():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            model_names.append(name)
            pred_idx = np.argmax(probs)
            individual_predictions[name] = {
                "class": self.class_names[pred_idx],
                "confidence": float(probs[pred_idx])
            }
        
        equal_avg_probs = np.mean(all_probs, axis=0)
        preliminary_idx = np.argmax(equal_avg_probs)
        preliminary_class = self.class_names[preliminary_idx]
        use_hypercolumn_priority = preliminary_class in self.HYPERCOLUMN_PRIORITY_CLASSES
        avg_probs = self._get_weighted_average(all_probs, model_names, use_hypercolumn_priority)
        ensemble_idx = np.argmax(avg_probs)
        ensemble_class = self.class_names[ensemble_idx]
        ensemble_confidence = float(avg_probs[ensemble_idx])
        
        return {
            "ensemble_prediction": ensemble_class,
            "ensemble_confidence": ensemble_confidence,
            "individual_predictions": individual_predictions,
            "fracture_detected": ensemble_class != "Healthy",
            "all_probabilities": {self.class_names[i]: float(avg_probs[i]) for i in range(len(avg_probs))},
        }

class ExplainabilityAgent:
    def __init__(self, model, class_names: List[str], device, body_part: str = "bone"):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.body_part = body_part
        self.transforms = get_transforms()
        self.target_layer = self._get_target_layer()
    
    def _get_target_layer(self):
        if self.model is None: return None
        for attr in ['layer4', 'features', 'stages', 'blocks']:
            if hasattr(self.model, attr):
                layer = getattr(self.model, attr)
                if isinstance(layer, nn.Sequential) and len(layer) > 0:
                    return [layer[-1]]
                return [layer]
        layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                layers.append(module)
        return [layers[-1]] if layers else None
    
    def generate_gradcam(self, image: Image.Image, target_class: int = None) -> Optional[np.ndarray]:
        if not GRADCAM_AVAILABLE or self.model is None or self.target_layer is None:
            return None
        try:
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            with GradCAM(model=self.model, target_layers=self.target_layer) as cam:
                targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                return grayscale_cam[0]
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return None
    
    def visualize_gradcam(self, image: Image.Image, cam_array: np.ndarray) -> Image.Image:
        if cam_array is None: return image
        img_array = np.array(image.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(img_array.astype(np.float32), cam_array, use_rgb=True)
        return Image.fromarray(visualization)

    def generate_explanation(self, prediction: str, confidence: float, cam_array: np.ndarray = None) -> str:
        if prediction == "Healthy":
            if confidence > 0.90:
                return f"The {self.body_part} appears **healthy** with high confidence ({confidence:.2f}). No fracture pattern was detected."
            else:
                return f"The {self.body_part} is likely **healthy** ({confidence:.2f}), though some areas warrant closer examination."
        
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
        
        conf_desc = "high" if confidence > 0.9 else ("moderate" if confidence > 0.7 else "low")
        return f"A fracture pattern consistent with **{prediction}** is detected with {conf_desc} confidence ({confidence:.2f}).{location_text}"

class EducationalAgent:
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
    
    def translate(self, prediction: str, confidence: float) -> Dict[str, str]:
        fracture_detected = prediction != "Healthy"
        severity_layman = self.severity_map.get(prediction, "Unknown")
        
        if not fracture_detected:
            summary = f"Great news! The AI analysis suggests your bone looks healthy. The system is {confidence*100:.0f}% confident."
            action_plan = "Recommended Actions:\n1. If pain persists, discuss with your doctor.\n2. No immediate treatment appears necessary."
        else:
            summary = f"The AI analysis has detected what appears to be a **{prediction}** fracture. This is classified as **{severity_layman}**."
            kb_info = MEDICAL_KNOWLEDGE_BASE.get(prediction, {})
            guidelines = kb_info.get("treatment_guidelines", ["Consult with an orthopedic specialist."])
            action_plan = "Recommended Actions:\n" + "\n".join([f"{i+1}. {g}" for i, g in enumerate(guidelines)])
        
        return {
            "patient_summary": summary,
            "severity_layman": severity_layman,
            "next_steps_action_plan": action_plan
        }

class KnowledgeAgent:
    def __init__(self):
        self.knowledge_base = MEDICAL_KNOWLEDGE_BASE
    
    def get_medical_summary(self, diagnosis: str, confidence: float) -> Dict[str, Any]:
        diagnosis = diagnosis.strip()
        raw = self.knowledge_base.get(diagnosis, {})
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
# API
# ============================================================================

# Global State
models = {}
device = torch.device("cpu")
ensemble_agent = None

def get_model(name: str, num_classes: int):
    # Check if custom hypercolumn model
    if "hypercolumn" in name.lower() or "cbam" in name.lower():
        return HypercolumnCBAMDenseNet(num_classes=num_classes)
    # Otherwise standard timm model
    model_name = MODEL_CONFIGS.get(name, name)
    try:
        model = timm.create_model(model_name, pretrained=False)
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            model.reset_classifier(num_classes=num_classes)
        return model
    except Exception as e:
        print(f"Error creating model {name}: {e}")
        return None

@app.on_event("startup")
def load_models_startup():
    global models, ensemble_agent
    models_dir = "./models"
    if not os.path.exists(models_dir):
        print("Models directory not found.")
        return

    for filename, config_name in MODEL_FILES.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            try:
                model = get_model(config_name, NUM_CLASSES)
                if model:
                    checkpoint = torch.load(path, map_location=device)
                    s_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                    model.load_state_dict(s_dict, strict=False)
                    model.to(device)
                    model.eval()
                    models[config_name] = model
                    print(f"Loaded {config_name}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    if models:
        ensemble_agent = ModelEnsembleAgent(models, CLASS_NAMES, device)

class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any]
    history: List[Dict[str, str]]
    user_data: Optional[Dict[str, str]] = None

@app.get("/")
def read_root():
    return {"status": "MedAI V2 Running", "models_loaded": list(models.keys())}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if not models or not ensemble_agent:
        return {"error": "Models not loaded"}
    
    try:
        # 1. Read Image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        
        # 2. Ensemble Inference
        ensemble_result = ensemble_agent.run_ensemble(image)
        prediction = ensemble_result['ensemble_prediction']
        confidence = ensemble_result['ensemble_confidence']
        
        # 3. Explainability (Grad-CAM)
        # Use first available model for visualization
        primary_model = next(iter(models.values()))
        explain_agent = ExplainabilityAgent(primary_model, CLASS_NAMES, device)
        
        pred_idx = CLASS_NAMES.index(prediction)
        cam_array = explain_agent.generate_gradcam(image, pred_idx)
        
        explanation_text = explain_agent.generate_explanation(prediction, confidence, cam_array)
        
        gradcam_b64 = None
        if cam_array is not None:
            viz_img = explain_agent.visualize_gradcam(image, cam_array)
            buf = io.BytesIO()
            viz_img.save(buf, format="PNG")
            gradcam_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # 4. Educational Content
        edu_agent = EducationalAgent()
        edu_result = edu_agent.translate(prediction, confidence)
        
        # 5. Knowledge Base
        know_agent = KnowledgeAgent()
        kb_result = know_agent.get_medical_summary(prediction, confidence)
        
        return {
            "prediction": ensemble_result,
            "explanation": {
                "text": explanation_text,
                "heatmap_b64": gradcam_b64
            },
            "educational": edu_result,
            "knowledge_base": kb_result
        }
        
    except Exception as e:
        logger.exception("Diagnosis failed")
        return {"error": str(e)}

@app.post("/chat")
async def chat(req: ChatRequest):
    """Simple wrapper for OpenRouter chat with retry logic and context."""
    import time
    import random

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
    
    model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")
    
    # Construct System Prompt based on context
    medical_context = req.context
    
    user_info = ""
    if req.user_data:
        user_info = f"""
    Patient Context:
    - Age: {req.user_data.get('age', 'Not specified')}
    - Gender: {req.user_data.get('gender', 'Not specified')}
    - Medical History: {req.user_data.get('history', 'None provided')}
        """

    system_prompt = f"""
    You are MedAI, a helpful medical assistant specializing in bone fractures.
    
    Current Diagnosis Context:
    - Diagnosis: {medical_context.get('Diagnosis', 'Unknown')}
    - Severity: {medical_context.get('Severity_Rating', 'Unknown')}
    - Definition: {medical_context.get('Type_Definition', '')}
    {user_info}
    
    Treatment Guidelines:
    {chr(10).join(['- '+g for g in medical_context.get('Treatment_Guidelines', [])])}
    
    Instructions:
    - Answer the patient's questions based on the diagnosis and their specific context (age, history).
    - Be empathetic and clear, but ALWAYS clarify you are an AI assistant, not a doctor.
    - If the user's medical history suggests complications (e.g., diabetes, osteoporosis), mention relevant precautions.
    """
    
    messages = [{"role": "system", "content": system_prompt}] + req.history + [{"role": "user", "content": req.message}]
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}", 
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://medai-app.com", # Required by OpenRouter
                    "X-Title": "MedAI Fracture Detection"
                },
                json={"model": model, "messages": messages},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            if 'choices' not in data or not data['choices']:
                 raise ValueError("Invalid API response: no choices found")
            return {"reply": data['choices'][0]['message']['content']}
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < max_retries:
                # Exponential backoff with jitter
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                logger.warning(f"Rate limited (429). Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
                continue
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Upstream API Error: {str(e)}")
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
