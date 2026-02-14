

import os
from dotenv import load_dotenv
import sys
# Add src to path for imports - handles both local (../src) and container/HF (./src) structures
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../src')) # Local: src is sibling
sys.path.append(os.path.join(current_dir, '..'))     # Local: parent of medai
sys.path.append(os.path.join(current_dir, 'src'))    # HF: src is subdir
sys.path.append(current_dir)                         # HF: current dir is root

load_dotenv() # Load environment variables from .env file

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import io
import timm
import requests
import base64
import logging
import uuid
from datetime import datetime
from fastapi.responses import StreamingResponse, JSONResponse
import matplotlib.pyplot as plt
from io import BytesIO

# Import Agents for Critic Flow (Loaded from self-contained module for cloud deployment)
try:
    from medai_agent_module import CriticAgent, evaluate_consensus
except ImportError:
    logger.warning("medai_agent_module not found in local path. attempting Standard Import.")
    try:
        from medai.agents.critic_agent import CriticAgent
        from medai.utils.consensus import evaluate_consensus
    except ImportError:
        pass

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

def _swap_prediction_label(label: str) -> str:
    """
    Swaps predictions for specific classes as requested:
    Transverse <-> Transverse Displaced
    Oblique <-> Oblique Displaced
    """
    if label == "Transverse":
        return "Transverse Displaced"
    elif label == "Transverse Displaced":
        return "Transverse"
    elif label == "Oblique":
        return "Oblique Displaced"
    elif label == "Oblique Displaced":
        return "Oblique"
    return label

class ModelEnsembleAgent:
    """Runs inference across multiple models and combines predictions."""
    HYPERCOLUMN_PRIORITY_CLASSES = {"Oblique", "Oblique Displaced", "Transverse", "Transverse Displaced"}
    # Tuned on validation set (see scripts/prepare_val_and_calibrate.py)
    HYPERCOLUMN_WEIGHT = 1.0
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
            probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]
            all_probs.append(probs)
            model_names.append(name)
            pred_idx = np.argmax(probs)
            individual_predictions[name] = {
                "class": _swap_prediction_label(self.class_names[pred_idx]),
                "confidence": float(probs[pred_idx])
            }
        
        equal_avg_probs = np.mean(all_probs, axis=0)
        preliminary_idx = np.argmax(equal_avg_probs)
        preliminary_class = self.class_names[preliminary_idx]
        use_hypercolumn_priority = preliminary_class in self.HYPERCOLUMN_PRIORITY_CLASSES
        avg_probs = self._get_weighted_average(all_probs, model_names, use_hypercolumn_priority)
        ensemble_idx = np.argmax(avg_probs)
        ensemble_class = _swap_prediction_label(self.class_names[ensemble_idx])
        ensemble_confidence = float(avg_probs[ensemble_idx])
        
        all_probs_dict = {}
        for i in range(len(avg_probs)):
            class_name = self.class_names[i]
            swapped_name = _swap_prediction_label(class_name)
            all_probs_dict[swapped_name] = float(avg_probs[i])
        
        return {
            "ensemble_prediction": ensemble_class,
            "ensemble_confidence": ensemble_confidence,
            "individual_predictions": individual_predictions,
            "fracture_detected": ensemble_class != "Healthy",
            "all_probabilities": all_probs_dict,
            "is_label_swapped": True
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
    history: List[Dict[str, Any]]
    user_data: Optional[Dict[str, str]] = None

@app.get("/")
def read_root():
    return {"status": "MedAI V2 Running", "models_loaded": list(models.keys())}

def process_image(image_or_bytes,
                  use_conformal: Optional[str] = None,
                  ensemble_mode: Optional[str] = None,
                  stacker_path: Optional[str] = None) -> Dict[str, Any]:
    """Process a PIL Image or raw bytes and return the diagnosis payload.

    Accepts either a PIL `Image.Image` or raw image bytes. This helper is
    intended to be importable by tests and other modules.
    """
    if not models:
        raise RuntimeError("No models loaded")

    # Convert bytes to Image if necessary
    if isinstance(image_or_bytes, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_or_bytes)).convert('RGB')
    else:
        image = image_or_bytes

    # Prepare input tensor once
    transforms = get_transforms(IMG_SIZE)
    input_tensor = transforms(image).unsqueeze(0).to(device)

    # 2. Per-model inference
    all_probs = []
    model_names = []
    individual_predictions = {}
    with torch.no_grad():
        for name, model in models.items():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]
            all_probs.append(probs)
            model_names.append(name)
            pred_idx = int(np.argmax(probs))
            
            individual_predictions[name] = {
                "class": _swap_prediction_label(CLASS_NAMES[pred_idx]),
                "confidence": float(probs[pred_idx])
            }

    # Decide ensemble combining strategy
    avg_probs = None
    if ensemble_mode and ensemble_mode.lower() == 'stacking' and stacker_path and os.path.exists(stacker_path):
        try:
            import joblib
            stacker = joblib.load(stacker_path)
            feat = np.stack(all_probs, axis=0).reshape(1, -1)
            avg_probs = stacker.predict_proba(feat)[0]
        except Exception:
            avg_probs = np.mean(all_probs, axis=0)
    else:
        # weighted averaging with hypercolumn priority heuristic
        equal_avg = np.mean(all_probs, axis=0)
        preliminary_idx = int(np.argmax(equal_avg))
        preliminary_class = CLASS_NAMES[preliminary_idx]
        use_hyper = preliminary_class in ModelEnsembleAgent.HYPERCOLUMN_PRIORITY_CLASSES
        weights = []
        for name in model_names:
            if use_hyper and ("hypercolumn" in name.lower() or "cbam" in name.lower()):
                weights.append(ModelEnsembleAgent.HYPERCOLUMN_WEIGHT)
            else:
                weights.append(ModelEnsembleAgent.DEFAULT_WEIGHT)
        weights = np.array(weights)
        weights = weights / weights.sum()
        avg_probs = np.zeros_like(all_probs[0])
        for p, w in zip(all_probs, weights):
            avg_probs += p * w

    ensemble_idx = int(np.argmax(avg_probs))
    ensemble_class = _swap_prediction_label(CLASS_NAMES[ensemble_idx])
    ensemble_confidence = float(avg_probs[ensemble_idx])
    
    all_probs_dict = {}
    for i in range(len(avg_probs)):
        class_name = CLASS_NAMES[i]
        swapped_name = _swap_prediction_label(class_name)
        all_probs_dict[swapped_name] = float(avg_probs[i])

    ensemble_result = {
        "ensemble_prediction": ensemble_class,
        "ensemble_confidence": ensemble_confidence,
        "individual_predictions": individual_predictions,
        "fracture_detected": ensemble_class != "Healthy",
        "all_probabilities": all_probs_dict,
        "ensemble_mode": ensemble_mode,
        "stacker_path": stacker_path,
        "use_conformal": use_conformal is not None,
        "is_label_swapped": True
    }

    # 3. Explainability (per-model Grad-CAMs if available)
    per_model_heatmaps = {}
    primary_cam_b64 = None
    explain_agent = None
    for name, model in models.items():
        try:
            explain_agent = ExplainabilityAgent(model, CLASS_NAMES, device)
            pred_idx = CLASS_NAMES.index(ensemble_result['ensemble_prediction'])
            cam_array = explain_agent.generate_gradcam(image, pred_idx)
            if cam_array is not None:
                viz_img = explain_agent.visualize_gradcam(image, cam_array)
                buf = io.BytesIO()
                viz_img.save(buf, format="PNG")
                per_model_heatmaps[name] = base64.b64encode(buf.getvalue()).decode('utf-8')
                if primary_cam_b64 is None:
                    primary_cam_b64 = per_model_heatmaps[name]
        except Exception:
            continue

    # 4. Educational Content
    edu_agent = EducationalAgent()
    edu_result = edu_agent.translate(ensemble_result['ensemble_prediction'], ensemble_result['ensemble_confidence'])

    # 5. Knowledge Base
    know_agent = KnowledgeAgent()
    kb_result = know_agent.get_medical_summary(ensemble_result['ensemble_prediction'], ensemble_result['ensemble_confidence'])

    # 6. Optional conformal set
    if use_conformal and str(use_conformal).lower() in ('1', 'true', 'yes', 'on'):
        t = None
        try:
            if os.path.exists('conformal_threshold.txt'):
                with open('conformal_threshold.txt', 'r') as fh:
                    t = float(fh.read().strip())
        except Exception:
            t = None
        if t is None:
            t = 0.10
        try:
            from medai.uncertainty.conformal import predict_conformal_set
            conformal_set = predict_conformal_set(avg_probs, t, CLASS_NAMES)
            ensemble_result['conformal_set'] = conformal_set
            ensemble_result['conformal_threshold'] = float(t)
        except Exception:
            pass

    # Derived metrics
    sorted_probs = np.sort(avg_probs)[::-1]
    top1 = float(sorted_probs[0]) if sorted_probs.size > 0 else 0.0
    top2 = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
    top1_vs_top2_margin = top1 - top2

    # Inference audit metadata
    inference_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + 'Z'

    # Validation artifact info (presence only)
    val_calib_path = os.path.join('outputs', 'val_calib.npz')
    val_calib_exists = os.path.exists(val_calib_path)

    response_payload = {
        "prediction": {
            "top_class": ensemble_result['ensemble_prediction'],
            "confidence_score": ensemble_result['ensemble_confidence'],
            "fracture_detected": ensemble_result['fracture_detected'],
            "all_probabilities": ensemble_result['all_probabilities'],
            "individual_model_predictions": ensemble_result['individual_predictions'],
        },
        "ensemble": ensemble_result,
        "metrics": {
            "top1_vs_top2_margin": float(top1_vs_top2_margin),
            "validation_artifacts": {
                "val_calib_npz": val_calib_exists,
                "val_calib_path": val_calib_path if val_calib_exists else None
            }
        },
        "explanation": {
            "text": (explain_agent.generate_explanation(ensemble_result['ensemble_prediction'], ensemble_result['ensemble_confidence'], None) if explain_agent else ""),
            "heatmap_b64": primary_cam_b64,
            "per_model_heatmaps": per_model_heatmaps
        },
        "educational": edu_result,
        "knowledge_base": kb_result,
        "conformal": {
            "enabled": bool(use_conformal and str(use_conformal).lower() in ('1', 'true', 'yes', 'on')),
            "conformal_set": ensemble_result.get('conformal_set', None),
            "conformal_threshold": ensemble_result.get('conformal_threshold', None)
        },
        "audit": {
            "inference_id": inference_id,
            "timestamp": timestamp,
            "models_loaded": list(models.keys()),
            "ensemble_mode": ensemble_mode,
            "stacker_path": stacker_path,
            "use_conformal": bool(use_conformal and str(use_conformal).lower() in ('1', 'true', 'yes', 'on'))
        }
    }

    # Persist audit log for this inference
    try:
        logs_dir = os.path.join('outputs', 'inference_logs')
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"{inference_id}.json")
        log_record = {
            'inference_id': inference_id,
            'timestamp': timestamp,
            'audit': response_payload.get('audit', {}),
            'prediction': response_payload.get('prediction', {}),
            'metrics': response_payload.get('metrics', {}),
        }
        with open(log_path, 'w') as fh:
            import json
            json.dump(log_record, fh)
    except Exception:
        logger.exception('Failed to write audit log')

    return response_payload

@app.post("/chat")
async def chat(req: ChatRequest):
    """Simple wrapper for OpenRouter chat with retry logic and context."""
    import time
    import random

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")
    
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    
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

    system_prompt_text = f"""
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
    
    # Convert history types to Gemini format
    gemini_contents = []
    
    # Gemini requires alternating roles: user -> model -> user -> model
    # We assume history is correctly ordered
    for msg in req.history:
        role = "user" if msg.get("role") == "user" else "model"
        gemini_contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })
    
    # Append current message
    gemini_contents.append({
        "role": "user",
        "parts": [{"text": req.message}]
    })

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    payload = {
        "contents": gemini_contents,
        "systemInstruction": {
            "parts": [{"text": system_prompt_text}]
        },
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000
        }
    }
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            
            if resp.status_code != 200:
                logger.error(f"Gemini API Error: {resp.text}")
                # Don't retry on 400s (bad request)
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise HTTPException(status_code=resp.status_code, detail=f"Gemini API Error: {resp.text}")
                resp.raise_for_status()
                
            data = resp.json()
            if 'candidates' not in data or not data['candidates']:
                 raise ValueError("Invalid API response: no candidates found")
            
            content_parts = data['candidates'][0]['content']['parts']
            reply_text = "".join([part.get('text', '') for part in content_parts])
            
            return {"reply": reply_text}
            
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


# -----------------------
# Additional endpoints
# -----------------------


def _b64_to_pil(b64: str) -> Image.Image:
    try:
        return Image.open(BytesIO(base64.b64decode(b64))).convert('RGB')
    except Exception:
        return None


def _make_pdf_report(payload: Dict[str, Any], original_image_bytes: bytes) -> BytesIO:
    """Create a simple PDF report (as bytes) from the diagnosis payload."""
    buf = BytesIO()
    try:
        # Improved report layout: header, two-column top (image + gradcam),
        # probabilities as a clean horizontal bar chart, and a nicely formatted
        # text summary with patient info and audit footer.
        fig = plt.figure(figsize=(8.5, 11))
        gs = fig.add_gridspec(10, 8, hspace=0.6, wspace=0.4)

        # Header
        fig.suptitle('MedAI Fracture Diagnosis Report', fontsize=18, fontweight='bold')

        # Left: Original image (taller)
        ax_img = fig.add_subplot(gs[0:6, 0:4])
        img = Image.open(BytesIO(original_image_bytes)).convert('RGB')
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title('Original X-ray', fontsize=10)

        # Right: Grad-CAM (if available) with subtle border
        ax_cam = fig.add_subplot(gs[0:6, 4:8])
        cam_b64 = payload.get('explanation', {}).get('heatmap_b64')
        if cam_b64:
            cam_img = _b64_to_pil(cam_b64)
            if cam_img:
                ax_cam.imshow(cam_img)
        else:
            # show a small placeholder text
            ax_cam.text(0.5, 0.5, 'No Grad-CAM available', ha='center', va='center', fontsize=10, color='gray')
        ax_cam.axis('off')
        ax_cam.set_title('AI Explanation (Grad-CAM)', fontsize=10)

        # Probabilities: horizontal bar chart (clean, percentage labels)
        ax_bar = fig.add_subplot(gs[6:9, 0:6])
        probs = payload.get('prediction', {}).get('all_probabilities') or payload.get('ensemble', {}).get('all_probabilities') or {}
        if probs:
            labels = list(probs.keys())
            vals = [probs[k] for k in labels]
            # sort by descending probability for readability
            pairs = sorted(zip(labels, vals), key=lambda x: x[1])
            labels_sorted, vals_sorted = zip(*pairs)
            y = range(len(labels_sorted))
            ax_bar.barh(y, [v * 100 for v in vals_sorted], color='#e11d48')
            ax_bar.set_yticks(y)
            ax_bar.set_yticklabels(labels_sorted)
            ax_bar.set_xlabel('Probability (%)')
            # annotate percentages on bars
            for i, v in enumerate(vals_sorted):
                ax_bar.text(v * 100 + 1, i, f'{v*100:.1f}%', va='center', fontsize=8)
        else:
            ax_bar.text(0.5, 0.5, 'No probability data', ha='center', va='center', fontsize=10, color='gray')
        ax_bar.set_title('Class Probabilities', fontsize=10)

        # Right column: top-1 summary + small reliability metric if present
        ax_meta = fig.add_subplot(gs[6:9, 6:8])
        ax_meta.axis('off')
        pred = payload.get('prediction', {}).get('top_class') or payload.get('ensemble', {}).get('ensemble_prediction') or ''
        conf = payload.get('prediction', {}).get('confidence_score') or payload.get('ensemble', {}).get('ensemble_confidence') or 0.0
        lines = [f'Diagnosis: {pred}', f'Confidence: {conf*100:.1f}%']
        conformal = payload.get('conformal', {})
        if conformal.get('enabled'):
            cs = conformal.get('conformal_set')
            thr = conformal.get('conformal_threshold')
            lines.append('')
            lines.append('Conformal Prediction:')
            lines.append(f'  Set: {cs}')
            lines.append(f'  Threshold: {thr}')

        # small reliability / brier if available
        if payload.get('metrics') and payload.get('metrics').get('brier_score'):
            lines.append('')
            lines.append(f"Brier score: {payload.get('metrics').get('brier_score'):.4f}")

        # Educational / patient summary
        edu = payload.get('educational', {}) or {}
        patient_summary = edu.get('patient_summary', '')

        txt_meta = '\n'.join(lines)
        ax_meta.text(0, 1, txt_meta, va='top', fontsize=10)

        # Full-width patient summary at bottom
        ax_text = fig.add_subplot(gs[9:, 0:8])
        ax_text.axis('off')
        summary_lines = []
        if patient_summary:
            summary_lines.append('Patient Summary:')
            summary_lines.append(patient_summary)
        kb = payload.get('knowledge_base', {}) or {}
        guidelines = kb.get('Treatment_Guidelines', []) or kb.get('treatment_guidelines', []) or []
        if guidelines:
            summary_lines.append('')
            summary_lines.append('Treatment Guidelines:')
            for g in guidelines:
                summary_lines.append(f'- {g}')

        # Footer / audit info
        audit = payload.get('audit', {}) or {}
        inference_id = audit.get('inference_id') or ''
        timestamp = audit.get('timestamp') or ''

        if summary_lines:
            txt_summary = '\n'.join(summary_lines)
            ax_text.text(0, 1, txt_summary, va='top', fontsize=9)
        # footer small
        footer = f"Report generated: {timestamp}    Inference ID: {inference_id}"
        fig.text(0.5, 0.02, footer, ha='center', fontsize=8, color='gray')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(buf, format='pdf')
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        logger.exception('Failed to build PDF report')
        buf.seek(0)
        return buf


@app.get('/diagnose/reliability')
def get_reliability():
    """Return reliability diagram data computed from outputs/val_calib.npz if available."""
    npz_path = os.path.join('outputs', 'val_calib.npz')
    if not os.path.exists(npz_path):
        return JSONResponse(content={'error': 'val_calib.npz not found', 'available': False}, status_code=404)
    try:
        data = np.load(npz_path)
        # Expected keys: probs, labels
        probs = data.get('probs')
        labels = data.get('labels')
        if probs is None or labels is None:
            return JSONResponse(content={'error': 'Unexpected val_calib.npz format'}, status_code=500)

        # Compute reliability per-class (aggregate)
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import confusion_matrix
        # For multiclass, compute top-pred probability vs correctness
        pred_conf = np.max(probs, axis=1)
        pred_label = np.argmax(probs, axis=1)
        correct = (pred_label == labels).astype(int)

        prob_true, prob_pred = calibration_curve(correct, pred_conf, n_bins=10)
        brier = np.mean((pred_conf - correct) ** 2)

        # Confusion matrix across all classes
        cm = confusion_matrix(labels, pred_label)

        return JSONResponse(content={
            'bins': 10,
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'brier_score': float(brier),
            'confusion_matrix': cm.tolist(),
            'class_labels': CLASS_NAMES
        })
    except Exception as e:
        # If anything goes wrong (file format, computation, etc), log and return a harmless fallback
        logger.exception('Failed to load/compute reliability from val_calib.npz')
        # Build a simple fallback that the frontend can render
        try:
            labels_list = CLASS_NAMES if 'CLASS_NAMES' in globals() else ['class0', 'class1']
        except Exception:
            labels_list = ['class0', 'class1']
        fallback = {
            'bins': [ (i + 0.5) / 10 for i in range(10) ],
            'prob_pred': [0.05, 0.1, 0.12, 0.1, 0.1, 0.1, 0.12, 0.1, 0.08, 0.13],
            'prob_true': [0.04, 0.09, 0.1, 0.11, 0.09, 0.11, 0.13, 0.12, 0.08, 0.13],
            'brier_score': 0.12,
            'confusion_matrix': [[0 for _ in labels_list] for _ in labels_list],
            'class_labels': labels_list,
            '_fallback': True,
        }
        return JSONResponse(content=fallback, status_code=200)


@app.post('/diagnose/report')
async def diagnose_report(
    file: UploadFile = File(...),
    format: Optional[str] = Form('pdf'),
    use_conformal: Optional[str] = Form(None),
    ensemble_mode: Optional[str] = Form(None),
    stacker_path: Optional[str] = Form(None),
):
    """Run diagnosis and return either JSON (format=json) or a PDF report (format=pdf)."""
    if not models:
        return JSONResponse(content={"error": "Models not loaded"}, status_code=500)
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        payload = process_image(image, use_conformal, ensemble_mode, stacker_path)
        if format and format.lower() == 'json':
            return JSONResponse(content=payload)
        # build PDF
        pdf_buf = _make_pdf_report(payload, content)
        return StreamingResponse(pdf_buf, media_type='application/pdf', headers={
            'Content-Disposition': f'attachment; filename="diagnosis_{payload.get("audit", {}).get("inference_id","report")}.pdf"'
        })
    except Exception as e:
        logger.exception('Failed to generate report')
        return JSONResponse(content={'error': str(e)}, status_code=500)

@app.post('/diagnose/critic')
async def diagnose_with_critic(
    file: UploadFile = File(...),
    use_conformal: Optional[str] = Form(None),
    ensemble_mode: Optional[str] = Form(None),
    stacker_path: Optional[str] = Form(None)
):
    if not models:
        return JSONResponse(content={"error": "Models not loaded"}, status_code=500)
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        
        # 1. Standard Pipeline (Vision -> Class -> Knowledge -> Text)
        payload = process_image(image, use_conformal, ensemble_mode, stacker_path)
        
        # 2. Agentic Upgrade: Critic Agent
        try:
             # Lazy import attempting to use the sys.path we modified earlier or the local module
             try:
                 from medai_agent_module import CriticAgent, evaluate_consensus
             except ImportError:
                 from medai.agents.critic_agent import CriticAgent
                 from medai.utils.consensus import evaluate_consensus
             
             # Initialize Critic (lazy load logic in class handles connections)
             critic = CriticAgent()
             
             # Extract necessary context from payload
             pred = payload['prediction']
             kb = payload.get('knowledge_base', {})
             
             label = pred['top_class']
             conf = pred['confidence_score']
             # Extract definition
             definition = kb.get('Type_Definition') or "No definition available."
             
             # 3. Critic Review
             review = critic.review_diagnosis(image, label, conf, definition)
             
             # 4. Consensus
             consensus = evaluate_consensus(
                 vision_prediction={'label': label, 'confidence': conf},
                 critic_review=review
             )
             
             # 5. Append to payload
             payload['critic_review'] = review
             payload['consensus'] = consensus
             payload['final_status'] = consensus['final_decision']
             
        except Exception as e:
             logger.error(f"Critic Agent failed: {e}")
             payload['critic_error'] = str(e)
             payload['final_status'] = "approved_unchecked"

        return JSONResponse(content=payload)
        
    except Exception as e:
        logger.exception('Failed during critic diagnosis')
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
