import os
import sys
import argparse
import torch
from PIL import Image
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- 1. CONFIGURATION ---
from src.utils import get_device, get_model, get_transforms

DEVICE = get_device()

# --- 2. DIAGNOSTIC AGENT CORE ---

class DiagnosticAgent:
    def __init__(self, checkpoint_path: str, model_name: str, num_classes: int, img_size: int, class_names: List[str]):
        self.device = DEVICE
        self.img_size = img_size
        self.class_names = class_names
        
        # 1. Load Model Architecture
        self.model = get_model(model_name, num_classes, pretrained=False).to(self.device)
        
        # 2. Load Weights from Checkpoint
        try:
            ck = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ck.get('model_state_dict', ck)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✅ Diagnostic Agent loaded model from {checkpoint_path} on {self.device}.")
        except FileNotFoundError:
            print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
            exit(1)
        except Exception as e:
            print(f"❌ Error loading model state: {e}")
            exit(1)

        # 3. Setup Transforms
        self.transform = get_transforms('val', self.img_size)

    def run_diagnosis(self, image_path: str) -> Dict[str, Any]:
        """
        Runs the image classification model, detects fractures, and outputs scores.
        
        This method includes the fix for FileNotFoundError by resolving the path.
        """
        
        # CRITICAL FIX: Convert relative path to absolute path for reliable file access
        full_image_path = os.path.abspath(image_path)

        if not os.path.exists(full_image_path):
            # Report the original path back to the user for clarity
            return {"error": f"Image file not found at {image_path}"}

        # 1. Image Loading and Preprocessing
        try:
            # Use the resolved full path for PIL to open
            img = Image.open(full_image_path).convert('RGB') 
        except Exception as e:
            return {"error": f"Failed to open image at {full_image_path}. Reason: {e}"}


        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 2. Model Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        # Softmax to get probabilities (confidence scores)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        
        # 3. Score Calculation
        
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        uncertainty = 1.0 - confidence
        predicted_class_name = self.class_names[predicted_idx]

        # Determine Fracture Presence (assuming 'Healthy' is a known class)
        is_fracture_detected = (predicted_class_name != 'Healthy')
        
        return {
            "image_path": image_path,
            "fracture_detected": is_fracture_detected,
            "predicted_class": predicted_class_name,
            "severity_type": predicted_class_name,  # Proxy for severity
            "confidence_score": confidence,
            "uncertainty_score": uncertainty,
            "all_probabilities": probabilities.cpu().numpy().tolist()
        }

# --- 3. EXECUTION ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a diagnostic agent on a single image.')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the image file to diagnose.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (e.g., outputs/swin_mps/best.pth)')
    parser.add_argument('--model', type=str, default='swin', choices=['swin', 'convnext', 'densenet'])
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--class-names', type=str, required=True, 
                        help='Comma-separated list of class names (e.g., "A,B,C")')
    
    args = parser.parse_args()

    # Convert class names string to a list
    class_names_list = [c.strip() for c in args.class_names.split(',')]

    # Ensure 'Healthy' is in the list for the 'fracture_detected' check to work reliably
    if 'Healthy' not in class_names_list:
        print("Warning: 'Healthy' class not found in --class-names list. Fracture detection may be inaccurate.")
    
    # Initialize the Agent
    agent = DiagnosticAgent(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        num_classes=args.num_classes,
        img_size=args.img_size,
        class_names=class_names_list
    )

    # Run the Diagnosis
    result = agent.run_diagnosis(args.image_path)

    # Output Results
    print("\n--- DIAGNOSTIC RESULTS ---")
    if "error" in result:
        print(f"Status: FAILED\nReason: {result['error']}")
    else:
        print(f"Status: SUCCESS")
        print(f"Image: {result['image_path']}")
        print(f"Fracture Detected: {'YES' if result['fracture_detected'] else 'NO'}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"--- Scores ---")
        print(f"Severity Type: {result['severity_type']}")
        print(f"Confidence Score: {result['confidence_score']:.4f}")
        print(f"Uncertainty Score: {result['uncertainty_score']:.4f}")
    print("--------------------------\n")
