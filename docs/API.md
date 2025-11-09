# API Reference - Patient Chat App

## Overview

This document provides a complete API reference for the `patient_chat_app.py` application and its integration with the agent system.

---

## Workflow Functions

### 1. Diagnostic Agent

**Function**: `run_diagnostic_agent(image_path: str) -> Dict[str, Any]`

Runs a single deep learning model (Swin Transformer by default) on an X-ray image.

**Parameters:**
- `image_path` (str): Path to the X-ray image file

**Returns:**
```python
{
    "image_path": str,
    "fracture_detected": bool,
    "predicted_class": str,
    "severity_type": str,
    "confidence_score": float (0-1),
    "uncertainty_score": float (0-1),
    "all_probabilities": List[float]
}
```

**Example:**
```python
from apps.patient_chat_app import run_diagnostic_agent

result = run_diagnostic_agent("xray.jpg")
print(f"Fracture: {result['fracture_detected']}")
print(f"Type: {result['predicted_class']}")
print(f"Confidence: {result['confidence_score']:.2%}")
```

**Exceptions:**
- `FileNotFoundError`: Image not found
- `RuntimeError`: Model checkpoint missing at `./outputs/best_swin.pth`

---

### 2. Ensemble Agent

**Function**: `run_ensemble_agent(image_path: str) -> Dict[str, Any]`

Runs an ensemble of 5 deep learning models and aggregates predictions using soft voting.

**Models Used:**
1. Swin Transformer
2. MobileNetV2
3. DenseNet169
4. EfficientNetV2
5. MaxViT

**Parameters:**
- `image_path` (str): Path to the X-ray image file

**Returns:**
```python
{
    "image_path": str,
    "ensemble_prediction": str,
    "ensemble_confidence": float (0-1),
    "individual_predictions": {
        "swin": {"class": str, "confidence": float},
        "mobilenetv2": {"class": str, "confidence": float},
        "densenet169": {"class": str, "confidence": float},
        "efficientnetv2": {"class": str, "confidence": float},
        "maxvit": {"class": str, "confidence": float}
    },
    "fracture_detected": bool
}
```

**Example:**
```python
from apps.patient_chat_app import run_ensemble_agent

result = run_ensemble_agent("xray.jpg")
print(f"Ensemble: {result['ensemble_prediction']}")
print(f"Confidence: {result['ensemble_confidence']:.2%}")
print(f"Fracture Detected: {result['fracture_detected']}")

# View individual model predictions
for model_name, pred in result['individual_predictions'].items():
    print(f"{model_name}: {pred['class']} ({pred['confidence']:.2%})")
```

**Exceptions:**
- `FileNotFoundError`: Image or model checkpoints not found
- `RuntimeError`: No models could be loaded

---

### 3. Educational Agent

**Function**: `run_educational_agent(diagnosis_result: Dict[str, Any], explanation_text: str = "") -> Dict[str, str]`

Translates technical diagnosis into patient-friendly language.

**Parameters:**
- `diagnosis_result` (Dict): Diagnosis output from diagnostic/ensemble agent
- `explanation_text` (str, optional): Technical explanation from explainability agent

**Returns:**
```python
{
    "patient_summary": str,
    "patient_severity_assessment": str,
    "next_steps_action_plan": str
}
```

**Example:**
```python
from apps.patient_chat_app import run_educational_agent

diagnosis = {
    "fracture_detected": True,
    "predicted_class": "Transverse",
    "confidence_score": 0.92
}

result = run_educational_agent(diagnosis, "Technical explanation...")
print("Patient Summary:")
print(result['patient_summary'])
print("\nSeverity:")
print(result['patient_severity_assessment'])
print("\nNext Steps:")
print(result['next_steps_action_plan'])
```

**Returned Severity Levels:**
- `None` - Healthy bone, no fracture
- `Mild` - Greenstick fractures
- `Moderate` - Transverse, Oblique fractures
- `Serious` - Spiral, Displaced fractures
- `Severe` - Comminuted fractures

---

### 4. Explainability Agent

**Function**: `run_explainability_agent(diagnosis_result: Dict[str, Any]) -> str`

Generates human-readable explanations of model predictions using Grad-CAM heatmap analysis.

**Parameters:**
- `diagnosis_result` (Dict): Diagnosis output from diagnostic/ensemble agent

**Returns:**
- `str`: Text-based explanation of model decision

**Example:**
```python
from apps.patient_chat_app import run_explainability_agent

diagnosis = {
    "predicted_class": "Greenstick",
    "confidence_score": 0.92,
    "fracture_detected": True
}

explanation = run_explainability_agent(diagnosis)
print(explanation)
# Output: "The model detected a clear greenstick fracture pattern..."
```

**Explanation Components:**
- What the model detected
- Location of activation (centroid analysis)
- Confidence level
- Type-specific notes

---

### 5. Knowledge Agent

**Function**: `run_knowledge_agent(diagnosis: str, confidence: float) -> Dict[str, Any]`

Retrieves medical knowledge and evidence-based guidelines for a diagnosis.

**Parameters:**
- `diagnosis` (str): Diagnosis class name (e.g., "Transverse")
- `confidence` (float): Confidence score (0-1)

**Returns:**
```python
{
    "Diagnosis": str,
    "Ensemble_Confidence": str,
    "Type": str,
    "ICD_Code": str,
    "Severity": str,
    "Guidelines": List[str]
}
```

**Example:**
```python
from apps.patient_chat_app import run_knowledge_agent

result = run_knowledge_agent("Transverse", 0.92)
print(f"Definition: {result['Type']}")
print(f"Severity: {result['Severity']}")
print(f"ICD Code: {result['ICD_Code']}")
print("Guidelines:")
for guideline in result['Guidelines']:
    print(f"  - {guideline}")
```

**Available Diagnoses:**
- Comminuted
- Greenstick
- Healthy
- Oblique
- Oblique Displaced
- Spiral
- Transverse
- Transverse Displaced

---

### 6. Complete Workflow

**Function**: `run_complete_workflow(image_path: str) -> Dict[str, Any]`

Executes the complete diagnostic pipeline: Ensemble → Educational → Explainability → Knowledge agents.

**Parameters:**
- `image_path` (str): Path to the X-ray image file

**Returns:**
```python
{
    "ensemble_result": {
        "ensemble_prediction": str,
        "ensemble_confidence": float,
        "individual_predictions": Dict,
        "fracture_detected": bool
    },
    "educational_result": {
        "patient_summary": str,
        "patient_severity_assessment": str,
        "next_steps_action_plan": str
    },
    "explanation_result": str,
    "knowledge_result": {
        "Diagnosis": str,
        "Type": str,
        "Severity": str,
        "Guidelines": List[str],
        "ICD_Code": str
    }
}
```

**Example:**
```python
from apps.patient_chat_app import run_complete_workflow

result = run_complete_workflow("xray.jpg")

# Display full report
print("=== DIAGNOSTIC REPORT ===")
print(f"Diagnosis: {result['ensemble_result']['ensemble_prediction']}")
print(f"Confidence: {result['ensemble_result']['ensemble_confidence']:.2%}")
print("\n=== PATIENT SUMMARY ===")
print(result['educational_result']['patient_summary'])
print("\n=== MEDICAL GUIDELINES ===")
for guideline in result['knowledge_result']['Guidelines']:
    print(f"• {guideline}")
```

**Workflow Steps:**
1. Run ensemble agent for diagnosis
2. Run educational agent for patient summary
3. Run explainability agent for explanation
4. Run knowledge agent for medical guidelines

---

## LLM Chat Agent

### PatientInteractionAgent Class

**Class**: `PatientInteractionAgent`

Handles communication with Llama 3 model via Ollama for patient Q&A.

#### Constructor

**`__init__(medical_summary: Dict[str, Any], patient_history: Dict[str, Any])`**

Initialize the agent with medical context.

**Parameters:**
- `medical_summary` (Dict): Medical diagnosis information
  ```python
  {
      "Diagnosis": str,
      "Ensemble_Confidence": str,
      "Type": str,
      "Severity": str,
      "Guidelines": List[str]
  }
  ```
- `patient_history` (Dict): Patient demographic and medical history
  ```python
  {
      "age": int,
      "gender": str,
      "history": str
  }
  ```

**Raises:**
- `ConnectionError`: If Ollama server is not running

**Example:**
```python
from apps.patient_chat_app import PatientInteractionAgent

medical_summary = {
    "Diagnosis": "Transverse",
    "Ensemble_Confidence": "0.92",
    "Type": "A clean break straight across the bone",
    "Severity": "Moderate",
    "Guidelines": [
        "Immobilization with cast",
        "Regular X-rays to monitor healing",
        "Physical therapy after healing"
    ]
}

patient_history = {
    "age": 45,
    "gender": "Female",
    "history": "No major past issues"
}

try:
    agent = PatientInteractionAgent(medical_summary, patient_history)
except ConnectionError as e:
    print(f"Error: {e}")
    print("Make sure Ollama is running: ollama serve")
```

#### Methods

**`get_response(query: str) -> str`**

Get a response from Llama 3 model for a patient query.

**Parameters:**
- `query` (str): Patient's question about their diagnosis

**Returns:**
- `str`: LLM-generated response

**Example:**
```python
response = agent.get_response("How long will my recovery take?")
print(response)
```

**Response Characteristics:**
- Non-technical language suitable for patients
- Empathetic and reassuring tone
- Grounded in provided medical context
- Recommends consulting with healthcare provider
- Typical length: 2-4 paragraphs

**Example Response:**
```
Great question! Recovery time for your type of break typically takes 
6-12 weeks, depending on how well the bone heals and your overall health. 
Most of this time will be spent in a cast or splint to keep the break stable.

After the cast comes off, you'll likely need physical therapy to regain 
strength and movement. This can take another few weeks.

Everyone heals at their own pace, so it's important to follow your doctor's 
instructions closely. Make sure to attend all your follow-up appointments 
and physical therapy sessions.

For personalized recovery expectations, please discuss with your orthopedic 
specialist or doctor.
```

**Exceptions:**
- `ConnectionError`: If Ollama is not responding
- `RequestException`: If HTTP request fails
- `TimeoutError`: If response takes too long (>300 seconds)

---

## Data Classes

### Diagnosis Result
```python
{
    "image_path": str,
    "fracture_detected": bool,
    "predicted_class": str,
    "confidence_score": float,
    "uncertainty_score": float,
    "all_probabilities": List[float],
    "severity_type": str
}
```

### Ensemble Result
```python
{
    "image_path": str,
    "ensemble_prediction": str,
    "ensemble_confidence": float,
    "individual_predictions": Dict[str, Dict],
    "fracture_detected": bool
}
```

### Educational Result
```python
{
    "patient_summary": str,
    "patient_severity_assessment": str,
    "next_steps_action_plan": str
}
```

### Knowledge Result
```python
{
    "Diagnosis": str,
    "Ensemble_Confidence": str,
    "Type": str,
    "ICD_Code": str,
    "Severity": str,
    "Guidelines": List[str]
}
```

---

## Constants

```python
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique",
    "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
]
NUM_CLASSES = 8
IMG_SIZE = 224
```

---

## Error Handling

### Common Exceptions

```python
# Model checkpoint not found
FileNotFoundError: "Checkpoint file not found at {checkpoint_path}"

# Ollama not running
ConnectionError: "Ollama server is not running. Please start Ollama."

# Image processing failed
Exception: "Failed to open image at {image_path}. Reason: {error}"

# No models loaded
RuntimeError: "No models were successfully loaded. Cannot run ensemble."

# Invalid diagnosis class
ValueError: "Diagnosis not found in the knowledge base."

# Ollama timeout
TimeoutError: "Request to Ollama server timed out"
```

### Error Handling Example

```python
from apps.patient_chat_app import run_complete_workflow

try:
    result = run_complete_workflow("xray.jpg")
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Diagnosis successful!")
        
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ConnectionError as e:
    print(f"Connection error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Configuration

### Model Paths
```python
./outputs/best_swin.pth
./outputs/best_mobilenetv2.pth
./outputs/best_densenet169.pth
./outputs/best_efficientnetv2.pth
./outputs/best_maxvit.pth
```

### Ollama Configuration
```bash
# Install Ollama
brew install ollama  # macOS
# or download from https://ollama.ai

# Pull Llama 3 model
ollama pull llama3

# Run Ollama server
ollama serve
```

---

## Performance Metrics

| Operation | Time | Hardware |
|-----------|------|----------|
| Single Model Inference | 0.5-1s | GPU |
| Ensemble Inference | 2-5s | GPU |
| Educational Translation | <100ms | CPU |
| Knowledge Lookup | <50ms | CPU |
| LLM Response | 10-30s | CPU (Ollama) |
| Complete Workflow | 15-40s | GPU + CPU |

---

## Testing

### Unit Tests
```bash
python -m pytest tests/unit/test_patient_chat_app.py -v
```

### Integration Tests
```bash
python -m pytest tests/integration/ -v
```

### All Tests
```bash
bash ./run_tests.sh
```

### Test Coverage
```bash
bash ./run_tests.sh coverage
```

---

## Migration Guide

### From Old App to New App

**Old Code:**
```python
# Old single-purpose app
from apps.patient_chat_app import PatientInteractionAgent

medical_summary = {...}
patient_history = {...}
agent = PatientInteractionAgent(medical_summary, patient_history)
response = agent.get_response(query)
```

**New Code (with full workflow):**
```python
# New app with full workflow support
from apps.patient_chat_app import (
    run_complete_workflow,
    PatientInteractionAgent
)

# Get full diagnosis
result = run_complete_workflow("xray.jpg")

# Extract diagnosis info
medical_summary = result['knowledge_result']
patient_history = {"age": 45, "gender": "Female", "history": ""}

# Optional: Use chat
agent = PatientInteractionAgent(medical_summary, patient_history)
response = agent.get_response("What should I do?")
```

---

## Versioning

**Current Version**: 2.0
**Release Date**: November 9, 2025
**Python Version**: 3.9+
**PyTorch Version**: 1.9+
**Streamlit Version**: 1.0+

---

## License

See LICENSE file in repository.

---

**API Reference Version**: 2.0
**Last Updated**: November 9, 2025
