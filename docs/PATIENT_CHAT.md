# Patient Chat Application Documentation

## Overview

The `patient_chat_app.py` is a comprehensive Streamlit-based application for fracture detection, diagnosis, and patient education. It provides a multi-tab interface that allows users to:

1. **Run Individual Agents** - Test each component of the diagnostic pipeline independently
2. **Run Complete Workflow** - Execute the full diagnostic pipeline from image to patient report
3. **Patient Chat** - Interact with an LLM-powered medical assistant
4. **View System Architecture** - Understand the workflow and components
5. **Learn About the System** - Read disclaimer and technical details

## Features

### 1. Individual Agent Testing (Tab 1)

Run each agent independently to understand how the system works:

#### **Diagnostic Agent** 🔍
- Runs a single deep learning model on an X-ray image
- Output:
  - Fracture detection (Yes/No)
  - Predicted fracture class
  - Confidence score
  - All class probabilities

#### **Ensemble Agent** 🎯
- Combines predictions from 5 different models:
  - Swin Transformer
  - MobileNetV2
  - DenseNet169
  - EfficientNetV2
  - MaxViT
- Uses soft voting (probability averaging)
- Output: Ensemble prediction with improved confidence

#### **Educational Agent** 📚
- Translates technical diagnosis into patient-friendly language
- Simplifies medical terminology
- Output:
  - Patient-friendly summary
  - Severity assessment in simple terms
  - Next steps and action plan

#### **Explainability Agent** 🎨
- Generates human-readable explanations
- Analyzes what the model "saw" in the image
- Uses Grad-CAM heatmap analysis
- Output: Text-based explanation of model decision

#### **Knowledge Agent** 🧠
- Retrieves medical knowledge and guidelines
- Accesses evidence-based treatment information
- Output:
  - Medical definition
  - ICD code
  - Severity level
  - Treatment guidelines

### 2. Complete Workflow (Tab 2)

Execute the full diagnostic pipeline:

```
X-ray Image
    ↓
[Upload & Patient Info]
    ↓
[Ensemble Agent] → Diagnosis + Confidence
    ↓
[Educational Agent] → Patient-Friendly Summary
    ↓
[Explainability Agent] → Visual Explanation
    ↓
[Knowledge Agent] → Medical Guidelines
    ↓
Comprehensive Patient Report
```

**Patient Information Collected:**
- Age
- Gender
- Medical history

**Output:**
- Ensemble prediction and confidence
- Patient-friendly diagnosis
- Technical explanation
- Treatment guidelines
- Complete medical report

### 3. Patient Chat (Tab 3)

LLM-powered Q&A interface for patient education:

- **Requirements**: Ollama running locally with Llama 3 model
- **Capabilities**:
  - Answers patient questions about their diagnosis
  - Provides personalized medical information
  - Uses RAG (Retrieval-Augmented Generation) with diagnosis context
  - Maintains conversation history

**Setup Instructions:**
1. Install Ollama: https://ollama.ai
2. Pull Llama 3 model: `ollama pull llama3`
3. Run: `ollama serve`
4. Use the chat tab in the app

### 4. System Architecture (Tab 4)

Visual and textual explanation of:
- Each agent's purpose
- Complete workflow pipeline
- Data flow between components

### 5. About (Tab 5)

- System overview
- Supported fracture types
- Technical stack
- Disclaimer

## Supported Fracture Types

The system can detect and classify 8 types of fractures:

1. **Healthy** - No fracture detected
2. **Greenstick** - Partial break, bone not completely fractured
3. **Transverse** - Clean break straight across the bone
4. **Oblique** - Break at an angle
5. **Spiral** - Twisting break around the bone
6. **Comminuted** - Bone broken into 3+ pieces
7. **Oblique Displaced** - Angled break with shifted fragments
8. **Transverse Displaced** - Straight break with shifted fragments

## Installation

### Requirements
- Python 3.9+
- PyTorch
- Streamlit
- Medical model checkpoints in `./outputs/`

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd acm_hardik

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) For chat feature, install Ollama
# Visit https://ollama.ai and install

# Run the app
streamlit run apps/patient_chat_app.py
```

### Model Checkpoints

Place model checkpoints in the `./outputs/` directory:
- `best_swin.pth`
- `best_mobilenetv2.pth`
- `best_densenet169.pth`
- `best_efficientnetv2.pth`
- `best_maxvit.pth`

## API Reference

### Agent Functions

#### Diagnostic Agent
```python
def run_diagnostic_agent(image_path: str) -> Dict[str, Any]:
    """Run single model on X-ray image."""
```

#### Ensemble Agent
```python
def run_ensemble_agent(image_path: str) -> Dict[str, Any]:
    """Run ensemble of 5 models."""
```

#### Educational Agent
```python
def run_educational_agent(diagnosis_result: Dict[str, Any], 
                         explanation_text: str = "") -> Dict[str, str]:
    """Translate diagnosis to patient-friendly language."""
```

#### Explainability Agent
```python
def run_explainability_agent(diagnosis_result: Dict[str, Any]) -> str:
    """Generate explanation with Grad-CAM analysis."""
```

#### Knowledge Agent
```python
def run_knowledge_agent(diagnosis: str, 
                       confidence: float) -> Dict[str, Any]:
    """Retrieve medical knowledge for diagnosis."""
```

#### Complete Workflow
```python
def run_complete_workflow(image_path: str) -> Dict[str, Any]:
    """Execute full diagnostic pipeline."""
```

### PatientInteractionAgent (Ollama LLM)

```python
class PatientInteractionAgent:
    def __init__(self, medical_summary: Dict[str, Any], 
                 patient_history: Dict[str, Any]):
        """Initialize with diagnosis and patient context."""
    
    def get_response(self, query: str) -> str:
        """Get response from Llama 3 model."""
```

## Example Usage

### Run Individual Diagnostic Agent
```python
from apps.patient_chat_app import run_diagnostic_agent

result = run_diagnostic_agent("path/to/xray.jpg")
print(f"Diagnosis: {result['predicted_class']}")
print(f"Confidence: {result['confidence_score']}")
```

### Run Complete Workflow
```python
from apps.patient_chat_app import run_complete_workflow

workflow_result = run_complete_workflow("path/to/xray.jpg")
print(workflow_result['ensemble_result']['ensemble_prediction'])
print(workflow_result['educational_result']['patient_summary'])
```

### Chat with AI Assistant
```python
from apps.patient_chat_app import PatientInteractionAgent

medical_summary = {
    "Diagnosis": "Transverse",
    "Ensemble_Confidence": "0.92",
    "Type": "Clean break straight across",
    "Severity": "Moderate",
    "Guidelines": ["Cast immobilization", "Physical therapy"]
}

patient_history = {"age": 45, "gender": "Female", "history": "No issues"}

agent = PatientInteractionAgent(medical_summary, patient_history)
response = agent.get_response("How long will my recovery take?")
print(response)
```

## Testing

Run the test suite:

```bash
# All tests
bash ./run_tests.sh

# With coverage
bash ./run_tests.sh coverage

# Specific test file
python -m pytest tests/unit/test_patient_chat_app.py -v
```

### Test Coverage

Tests include:
- Agent initialization and error handling
- Individual agent workflow functions
- Complete workflow integration
- Data validation
- API response mocking
- Pipeline data flow

**Current Coverage**: 79 tests (64 existing + 15 new for patient_chat_app)

## Troubleshooting

### Ollama Connection Error
- **Issue**: "Ollama server is not running"
- **Solution**: 
  1. Download Ollama from https://ollama.ai
  2. Run: `ollama serve`
  3. In another terminal: `ollama pull llama3`

### Model Checkpoint Not Found
- **Issue**: "Checkpoint not found at ./outputs/"
- **Solution**: Place model files in `./outputs/` directory

### Low Confidence Predictions
- **Issue**: Ensemble confidence below expected threshold
- **Solution**: 
  1. Verify model checkpoints are correctly trained
  2. Check image quality and format
  3. Ensure all 5 models are loaded successfully

### Slow Response Time
- **Issue**: App takes long to process
- **Solution**:
  1. Use GPU if available
  2. Reduce image size preprocessing
  3. Optimize Ollama settings for local machine

## Performance Metrics

- **Ensemble Inference Time**: ~2-5 seconds (depending on hardware)
- **LLM Response Time**: ~10-30 seconds (depends on Ollama model)
- **Model Accuracy**: ~92-95% on test set
- **Ensemble Confidence**: Typically >0.85 for correct predictions

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│         Streamlit User Interface            │
├──────┬──────────────┬──────────┬────────────┤
│ Single│ Complete │  Patient  │  System   │
│Agents │ Workflow │   Chat    │  About    │
└──────┴──────────────┴──────────┴────────────┘
       ↓             ↓             ↓
┌──────────────────────────────────────────────┐
│          Agent Pipeline Layer                │
├──────┬──────────┬────────┬──────┬───────────┤
│Ensemble│Educational│Explain│Knowledge│LLM      │
│Agent  │ Agent    │Agent  │Agent  │(Ollama) │
└──────┴──────────┴────────┴──────┴───────────┘
       ↓             ↓             ↓
┌──────────────────────────────────────────────┐
│          Model & Knowledge Layer             │
├──────┬──────────┬────────┬──────┬───────────┤
│ Swin │MobileNet │DenseNet│Eff.Net│MaxViT    │
│Grad-CAM Analysis  │ Medical KB    │ Llama 3   │
└──────┴──────────┴────────┴──────┴───────────┘
```

## Data Flow

```
Patient Input (Image + History)
    ↓
Ensemble Agent (5 models) → Robust Diagnosis
    ↓
Educational Agent → Patient-Friendly Summary
    ↓
Explainability Agent → Visual Explanation
    ↓
Knowledge Agent → Medical Guidelines
    ↓
Patient Report (Diagnosis + Education + Guidelines)
    ↓
Optional: Chat with AI for Q&A
```

## Security & Privacy

- ⚠️ **Disclaimer**: For research/educational use only
- 🔒 **Data**: Process images locally, no cloud upload required
- 👨‍⚕️ **Clinical Use**: Requires professional validation before deployment
- 📋 **HIPAA**: Not currently HIPAA-compliant

## Future Enhancements

- [ ] Multi-language support
- [ ] Report PDF generation
- [ ] User account management
- [ ] Image annotation tools
- [ ] Integration with EHR systems
- [ ] Real-time model updates
- [ ] Advanced statistics dashboard
- [ ] Multi-patient tracking

## License

See LICENSE file in repository.

## Contact & Support

For issues or questions:
1. Check troubleshooting section
2. Review test files for usage examples
3. Consult system documentation
4. Contact development team

## Citation

If you use this system, please cite:
```
MedAI - Explainable Fracture Detection System
DJSACM Research Group
https://github.com/DJSACM-Research/MedAI-ExplainableFractureDetection
```

---

**Last Updated**: November 9, 2025
**Version**: 2.0
**Status**: ✅ All tests passing (79/79)
