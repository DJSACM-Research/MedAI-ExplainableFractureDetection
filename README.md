# MedAI - Explainable Fracture Detection

A comprehensive medical imaging AI system for bone fracture detection using deep learning with explainable AI (XAI) capabilities.

## Features

- **Multi-Model Ensemble**: 9 trained models including DenseNet169, EfficientNet, MobileNetV2, Swin Transformer, MaxViT, and custom HyperColumn-CBAM architectures
- **Explainable AI**: Grad-CAM visualizations for model interpretability
- **RAG-Powered Knowledge Base**: ChromaDB-based retrieval augmented generation for medical knowledge
- **Multi-Agent System**: Specialized agents for diagnostics, education, and patient interaction
- **Interactive UI**: Streamlit-based web application

## Fracture Classes

The system classifies X-ray images into 8 categories:

- Comminuted
- Greenstick
- Healthy
- Oblique
- Oblique Displaced
- Spiral
- Transverse
- Transverse Displaced

## Project Structure

```
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── src/
│   └── medai/
│       ├── agents/             # AI agents for various tasks
│       │   ├── diagnostic_agent.py
│       │   ├── educational_agent.py
│       │   ├── explain_agent.py
│       │   ├── cross_validation_agent.py
│       │   ├── knowledge_agent.py
│       │   └── patient_agent.py
│       ├── training/           # Model training pipelines
│       │   └── pipeline.py
│       ├── analysis/           # Data analysis tools
│       │   └── analyze.py
│       └── models/             # Model architectures
├── scripts/                    # Utility scripts
│   ├── test_all_models.py
│   ├── test_hypercolumn.py
│   └── visualize_gradcam.py
├── updated_models/             # Trained model checkpoints
├── data/                       # Dataset files
├── notebooks/                  # Jupyter notebooks for experiments
├── tests/                      # Unit and integration tests
└── chroma_db/                  # Vector database for RAG
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/MedAI-ExplainableFractureDetection.git
cd MedAI-ExplainableFractureDetection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

### Model Architectures

| Model                              | Architecture           | Description                                |
| ---------------------------------- | ---------------------- | ------------------------------------------ |
| swin                               | Swin Transformer Small | Vision transformer with shifted windows    |
| densenet169                        | DenseNet-169           | Dense connections for feature reuse        |
| efficientnetv2                     | EfficientNet-B0        | Compound scaling architecture              |
| mobilenetv2                        | MobileNetV2            | Lightweight mobile architecture            |
| maxvit                             | MaxViT Tiny            | Multi-axis vision transformer              |
| hypercolumn_cbam_densenet169       | Custom                 | DenseNet169 + Hypercolumn + CBAM attention |
| hypercolumn_cbam_densenet169_focal | Custom                 | Above + Focal loss training                |
| hypercolumn_densenet169            | Custom                 | DenseNet169 + Hypercolumn features         |
| hypercolumn_densenet169_old        | Custom                 | Legacy hypercolumn model                   |

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for full dependencies

## License

See [LICENSE](LICENSE) for details.
