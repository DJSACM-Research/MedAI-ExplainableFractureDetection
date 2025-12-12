# MedAI Inference Backend (Hugging Face Spaces)

This backend serves the custom PyTorch models for the MedAI Fracture Detection System using FastAPI. It is designed to be deployed on **Hugging Face Spaces**.

## Deployment Instructions

1. **Create a New Space** on Hugging Face.
   - SDK: **Docker** (recommended for custom system dependencies) or **Gradio** (if you adapt the script).
   - This template is ready for a basic Python environment or Docker.

2. **Upload Models**
   - Place your `.pth` model files (e.g., `best_hypercolumn_cbam_densenet169.pth`) in a `models/` directory in the Space.
   - You can git lfs track them.

3. **Update `models.py` / `load_models()`**
   - Edit `app.py` to actually load your specific `.pth` files in the `load_models()` function using `torch.load()`.
   - Ensure the class definitions in `app.py` match *exactly* with how the models were saved. (The provided `app.py` includes the `HypercolumnCBAMDenseNet` class as found in the original repo).

4. **Deploy**
   - Push `app.py`, `requirements.txt`, and your models to the Space.
   - The API will be available at `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME/diagnose`.

## Local Testing

```bash
pip install -r requirements.txt
python app.py
```
Server runs at `http://localhost:7860`.
