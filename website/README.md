# Project Website

The official web interface for the fracture detection system. Built with Next.js 14, Tailwind CSS, and Shadcn UI.

## Local Development

1. **Install Dependencies**

   ```bash
   npm install
   ```

2. **Run Development Server**
   ```bash
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) with your browser.

## Backend Connection

This frontend requires a running Python backend to perform inference using the custom PyTorch models.

1. Deploy the code in `../backend_hf` to a Hugging Face Space (Docker or Python SDK).
2. Set the backend URL in an environment variable `BACKEND_URL`.

**Example `.env.local`:**

```
BACKEND_URL=https://huggingface.co/spaces/username/fracture-backend
```

## Backend feature notes

The backend now includes a few optional features that the frontend can surface when configured:

- Conformal prediction: The backend can produce a `conformal_set` for each inference when enabled. Calibrate a nonconformity threshold on validation using `scripts/prepare_val_and_calibrate.py` and provide the resulting `conformal_threshold.txt` to the backend or app.
- Stacking ensemble: A trained stacking pipeline `outputs/stacker.joblib` (scaler + logistic regression) can be used instead of weighted averaging. The frontend includes a sidebar option to toggle `stacking` and provide the stacker path.
- Per-model Grad-CAM previews: The backend generates per-model Grad-CAM overlays when the `pytorch-grad-cam` dependency is available. The frontend's Explainability panel supports toggling per-model overlays.

When deploying the backend, ensure the following artifacts are available under the backend project or accessible paths: `outputs/stacker.joblib`, `conformal_threshold.txt`, and model checkpoints under `models/`.

## Deployment

This app is optimized for Vercel.

1. Push this code to GitHub.
2. Import project into Vercel.
3. Add `BACKEND_URL` environment variable in Vercel project settings.
