# MedAI Website

The official web interface for the MedAI Fracture Detection System. Built with Next.js 14, Tailwind CSS, and Shadcn UI.

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
BACKEND_URL=https://huggingface.co/spaces/username/medai-fracture-backend
```

## Deployment

This app is optimized for Vercel.

1. Push this code to GitHub.
2. Import project into Vercel.
3. Add `BACKEND_URL` environment variable in Vercel project settings.
