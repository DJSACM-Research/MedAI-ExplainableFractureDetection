import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // Forward to Python Backend
    // In production, this URL would be the HF Space URL
    const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:7860";

    const backendFormData = new FormData();
    backendFormData.append("file", file);

    // Forward optional analysis options from the frontend
    const useConformal = formData.get("use_conformal");
    const ensembleMode = formData.get("ensemble_mode");
    const stackerPath = formData.get("stacker_path");

    if (useConformal !== null)
      backendFormData.append("use_conformal", String(useConformal));
    if (ensembleMode !== null)
      backendFormData.append("ensemble_mode", String(ensembleMode));
    if (stackerPath !== null)
      backendFormData.append("stacker_path", String(stackerPath));

    const response = await fetch(`${BACKEND_URL}/diagnose`, {
      method: "POST",
      body: backendFormData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Backend Error:", errorText);
      return NextResponse.json(
        { error: "Failed to process image on backend" },
        { status: 500 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Diagnosis Error:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
