import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    // Forward the original multipart body and Content-Type header
    const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:7860";
    const rawBody = await req.arrayBuffer();
    const contentType =
      req.headers.get("content-type") || "multipart/form-data";

    // Try configured BACKEND_URL first, fall back to localhost (dev) if it fails.
    let response = await fetch(`${BACKEND_URL}/diagnose/report`, {
      method: "POST",
      body: rawBody,
      headers: {
        "content-type": contentType,
      },
    });
    if (!response.ok) {
      console.warn(
        `Primary backend ${BACKEND_URL} returned ${response.status}; trying http://127.0.0.1:7860`
      );
      try {
        response = await fetch(`http://127.0.0.1:7860/diagnose/report`, {
          method: "POST",
          body: rawBody,
          headers: {
            "content-type": contentType,
          },
        });
      } catch (e) {
        console.error("Fallback to local backend failed", e);
      }
    }

    if (!response.ok) {
      const text = await response.text();
      console.error("Backend report error:", text);
      return NextResponse.json(
        { error: "Backend report failed" },
        { status: 500 }
      );
    }

    // Stream the response back to the client
    const arrayBuffer = await response.arrayBuffer();
    const headers: Record<string, string> = {
      "Content-Type":
        response.headers.get("content-type") || "application/octet-stream",
    };
    const disposition = response.headers.get("content-disposition");
    if (disposition) headers["Content-Disposition"] = disposition;

    return new NextResponse(arrayBuffer, { status: 200, headers });
  } catch (err) {
    console.error(err);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
