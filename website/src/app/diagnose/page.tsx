
"use client";

import { useState } from "react";
import { UploadCloud, File, AlertCircle, CheckCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ProbabilityChart } from "@/components/medai/ProbabilityChart";
import { ChatInterface } from "@/components/medai/ChatInterface";
import ReactMarkdown from 'react-markdown';
import Image from "next/image";

interface DiagnosisResponse {
  prediction: {
    ensemble_prediction: string;
    ensemble_confidence: number;
    fracture_detected: boolean;
    all_probabilities: Record<string, number>;
  };
  explanation: {
    text: string;
    heatmap_b64: string | null;
  };
  educational: {
    patient_summary: string;
    severity_layman: string;
    next_steps_action_plan: string;
  };
  knowledge_base: Record<string, any>;
}

export default function DiagnosePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DiagnosisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null); // Reset previous results
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/diagnose", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Analysis failed. Backend might be offline.");

      const data = await response.json();
      if (data.error) throw new Error(data.error);
      if (!data.prediction) throw new Error("Invalid response from backend (Check if backend is running)");
      
      setResult(data);
    } catch (err: any) {
      setError(err.message || "An error occurred during analysis.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-6 lg:p-12">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Diagnosis Dashboard</h1>
            <p className="text-muted-foreground mt-2">
              Upload an X-ray to run the multi-agent analysis pipeline.
            </p>
          </div>
        </div>

        {/* Upload Section */}
        <div className="grid lg:grid-cols-3 gap-8">
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle>X-Ray Upload</CardTitle>
              <CardDescription>Supported formats: PNG, JPG, JPEG</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div 
                className={`
                  border-2 border-dashed rounded-xl p-8 text-center transition-colors
                  ${file ? 'border-primary/50 bg-primary/5' : 'border-neutral-700 hover:border-neutral-500'}
                `}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
              >
                {!preview ? (
                  <div className="space-y-4">
                    <div className="h-12 w-12 rounded-full bg-neutral-800 flex items-center justify-center mx-auto">
                      <UploadCloud className="h-6 w-6 text-neutral-400" />
                    </div>
                    <div>
                      <span className="font-medium text-primary cursor-pointer hover:underline">
                        <label htmlFor="file-upload" className="cursor-pointer">Click to upload</label>
                      </span>
                      <span className="text-neutral-500"> or drag and drop</span>
                    </div>
                    <input 
                      id="file-upload"
                      type="file" 
                      className="hidden" 
                      onChange={handleFileChange}
                      accept="image/*"
                    />
                  </div>
                ) : (
                  <div className="relative">
                    <img src={preview} alt="Preview" className="rounded-lg max-h-[300px] mx-auto object-contain" />
                    <Button 
                      variant="secondary" 
                      size="sm" 
                      className="absolute top-2 right-2"
                      onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                    >
                      Change
                    </Button>
                  </div>
                )}
              </div>

              {error && (
                <div className="p-4 rounded-lg bg-destructive/10 text-destructive text-sm flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" /> {error}
                </div>
              )}

              <Button 
                className="w-full h-12 text-lg" 
                onClick={handleAnalyze} 
                disabled={!file || loading}
              >
                {loading ? <><Loader2 className="mr-2 h-5 w-5 animate-spin" /> Analyzing...</> : "Run Analysis"}
              </Button>
            </CardContent>
          </Card>

          {/* Results Area */}
          <div className="lg:col-span-2 space-y-8">
            {result ? (
              <>
                {/* Top Stats Cards */}
                <div className="grid md:grid-cols-2 gap-4">
                  <Card className={result.prediction.fracture_detected ? "border-red-500/50 bg-red-500/5" : "border-green-500/50 bg-green-500/5"}>
                    <CardHeader className="pb-2">
                       <CardTitle className="text-lg font-medium text-muted-foreground">Primary Diagnosis</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold flex items-center gap-3">
                         {result.prediction.ensemble_prediction}
                         {result.prediction.fracture_detected ? 
                           <AlertCircle className="h-6 w-6 text-red-500" /> : 
                           <CheckCircle className="h-6 w-6 text-green-500" />
                         }
                      </div>
                      <p className="text-muted-foreground mt-1">
                        Confidence: {(result.prediction.ensemble_confidence * 100).toFixed(1)}%
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                       <CardTitle className="text-lg font-medium text-muted-foreground">Severity</CardTitle>
                    </CardHeader>
                    <CardContent>
                       <div className="text-3xl font-bold">
                         {result.educational.severity_layman.split(" ")[0]} 
                         <span className="text-base font-normal text-muted-foreground ml-2">
                           ({result.knowledge_base.Severity_Rating || "Unknown"})
                         </span>
                       </div>
                       <p className="text-muted-foreground mt-1 text-sm">
                         {result.knowledge_base.Type_Definition}
                       </p>
                    </CardContent>
                  </Card>
                </div>

                {/* Explanation & Heatmap */}
                <div className="grid md:grid-cols-2 gap-8">
                   <Card>
                     <CardHeader><CardTitle>AI Explanation & Grad-CAM</CardTitle></CardHeader>
                     <CardContent className="space-y-4">
                        {result.explanation.heatmap_b64 && (
                          <div className="rounded-lg overflow-hidden border border-white/10">
                            <img 
                              src={`data:image/png;base64,${result.explanation.heatmap_b64}`} 
                              alt="Grad-CAM Heatmap" 
                              className="w-full object-cover"
                            />
                          </div>
                        )}
                        <div className="text-sm leading-relaxed text-muted-foreground">
                          <ReactMarkdown>{result.explanation.text}</ReactMarkdown>
                        </div>
                     </CardContent>
                   </Card>

                   <ProbabilityChart probabilities={result.prediction.all_probabilities} />
                </div>

                {/* Educational Content & Chat */}
                <div className="grid md:grid-cols-2 gap-8">
                  <Card className="h-full">
                    <CardHeader><CardTitle>Patient Summary & Action Plan</CardTitle></CardHeader>
                    <CardContent className="prose prose-invert max-w-none text-sm space-y-4">
                      <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/20">
                          <p className="font-medium text-blue-200">{result.educational.patient_summary}</p>
                      </div>
                      <div className="whitespace-pre-wrap">
                        <ReactMarkdown>{result.educational.next_steps_action_plan}</ReactMarkdown>
                      </div>
                    </CardContent>
                  </Card>

                  <ChatInterface context={result.knowledge_base} />
                </div>

              </>
            ) : (
              <div className="h-full flex items-center justify-center border-2 border-dashed border-neutral-800 rounded-xl min-h-[400px]">
                <div className="text-center text-muted-foreground max-w-md">
                   {loading ? (
                     <div className="space-y-4">
                        <Loader2 className="h-10 w-10 animate-spin mx-auto text-primary" />
                        <p>Running multi-agent diagnostics...</p>
                     </div>
                   ) : (
                     <>
                       <Activity className="h-12 w-12 mx-auto mb-4 opacity-20" />
                       <h3 className="text-lg font-medium text-white mb-2">No Analysis Results</h3>
                       <p>Upload an X-ray image to start the diagnosis process.</p>
                     </>
                   )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Activity(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  )
}
