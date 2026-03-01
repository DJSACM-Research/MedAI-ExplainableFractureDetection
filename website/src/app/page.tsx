import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  Activity,
  ShieldCheck,
  Brain,
  Stethoscope,
  Lightbulb,
  MessageSquare,
} from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Hero Section */}
      <header className="px-6 lg:px-12 h-20 flex items-center justify-between border-b border-white/10 backdrop-blur-md sticky top-0 z-50">
          <div className="flex items-center gap-2">
          <Activity className="h-6 w-6 text-blue-500" />
          <span className="font-bold text-xl tracking-tight">Fracture Detection AI</span>
        </div>
        <nav className="hidden md:flex gap-8 text-sm font-medium text-muted-foreground">
          <Link href="#features" className="hover:text-white transition-colors">
            Features
          </Link>
          <Link
            href="#how-it-works"
            className="hover:text-white transition-colors"
          >
            How it Works
          </Link>
          {/* <Link href="#" className="hover:text-white transition-colors">
            GitHub
          </Link> */}
        </nav>
        <Link href="/diagnose">
          <Button
            variant="default"
            className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20"
          >
            Launch App
          </Button>
        </Link>
      </header>

      <main className="flex-1">
        {/* Hero */}
        <section className="relative pt-20 pb-32 px-6 lg:px-12 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-blue-500/10 to-transparent pointer-events-none" />
          <div className="max-w-4xl mx-auto text-center space-y-8 relative z-10">
            <div className="inline-flex items-center rounded-full border border-blue-500/30 bg-blue-500/10 px-3 py-1 text-sm font-medium text-blue-400">
              <span className="flex h-2 w-2 rounded-full bg-blue-500 mr-2 animate-pulse"></span>
              v2.0 Now Available with Multi-Agent Systems
            </div>
            <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-blue-100 to-blue-300">
              Advanced Fracture Detection <br /> Powered by AI
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              A state-of-the-art multi-agent system combining deep learning
              ensembles, explainable AI, and medical knowledge retrieval for
              accurate and transparent diagnoses.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
              <Link href="/diagnose">
                <Button
                  size="lg"
                  className="h-12 px-8 text-lg bg-blue-600 hover:bg-blue-700 shadow-xl shadow-blue-500/20 transition-all hover:scale-105"
                >
                  Start Diagnosis <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </Link>
              <Link href="#features">
                <Button
                  variant="outline"
                  size="lg"
                  className="h-12 px-8 text-lg border-white/10 hover:bg-white/5"
                >
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section id="features" className="py-24 px-6 lg:px-12 bg-black/20">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-16 space-y-4">
              <h2 className="text-3xl md:text-4xl font-bold">Why This System?</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Our system uses a novel multi-agent architecture to provide more
                than just a prediction. It explains, educates, and answers your
                questions.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              {[
                {
                  icon: Brain,
                  title: "Ensemble Intelligence",
                  desc: "Combines 4 diverse architectures — MaxViT, YOLOv2.6m, HyperColumn-CBAM-DenseNet169, and RAD-DINO — for state-of-the-art fracture detection accuracy.",
                },
                {
                  icon: ShieldCheck,
                  title: "Explainable AI",
                  desc: "Grad-CAM technology visualizing exactly which regions of the X-ray influenced the diagnosis.",
                },
                {
                  icon: Stethoscope,
                  title: "Medical Knowledge",
                  desc: "RAG-powered system retrieving verified medical guidelines and treatment protocols.",
                },
                {
                  icon: Lightbulb,
                  title: "Patient Education",
                  desc: "Automatically translates complex medical jargon into easy-to-understand summaries.",
                },
                {
                  icon: MessageSquare,
                  title: "Interactive Chat",
                  desc: "Ask follow-up questions to an AI assistant grounded in your specific diagnosis context.",
                },
                {
                  icon: Activity,
                  title: "Fracture Classification",
                  desc: "Detects 7 distinct fracture types including Spiral, Oblique, and Comminuted fractures.",
                },
              ].map((feature, i) => (
                <div
                  key={i}
                  className="group p-6 rounded-2xl border border-white/5 bg-white/5 hover:bg-white/10 transition-all hover:-translate-y-1"
                >
                  <div className="h-12 w-12 rounded-lg bg-blue-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <feature.icon className="h-6 w-6 text-blue-400" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {feature.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section id="how-it-works" className="py-24 px-6 lg:px-12">
          <div className="max-w-6xl mx-auto grid md:grid-cols-2 gap-16 items-center">
            <div className="space-y-8">
              <h2 className="text-3xl md:text-4xl font-bold">How It Works</h2>
              <div className="space-y-6">
                {[
                  "Upload your X-ray image (JPEG/PNG) to the secure interface.",
                  "Our Ensemble Agent runs the image through multiple neural networks.",
                  "The Explainability Agent generates a heatmap showing the fracture location.",
                  "The Knowledge Agent retrieves relevant medical guidelines.",
                  "You receive a comprehensive report with diagnosis, confidence, and next steps.",
                ].map((step, i) => (
                  <div key={i} className="flex gap-4">
                    <div className="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center text-sm font-bold flex-shrink-0">
                      {i + 1}
                    </div>
                    <p className="text-lg text-muted-foreground">{step}</p>
                  </div>
                ))}
              </div>
            </div>
            <div className="relative h-[400px] rounded-2xl overflow-hidden border border-white/10 bg-white/5 flex items-center justify-center">
              {/* Abstract visualization or placeholder */}
              <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/20 via-transparent to-purple-500/20" />
              <div className="text-center p-8">
                <Activity className="h-16 w-16 mx-auto text-blue-400 mb-4 animate-bounce" />
                <p className="text-xl font-medium">
                  Processing Pipeline Visualization
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="py-8 border-t border-white/10 text-center text-muted-foreground text-sm">
        <p>© 2026 Anonymized. Created by a research team.</p>
      </footer>
    </div>
  );
}
