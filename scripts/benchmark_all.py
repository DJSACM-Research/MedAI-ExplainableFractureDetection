import os
import subprocess
import re
import pandas as pd
import sys

# Configuration
PYTHON_EXEC = sys.executable
SCRIPT_PATH = "src/analysis/analyze.py"
CHECKPOINTS_DIR = "outputs"
DATA_ROOT = "data"
TEST_CSV = "data/balanced_augmented_dataset/test.csv"
CLASS_NAMES = "Comminuted,Greenstick,Healthy,Oblique,Oblique Displaced,Spiral,Transverse,Transverse Displaced"
OUTPUT_BASE = "outputs/analysis"
RESULTS_FILE = "RESULTS.md"

# Map model names (for script) to checkpoint files
MODELS = {
    "Swin Transformer": ("swin", "best_swin.pth"),
    "HyperColumn-DenseNet (Old)": ("hypercolumn_densenet169", "best_hypercolumn_densenet169_old.pth"),
    "HyperColumn-DenseNet (Focal)": ("hypercolumn_densenet169", "best_hypercolumn_cbam_densenet169_focal.pth"),
}

ENSEMBLE_MODELS = [
    ("swin", "best_swin.pth"),
    ("hypercolumn_densenet169", "best_hypercolumn_densenet169_old.pth"),
    ("hypercolumn_densenet169", "best_hypercolumn_cbam_densenet169_focal.pth")
]

def parse_output(output):
    metrics = {}
    lines = output.splitlines()
    
    # Regex for per-class metrics: "0 Comminuted: support=17, prec=0.895, rec=1.000, f1=0.944"
    class_regex = re.compile(r"\d+ ([\w\s]+): support=(\d+), prec=([\d\.]+), rec=([\d\.]+), f1=([\d\.]+)")
    
    # Regex for macro-f1: "macro-f1: 0.8523178210678211"
    macro_regex = re.compile(r"macro-f1: ([\d\.]+)")
    
    for line in lines:
        class_match = class_regex.search(line)
        if class_match:
            cls_name = class_match.group(1)
            metrics[cls_name] = {
                "precision": float(class_match.group(3)),
                "recall": float(class_match.group(4)),
                "f1": float(class_match.group(5))
            }
            
        macro_match = macro_regex.search(line)
        if macro_match:
            metrics["Macro F1"] = float(macro_match.group(1))
            
    return metrics

def main():
    all_results = {}
    failed_models = {}
    
    print(f"Starting Benchmark for {len(MODELS)} models...")
    
    for display_name, (model_arg, checkpoint_file) in MODELS.items():
        print(f"\n------------------------------------------------")
        print(f"Running analysis for: {display_name}")
        print(f"------------------------------------------------")
        
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            failed_models[display_name] = "Checkpoint file not found."
            continue
            
        out_dir = os.path.join(OUTPUT_BASE, model_arg)
        if "Focal" in display_name:
             out_dir = os.path.join(OUTPUT_BASE, "hypercolumn_densenet169_focal")
        elif "Old" in display_name:
             out_dir = os.path.join(OUTPUT_BASE, "hypercolumn_densenet169_old")
             
        os.makedirs(out_dir, exist_ok=True)
        
        cmd = [
            PYTHON_EXEC, SCRIPT_PATH,
            "--checkpoint", checkpoint_path,
            "--test-csv", TEST_CSV,
            "--img-root", DATA_ROOT,
            "--model", model_arg,
            "--class-names", CLASS_NAMES,
            "--out-dir", out_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            metrics = parse_output(result.stdout)
            metrics["checkpoint"] = checkpoint_file
            metrics["out_dir"] = out_dir
            all_results[display_name] = metrics
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running analysis for {display_name}")
            # Extract the last line of stderr as the error reason
            error_msg = e.stderr.strip().split('\n')[-1] if e.stderr else "Unknown error"
            if "RuntimeError" in e.stderr:
                 # Try to find the RuntimeError line
                 for line in e.stderr.split('\n'):
                     if "RuntimeError:" in line:
                         error_msg = line.strip()
                         break
            failed_models[display_name] = error_msg
            print(e.stderr)
            
    # --- Run Ensemble (All) ---
    print(f"\n------------------------------------------------")
    print(f"Running analysis for: Ensemble (All Models)")
    print(f"------------------------------------------------")
    
    ensemble_out_dir = os.path.join(OUTPUT_BASE, "ensemble_all")
    os.makedirs(ensemble_out_dir, exist_ok=True)
    
    ensemble_cmd = [PYTHON_EXEC, "src/analysis/ensemble_analyze.py"]
    for m_name, ck_file in ENSEMBLE_MODELS:
        ensemble_cmd.extend(["--model", m_name])
        ensemble_cmd.extend(["--checkpoint", os.path.join(CHECKPOINTS_DIR, ck_file)])
        
    ensemble_cmd.extend([
        "--test-csv", TEST_CSV,
        "--img-root", DATA_ROOT,
        "--class-names", CLASS_NAMES,
        "--out-dir", ensemble_out_dir
    ])
    
    try:
        result = subprocess.run(ensemble_cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        metrics = parse_output(result.stdout)
        metrics["checkpoint"] = "Ensemble (All)"
        metrics["out_dir"] = ensemble_out_dir
        all_results["Ensemble (All)"] = metrics
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running ensemble analysis")
        failed_models["Ensemble (All)"] = e.stderr.strip().split('\n')[-1] if e.stderr else "Unknown error"
        print(e.stderr)

    # --- Run Ensemble (HyperColumns Only) ---
    print(f"\n------------------------------------------------")
    print(f"Running analysis for: Ensemble (HyperColumns Only)")
    print(f"------------------------------------------------")
    
    hc_ensemble_out_dir = os.path.join(OUTPUT_BASE, "ensemble_hc")
    os.makedirs(hc_ensemble_out_dir, exist_ok=True)
    
    hc_ensemble_cmd = [PYTHON_EXEC, "src/analysis/ensemble_analyze.py"]
    # Only include the last two models from ENSEMBLE_MODELS (the HyperColumns)
    for m_name, ck_file in ENSEMBLE_MODELS[1:]:
        hc_ensemble_cmd.extend(["--model", m_name])
        hc_ensemble_cmd.extend(["--checkpoint", os.path.join(CHECKPOINTS_DIR, ck_file)])
        
    hc_ensemble_cmd.extend([
        "--test-csv", TEST_CSV,
        "--img-root", DATA_ROOT,
        "--class-names", CLASS_NAMES,
        "--out-dir", hc_ensemble_out_dir
    ])
    
    try:
        result = subprocess.run(hc_ensemble_cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        metrics = parse_output(result.stdout)
        metrics["checkpoint"] = "Ensemble (HC Only)"
        metrics["out_dir"] = hc_ensemble_out_dir
        all_results["Ensemble (HyperColumns Only)"] = metrics
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running ensemble analysis")
        failed_models["Ensemble (HC Only)"] = e.stderr.strip().split('\n')[-1] if e.stderr else "Unknown error"
        print(e.stderr)

    # Generate Markdown Report
    generate_report(all_results, failed_models)

def generate_report(results, failed_models):
    print(f"\nGenerating {RESULTS_FILE}...")
    
    with open(RESULTS_FILE, "w") as f:
        f.write("# Fracture Detection Model Benchmark Results\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        
        # 1. Summary Table
        f.write("## 1. Performance Summary\n\n")
        f.write("| Model | Macro F1 | Oblique F1 | Oblique Displaced F1 | Transverse F1 | Transverse Displaced F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        
        for name, metrics in results.items():
            macro = metrics.get("Macro F1", 0.0)
            obl = metrics.get("Oblique", {}).get("f1", 0.0)
            obl_disp = metrics.get("Oblique Displaced", {}).get("f1", 0.0)
            trans = metrics.get("Transverse", {}).get("f1", 0.0)
            trans_disp = metrics.get("Transverse Displaced", {}).get("f1", 0.0)
            
            f.write(f"| **{name}** | **{macro:.3f}** | {obl:.3f} | {obl_disp:.3f} | {trans:.3f} | {trans_disp:.3f} |\n")
            
        f.write("\n\n")
        
        # 2. Detailed Analysis per Model
        f.write("## 2. Detailed Analysis\n\n")
        
        for name, metrics in results.items():
            f.write(f"### {name}\n")
            f.write(f"- **Checkpoint**: `{metrics['checkpoint']}`\n")
            f.write(f"- **Macro F1**: {metrics.get('Macro F1', 0.0):.4f}\n\n")
            
            # Confusion Matrix Image
            cm_path = os.path.join(metrics['out_dir'], "confusion_matrix.png")
            f.write(f"![Confusion Matrix]({cm_path})\n\n")
            
            f.write("#### Per-Class Metrics\n")
            f.write("| Class | Precision | Recall | F1-Score |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            
            for cls_name in CLASS_NAMES.split(','):
                cls_name = cls_name.strip()
                if cls_name in metrics:
                    m = metrics[cls_name]
                    f.write(f"| {cls_name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |\n")
            f.write("\n---\n")

        # 3. Failed Models
        if failed_models:
            f.write("## 3. Failed Models\n\n")
            f.write("The following models failed to load or evaluate:\n\n")
            f.write("| Model | Error Reason |\n")
            f.write("| :--- | :--- |\n")
            for name, reason in failed_models.items():
                f.write(f"| {name} | `{reason}` |\n")

    print(f"✅ Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
