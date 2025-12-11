"""
Model Benchmarking for MedAI.

Provides comprehensive model evaluation and comparison utilities.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from medai.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    MODELS_DIR,
    MODEL_CONFIGS,
    DEVICE,
)
from medai.models.factory import load_model_from_checkpoint
from medai.training.metrics import calculate_metrics
from medai.utils.data import FractureDataset, create_dataloader
from medai.utils.transforms import get_transforms

__all__ = [
    "benchmark_model",
    "benchmark_all_models",
    "BenchmarkResult",
    "generate_benchmark_report",
]


@dataclass
class BenchmarkResult:
    """Results from a model benchmark."""
    
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    inference_time_ms: float
    num_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix": self.confusion_matrix,
            "inference_time_ms": self.inference_time_ms,
            "num_samples": self.num_samples,
        }


@torch.no_grad()
def benchmark_model(
    model_name: str,
    test_loader: DataLoader,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
) -> BenchmarkResult:
    """
    Benchmark a single model on a test dataset.
    
    Args:
        model_name: Name of the model architecture.
        test_loader: DataLoader for test data.
        checkpoint_path: Path to model checkpoint. Auto-detected if None.
        device: Compute device.
        class_names: List of class names.
        
    Returns:
        BenchmarkResult with comprehensive metrics.
    """
    import time
    
    device = device or DEVICE
    class_names = class_names or CLASS_NAMES
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / f"best_{model_name}.pth"
        if not checkpoint_path.exists():
            # Try alternate naming
            for pattern in [f"best_{model_name}v2.pth", f"{model_name}.pth"]:
                alt_path = MODELS_DIR / pattern
                if alt_path.exists():
                    checkpoint_path = alt_path
                    break
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        num_classes=NUM_CLASSES,
        device=device,
    )
    model.eval()
    
    # Collect predictions
    all_preds = []
    all_targets = []
    all_probs = []
    total_time = 0.0
    
    for imgs, labels, _ in tqdm(test_loader, desc=f"Benchmarking {model_name}"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        start_time = time.perf_counter()
        outputs = model(imgs)
        end_time = time.perf_counter()
        
        total_time += (end_time - start_time)
        
        probs = outputs.softmax(dim=1)
        preds = probs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())
    
    # Calculate metrics
    all_probs = np.vstack(all_probs)
    metrics = calculate_metrics(all_targets, all_preds, all_probs, class_names)
    
    # Calculate average inference time per sample
    num_samples = len(all_preds)
    avg_inference_time_ms = (total_time / num_samples) * 1000
    
    return BenchmarkResult(
        model_name=model_name,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        per_class_metrics=metrics.get("per_class", {}),
        confusion_matrix=metrics["confusion_matrix"],
        inference_time_ms=avg_inference_time_ms,
        num_samples=num_samples,
    )


def benchmark_all_models(
    test_csv: str,
    img_root: str = ".",
    models: Optional[List[str]] = None,
    batch_size: int = 32,
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark all available models.
    
    Args:
        test_csv: Path to test CSV file.
        img_root: Root directory for images.
        models: List of model names. Uses all available if None.
        batch_size: Batch size for inference.
        
    Returns:
        Dictionary mapping model names to BenchmarkResults.
    """
    # Default to all configured models
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    
    # Create test dataset and loader
    test_ds = FractureDataset(
        test_csv,
        img_root=img_root,
        transform=get_transforms("val"),
    )
    test_loader = create_dataloader(test_ds, batch_size, shuffle=False)
    
    # Benchmark each model
    results = {}
    for model_name in models:
        try:
            result = benchmark_model(model_name, test_loader)
            results[model_name] = result
            print(
                f"✅ {model_name}: Acc={result.accuracy:.4f}, "
                f"F1={result.f1:.4f}, Time={result.inference_time_ms:.2f}ms"
            )
        except Exception as e:
            print(f"❌ {model_name}: Failed - {e}")
    
    return results


def generate_benchmark_report(
    results: Dict[str, BenchmarkResult],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a markdown benchmark report.
    
    Args:
        results: Dictionary of benchmark results.
        output_path: Path to save the report. Prints if None.
        
    Returns:
        Markdown report string.
    """
    lines = [
        "# MedAI Model Benchmark Results",
        "",
        "## Summary",
        "",
        "| Model | Accuracy | Precision | Recall | F1 Score | Inference (ms) |",
        "|-------|----------|-----------|--------|----------|----------------|",
    ]
    
    # Sort by accuracy descending
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].accuracy,
        reverse=True,
    )
    
    for name, result in sorted_results:
        lines.append(
            f"| {name} | {result.accuracy:.4f} | {result.precision:.4f} | "
            f"{result.recall:.4f} | {result.f1:.4f} | {result.inference_time_ms:.2f} |"
        )
    
    lines.extend([
        "",
        "## Per-Class Performance",
        "",
    ])
    
    # Add per-class breakdown for best model
    if sorted_results:
        best_name, best_result = sorted_results[0]
        lines.extend([
            f"### {best_name} (Best Model)",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|-------|-----------|--------|-----|---------|",
        ])
        
        for class_name, metrics in best_result.per_class_metrics.items():
            lines.append(
                f"| {class_name} | {metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['support']} |"
            )
    
    report = "\n".join(lines)
    
    if output_path:
        Path(output_path).write_text(report)
        print(f"Report saved to {output_path}")
    
    return report


def main():
    """CLI entry point for benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark fracture classification models")
    parser.add_argument("--test-csv", required=True, help="Path to test CSV")
    parser.add_argument("--img-root", default=".", help="Image root directory")
    parser.add_argument("--models", help="Comma-separated model names")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", help="Output report path")
    
    args = parser.parse_args()
    
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    
    results = benchmark_all_models(
        test_csv=args.test_csv,
        img_root=args.img_root,
        models=models,
        batch_size=args.batch_size,
    )
    
    report = generate_benchmark_report(results, args.output)
    
    if not args.output:
        print("\n" + report)


if __name__ == "__main__":
    main()
