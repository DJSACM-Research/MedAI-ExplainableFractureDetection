"""
Training Metrics for MedAI.

Provides functions for calculating and tracking classification metrics.
"""

from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
)

__all__ = [
    "calculate_metrics",
    "MetricsTracker",
    "compute_class_weights",
]


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = "macro",
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (optional, for AUC).
        class_names: List of class names for per-class metrics.
        average: Averaging method ('macro', 'micro', 'weighted').
        
    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
            - confusion_matrix: Confusion matrix array
            - per_class: Dict of per-class metrics (if class_names provided)
            - auc: ROC AUC (if y_prob provided)
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }
    
    # Per-class metrics
    if class_names:
        per_class = {}
        p_per, r_per, f1_per, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for i, name in enumerate(class_names):
            if i < len(p_per):
                per_class[name] = {
                    "precision": float(p_per[i]),
                    "recall": float(r_per[i]),
                    "f1": float(f1_per[i]),
                    "support": int(support[i]) if support[i] else 0,
                }
        
        results["per_class"] = per_class
    
    # ROC AUC (if probabilities provided)
    if y_prob is not None:
        try:
            # Multi-class AUC using one-vs-rest
            auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average=average
            )
            results["auc"] = float(auc)
        except Exception:
            results["auc"] = None
    
    return results


def compute_class_weights(
    labels: List[int],
    num_classes: int,
) -> np.ndarray:
    """
    Compute class weights for handling class imbalance.
    
    Uses inverse frequency weighting: weight = N / (C * N_c)
    where N is total samples, C is number of classes, N_c is class count.
    
    Args:
        labels: List of class labels.
        num_classes: Total number of classes.
        
    Returns:
        Array of class weights.
    """
    from collections import Counter
    
    counts = Counter(labels)
    total = len(labels)
    weights = []
    
    for i in range(num_classes):
        count = counts.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return np.array(weights, dtype=np.float32)


class MetricsTracker:
    """
    Tracks and aggregates training/validation metrics over epochs.
    
    Examples:
        >>> tracker = MetricsTracker()
        >>> tracker.update('train', loss=0.5, accuracy=0.8)
        >>> tracker.update('val', loss=0.4, accuracy=0.85)
        >>> print(tracker.get_best('val', 'accuracy'))
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names
        self.history: Dict[str, List[Dict[str, float]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.best_metrics: Dict[str, Dict[str, Tuple[float, int]]] = {
            "train": {},
            "val": {},
            "test": {},
        }
    
    def update(
        self,
        phase: str,
        epoch: Optional[int] = None,
        **metrics: float,
    ) -> None:
        """
        Record metrics for a training phase.
        
        Args:
            phase: 'train', 'val', or 'test'.
            epoch: Current epoch number.
            **metrics: Metric name-value pairs.
        """
        if epoch is None:
            epoch = len(self.history[phase])
        
        metrics["epoch"] = epoch
        self.history[phase].append(metrics)
        
        # Update best metrics
        for name, value in metrics.items():
            if name == "epoch":
                continue
            
            is_loss = "loss" in name.lower()
            current_best = self.best_metrics[phase].get(name)
            
            if current_best is None:
                self.best_metrics[phase][name] = (value, epoch)
            elif is_loss and value < current_best[0]:
                self.best_metrics[phase][name] = (value, epoch)
            elif not is_loss and value > current_best[0]:
                self.best_metrics[phase][name] = (value, epoch)
    
    def get_best(self, phase: str, metric: str) -> Tuple[float, int]:
        """Get best value and epoch for a metric."""
        return self.best_metrics.get(phase, {}).get(metric, (None, None))
    
    def get_last(self, phase: str, metric: str) -> Optional[float]:
        """Get most recent value for a metric."""
        if not self.history[phase]:
            return None
        return self.history[phase][-1].get(metric)
    
    def get_history(self, phase: str) -> List[Dict[str, float]]:
        """Get full history for a phase."""
        return self.history[phase]
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        return {
            "epochs_completed": len(self.history["train"]),
            "best_val_accuracy": self.get_best("val", "accuracy"),
            "best_val_f1": self.get_best("val", "f1"),
            "best_val_loss": self.get_best("val", "loss"),
            "final_train_loss": self.get_last("train", "loss"),
            "final_val_loss": self.get_last("val", "loss"),
        }
    
    def to_dataframe(self, phase: str = "val"):
        """Convert history to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.history[phase])
