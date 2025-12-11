"""
Training Pipeline for MedAI.

Provides training loop, configuration, and utilities for model training.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from medai.config import DEVICE, TRAINING_DEFAULTS, NUM_CLASSES, OUTPUTS_DIR
from medai.training.metrics import calculate_metrics, MetricsTracker, compute_class_weights

__all__ = ["Trainer", "TrainingConfig", "mixup_data", "mixup_criterion"]


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Model
    model_name: str = "hypercolumn_densenet169"
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    
    # Training
    epochs: int = TRAINING_DEFAULTS["epochs"]
    batch_size: int = TRAINING_DEFAULTS["batch_size"]
    learning_rate: float = TRAINING_DEFAULTS["learning_rate"]
    weight_decay: float = TRAINING_DEFAULTS["weight_decay"]
    num_workers: int = TRAINING_DEFAULTS["num_workers"]
    
    # Augmentation
    use_mixup: bool = True
    mixup_alpha: float = TRAINING_DEFAULTS["mixup_alpha"]
    label_smoothing: float = TRAINING_DEFAULTS["label_smoothing"]
    
    # Scheduler
    scheduler: str = TRAINING_DEFAULTS["scheduler"]
    warmup_epochs: int = TRAINING_DEFAULTS["warmup_epochs"]
    
    # Early stopping
    early_stopping_patience: int = TRAINING_DEFAULTS["early_stopping_patience"]
    early_stopping_metric: str = "val_accuracy"
    
    # Output
    output_dir: Path = field(default_factory=lambda: OUTPUTS_DIR)
    save_best: bool = True
    save_last: bool = True
    
    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "medai-fracture"
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: torch.device = DEVICE,
) -> tuple:
    """
    Apply mixup augmentation to batch.
    
    Args:
        x: Input images (B, C, H, W).
        y: Labels (B,).
        alpha: Mixup alpha parameter.
        device: Compute device.
        
    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer:
    """
    Training pipeline for fracture classification models.
    
    Supports:
    - Mixup augmentation
    - Label smoothing
    - Learning rate scheduling
    - Early stopping
    - Best model checkpointing
    - Metrics tracking
    
    Args:
        model: The neural network model.
        config: Training configuration.
        class_names: List of class names.
        device: Compute device.
        
    Examples:
        >>> model = get_model('swin', num_classes=8)
        >>> trainer = Trainer(model, TrainingConfig(epochs=20))
        >>> trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or TrainingConfig()
        self.device = device or DEVICE
        self.model = model.to(self.device)
        self.class_names = class_names
        
        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metrics_tracker = MetricsTracker(class_names)
        
        # State
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
    
    def setup(self, train_loader: DataLoader) -> None:
        """Setup optimizer, scheduler, and criterion."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        if self.config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6,
            )
        elif self.config.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1,
            )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing,
        )
    
    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(loader, desc=f"Train Epoch {self.current_epoch}")
        
        for imgs, labels, _ in pbar:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Apply mixup if enabled
            if self.config.use_mixup:
                imgs, targets_a, targets_b, lam = mixup_data(
                    imgs, labels, self.config.mixup_alpha, self.device
                )
                outputs = self.model(imgs)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            
            preds = outputs.softmax(dim=1).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            
            pbar.set_postfix({"loss": loss.item()})
        
        # Calculate metrics
        epoch_loss = running_loss / len(loader.dataset)
        metrics = calculate_metrics(all_targets, all_preds)
        metrics["loss"] = epoch_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        for imgs, labels, _ in tqdm(loader, desc="Validating"):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * imgs.size(0)
            
            probs = outputs.softmax(dim=1)
            preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(loader.dataset)
        all_probs = np.vstack(all_probs)
        
        metrics = calculate_metrics(
            all_targets, all_preds, all_probs, self.class_names
        )
        metrics["loss"] = epoch_loss
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Run full training pipeline.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Optional test data loader for final evaluation.
            
        Returns:
            Dictionary with training results and best metrics.
        """
        self.setup(train_loader)
        
        print(f"Training on {self.device}")
        print(f"Model: {self.config.model_name}")
        print(f"Epochs: {self.config.epochs}")
        print("-" * 50)
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_one_epoch(train_loader)
            self.metrics_tracker.update("train", epoch, **train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.metrics_tracker.update("val", epoch, **val_metrics)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Print progress
            print(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Check for improvement
            current_metric = val_metrics["accuracy"]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                
                if self.config.save_best:
                    self._save_checkpoint("best.pth")
                    print(f"  ✅ New best model saved! (Acc: {self.best_metric:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        if self.config.save_last:
            self._save_checkpoint("last.pth")
        
        # Final test evaluation
        test_metrics = None
        if test_loader:
            print("\nEvaluating on test set...")
            test_metrics = self.validate(test_loader)
            self.metrics_tracker.update("test", 0, **test_metrics)
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        return {
            "best_val_accuracy": self.best_metric,
            "epochs_trained": self.current_epoch,
            "test_metrics": test_metrics,
            "history": self.metrics_tracker.summary(),
        }
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.config.output_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }, path)


def main():
    """CLI entry point for training."""
    import argparse
    
    from medai.models.factory import get_model
    from medai.utils.data import FractureDataset, create_dataloader
    from medai.utils.transforms import get_transforms
    from medai.config import CLASS_NAMES
    
    parser = argparse.ArgumentParser(description="Train fracture classification model")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv")
    parser.add_argument("--img-root", default=".")
    parser.add_argument("--model", default="hypercolumn_densenet169")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="outputs")
    
    args = parser.parse_args()
    
    # Create datasets
    train_ds = FractureDataset(
        args.train_csv,
        img_root=args.img_root,
        transform=get_transforms("train"),
    )
    val_ds = FractureDataset(
        args.val_csv,
        img_root=args.img_root,
        transform=get_transforms("val"),
    )
    
    # Create data loaders
    train_loader = create_dataloader(train_ds, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, args.batch_size)
    
    test_loader = None
    if args.test_csv:
        test_ds = FractureDataset(
            args.test_csv,
            img_root=args.img_root,
            transform=get_transforms("val"),
        )
        test_loader = create_dataloader(test_ds, args.batch_size)
    
    # Create model and trainer
    model = get_model(args.model, NUM_CLASSES, pretrained=True)
    
    config = TrainingConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=Path(args.output_dir),
    )
    
    trainer = Trainer(model, config, CLASS_NAMES)
    results = trainer.fit(train_loader, val_loader, test_loader)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
