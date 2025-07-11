#!/usr/bin/env python3
"""
trainer.py

Secure and modular training functionality with comprehensive logging and monitoring.
Implements proper error handling, checkpointing, and MLflow integration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Tuple, Dict, Any, Optional
import logging
import datetime
import os
from pathlib import Path

from config import get_config
from model import save_model_checkpoint

# Configure logging
logger = logging.getLogger(__name__)


class SecureTrainer:
    """
    Secure trainer class with comprehensive monitoring and error handling.
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, test_loader: DataLoader,
                 criterion: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device, class_names: List[str],
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            class_names: List of class names
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.class_names = class_names
        self.scheduler = scheduler
        self.config = get_config()
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info("Trainer initialized successfully")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            try:
                # Move data to device
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                # Log progress every 100 batches
                if batch_idx % 100 == 0:
                    logger.debug(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                               f"Loss: {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                raise
        
        avg_loss = running_loss / total_samples
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                try:
                    # Move data to device
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    # Statistics
                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    raise
        
        avg_loss = running_loss / total_samples
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop with MLflow logging.
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting training...")
        
        # Generate run name
        run_name = f"pavement_cnn_secure_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            try:
                # Log configuration and model info
                self._log_experiment_info()
                
                best_val_accuracy = 0.0
                best_epoch = 0
                
                for epoch in range(self.config.model.num_epochs):
                    logger.info(f"Epoch {epoch + 1}/{self.config.model.num_epochs}")
                    
                    # Training phase
                    train_loss, train_acc = self.train_epoch(epoch)
                    self.train_losses.append(train_loss)
                    self.train_accuracies.append(train_acc)
                    
                    # Validation phase
                    val_loss, val_acc = self.validate_epoch(epoch)
                    self.val_losses.append(val_loss)
                    self.val_accuracies.append(val_acc)
                    
                    # Update learning rate
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Log metrics
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                    
                    if self.scheduler:
                        mlflow.log_metric("learning_rate", self.scheduler.get_last_lr()[0], step=epoch)
                    
                    # Print progress
                    logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                    # Save best model
                    if val_acc > best_val_accuracy:
                        best_val_accuracy = val_acc
                        best_epoch = epoch
                        self._save_checkpoint(epoch, val_loss, "best_model.pth")
                    
                    # Save periodic checkpoint
                    if (epoch + 1) % 10 == 0:
                        self._save_checkpoint(epoch, val_loss, f"checkpoint_epoch_{epoch + 1}.pth")
                
                # Generate and log artifacts
                self._generate_training_plots()
                
                # Evaluate on test set
                test_results = self._evaluate_test_set()
                
                # Log final model
                mlflow.pytorch.log_model(self.model, "model")
                
                logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch + 1}")
                
                return {
                    "best_val_accuracy": best_val_accuracy,
                    "best_epoch": best_epoch,
                    "test_results": test_results,
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                    "train_accuracies": self.train_accuracies,
                    "val_accuracies": self.val_accuracies
                }
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise
    
    def _log_experiment_info(self) -> None:
        """Log experiment configuration and model info to MLflow."""
        # Log description
        mlflow.set_tag("description", self.config.get_description())
        
        # Log model configuration
        mlflow.log_param("num_epochs", self.config.model.num_epochs)
        mlflow.log_param("batch_size", self.config.model.batch_size)
        mlflow.log_param("learning_rate", self.config.model.learning_rate)
        mlflow.log_param("input_size", self.config.model.input_size)
        mlflow.log_param("num_classes", self.config.model.num_classes)
        mlflow.log_param("random_seed", self.config.model.random_seed)
        
        # Log data configuration
        mlflow.log_param("s3_bucket", self.config.s3.bucket_name)
        mlflow.log_param("s3_prefix", self.config.s3.prefix)
        mlflow.log_param("train_ratio", self.config.model.train_ratio)
        mlflow.log_param("val_ratio", self.config.model.val_ratio)
        mlflow.log_param("test_ratio", self.config.model.test_ratio)
        
        # Log model info
        if hasattr(self.model, 'get_model_info'):
            model_info = self.model.get_model_info()
            for key, value in model_info.items():
                mlflow.log_param(f"model_{key}", value)
        
        # Log device info
        mlflow.log_param("device", str(self.device))
        
        # Log class names
        mlflow.log_param("class_names", self.class_names)
    
    def _save_checkpoint(self, epoch: int, loss: float, filename: str) -> None:
        """Save model checkpoint."""
        filepath = self.checkpoint_dir / filename
        save_model_checkpoint(self.model, self.optimizer, epoch, loss, str(filepath))
    
    def _generate_training_plots(self) -> None:
        """Generate and log training plots."""
        # Loss curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Train Loss")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.train_accuracies) + 1), self.train_accuracies, label="Train Accuracy")
        plt.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        training_curves_path = "training_curves.png"
        plt.savefig(training_curves_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(training_curves_path)
        plt.close()
        
        # Clean up
        if os.path.exists(training_curves_path):
            os.remove(training_curves_path)
    
    def _evaluate_test_set(self) -> Dict[str, Any]:
        """Evaluate model on test set and generate reports."""
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Log test metrics
        mlflow.log_metric("test_accuracy", class_report["accuracy"])
        
        # Log per-class metrics
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                mlflow.log_metric(f"{class_name}_precision", metrics["precision"])
                mlflow.log_metric(f"{class_name}_recall", metrics["recall"])
                mlflow.log_metric(f"{class_name}_f1_score", metrics["f1-score"])
        
        # Save classification report
        report_text = classification_report(all_labels, all_predictions, target_names=self.class_names)
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_path)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # Clean up
        for file_path in [report_path, cm_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        logger.info(f"Test accuracy: {class_report['accuracy']:.4f}")
        
        return {
            "accuracy": class_report["accuracy"],
            "classification_report": class_report,
            "confusion_matrix": cm.tolist()
        }
