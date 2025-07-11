#!/usr/bin/env python3
"""
model.py

Secure and modular CNN model architecture for pavement surface classification.
Implements proper error handling, logging, and security practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

from config import get_config

# Configure logging
logger = logging.getLogger(__name__)


class PavementNet(nn.Module):
    """
    CNN architecture for pavement surface classification.
    
    Architecture:
    - 3 Convolutional layers with increasing channels (32, 64, 128)
    - MaxPooling after each conv layer
    - Adaptive average pooling
    - 3 Fully connected layers (512, 256, num_classes)
    - ReLU activations throughout
    """
    
    def __init__(self, num_classes: int = 3, input_channels: int = 1):
        """
        Initialize the PavementNet model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for grayscale)
        """
        super(PavementNet, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adapt_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized PavementNet with {num_classes} classes and {input_channels} input channels")
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Validate input shape
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.size(1)}")
        
        # Convolutional layers with pooling
        x = F.relu(self.conv1(x))  # (batch, 32, 256, 256)
        x = self.pool(x)           # (batch, 32, 128, 128)
        
        x = F.relu(self.conv2(x))  # (batch, 64, 128, 128)
        x = self.pool(x)           # (batch, 64, 64, 64)
        
        x = F.relu(self.conv3(x))  # (batch, 128, 64, 64)
        x = self.pool(x)           # (batch, 128, 32, 32)
        
        # Adaptive pooling
        x = self.adapt_pool(x)     # (batch, 128, 8, 8)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 128*8*8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)            # (batch, num_classes)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "architecture": "PavementNet",
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }


class ModelFactory:
    """Factory class for creating and configuring models."""
    
    def __init__(self):
        self.config = get_config()
    
    def create_model(self, device: torch.device) -> PavementNet:
        """
        Create and configure the model.
        
        Args:
            device: Device to move the model to
            
        Returns:
            Configured PavementNet model
        """
        logger.info("Creating PavementNet model...")
        
        model = PavementNet(
            num_classes=self.config.model.num_classes,
            input_channels=self.config.model.input_channels
        )
        
        # Move model to device
        model = model.to(device)
        
        # Log model information
        model_info = model.get_model_info()
        logger.info(f"Model created with {model_info['total_parameters']:,} parameters")
        logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
        
        return model
    
    def create_loss_function(self, class_weights: Optional[torch.Tensor] = None, 
                           device: torch.device = torch.device('cpu')) -> nn.CrossEntropyLoss:
        """
        Create loss function with optional class weighting.
        
        Args:
            class_weights: Optional class weights for handling imbalanced data
            device: Device to move weights to
            
        Returns:
            Configured loss function
        """
        if class_weights is not None:
            class_weights = class_weights.to(device)
            logger.info(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
        else:
            logger.info("Using standard CrossEntropyLoss")
        
        return nn.CrossEntropyLoss(weight=class_weights)
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Adam:
        """
        Create optimizer for the model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Configured optimizer
        """
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=1e-4  # L2 regularization
        )
        
        logger.info(f"Created Adam optimizer with lr={self.config.model.learning_rate}")
        return optimizer
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.StepLR:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            
        Returns:
            Configured scheduler
        """
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
        
        logger.info("Created StepLR scheduler (step_size=10, gamma=0.1)")
        return scheduler


def get_device() -> torch.device:
    """
    Get the best available device for training.
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device


def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                         epoch: int, loss: float, filepath: str) -> None:
    """
    Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def load_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                         filepath: str, device: torch.device) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to load to
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from {filepath}")
    logger.info(f"Checkpoint epoch: {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'model_info': checkpoint.get('model_info', {})
    }
