#!/usr/bin/env python3
"""
main.py

Secure main entry point for the pavement classification model.
Implements proper error handling, logging, and security practices.
"""

import os
import sys
import logging
import torch
import mlflow
import random
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config, validate_environment
from data_loader import DataLoaderFactory
from model import ModelFactory, get_device
from trainer import SecureTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")


def setup_mlflow() -> None:
    """Setup MLflow configuration."""
    config = get_config()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    # Enable autologging with security considerations
    mlflow.pytorch.autolog(
        log_models=False,  # We'll log manually for better control
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False
    )
    
    logger.info(f"MLflow configured - URI: {config.mlflow.tracking_uri}")
    logger.info(f"Experiment: {config.mlflow.experiment_name}")


def validate_system_requirements() -> bool:
    """Validate system requirements and dependencies."""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        # Check PyTorch installation
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.info("CUDA not available, using CPU")
        
        # Check disk space (basic check)
        disk_usage = os.statvfs('.')
        free_space_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
        if free_space_gb < 5.0:  # Require at least 5GB free space
            logger.warning(f"Low disk space: {free_space_gb:.1f} GB available")
        
        return True
        
    except Exception as e:
        logger.error(f"System validation failed: {e}")
        return False


def main() -> None:
    """Main training function."""
    logger.info("=" * 60)
    logger.info("Starting Secure Pavement Classification Training")
    logger.info("=" * 60)
    
    try:
        # Validate environment and system
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        if not validate_system_requirements():
            logger.error("System requirements validation failed")
            sys.exit(1)
        
        # Load and validate configuration
        config = get_config()
        if not config.validate_all():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # Set random seeds for reproducibility
        set_random_seeds(config.model.random_seed)
        
        # Setup MLflow
        setup_mlflow()
        
        # Get device
        device = get_device()
        
        # Create data loaders
        logger.info("Creating data loaders...")
        data_factory = DataLoaderFactory()
        train_loader, val_loader, test_loader, class_names = data_factory.create_data_loaders()
        
        # Calculate class weights for imbalanced data
        class_weights = data_factory.calculate_class_weights(train_loader)
        
        # Create model and training components
        logger.info("Creating model and training components...")
        model_factory = ModelFactory()
        model = model_factory.create_model(device)
        criterion = model_factory.create_loss_function(class_weights, device)
        optimizer = model_factory.create_optimizer(model)
        scheduler = model_factory.create_scheduler(optimizer)
        
        # Create trainer
        trainer = SecureTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            class_names=class_names,
            scheduler=scheduler
        )
        
        # Start training
        logger.info("Starting training process...")
        results = trainer.train()
        
        # Log final results
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
        logger.info(f"Best epoch: {results['best_epoch'] + 1}")
        logger.info(f"Test accuracy: {results['test_results']['accuracy']:.4f}")
        logger.info("=" * 60)
        
        # Print MLflow run info
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"MLflow run ID: {active_run.info.run_id}")
            logger.info(f"MLflow run URL: {config.mlflow.tracking_uri}/#/experiments/{active_run.info.experiment_id}/runs/{active_run.info.run_id}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
