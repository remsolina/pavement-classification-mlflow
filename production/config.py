#!/usr/bin/env python3
"""
config.py

Secure configuration management for the pavement classification model.
Uses environment variables and secure defaults.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture and training configuration."""
    # Model architecture
    input_channels: int = 1  # Grayscale
    input_size: tuple = (256, 256)
    num_classes: int = 3  # asphalt, chip-sealed, gravel
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 0.001
    
    # Data split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Random seed for reproducibility
    random_seed: int = 42


@dataclass
class S3Config:
    """S3 configuration with secure credential handling."""
    bucket_name: str
    prefix: str
    region: str = "us-west-2"
    
    @property
    def access_key_id(self) -> Optional[str]:
        """Get AWS access key from environment variable."""
        return os.getenv('AWS_ACCESS_KEY_ID')
    
    @property
    def secret_access_key(self) -> Optional[str]:
        """Get AWS secret key from environment variable."""
        return os.getenv('AWS_SECRET_ACCESS_KEY')
    
    @property
    def endpoint_url(self) -> str:
        """Get S3 endpoint URL."""
        return f"https://s3.{self.region}.amazonaws.com"
    
    def validate(self) -> bool:
        """Validate S3 configuration."""
        if not self.bucket_name:
            logger.error("S3 bucket name is required")
            return False
        
        if not self.access_key_id:
            logger.error("AWS_ACCESS_KEY_ID environment variable is required")
            return False
        
        if not self.secret_access_key:
            logger.error("AWS_SECRET_ACCESS_KEY environment variable is required")
            return False
        
        return True


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str
    experiment_name: str = "Pytorch_CNN_from_Scratch_Pavement_Surface_Classification"
    
    def validate(self) -> bool:
        """Validate MLflow configuration."""
        if not self.tracking_uri:
            logger.error("MLflow tracking URI is required")
            return False
        return True


class ConfigManager:
    """Centralized configuration manager with validation."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.s3 = self._load_s3_config()
        self.mlflow = self._load_mlflow_config()
    
    def _load_s3_config(self) -> S3Config:
        """Load S3 configuration from environment variables."""
        bucket_name = os.getenv('S3_BUCKET_NAME', 'myfinaldata')
        prefix = os.getenv('S3_PREFIX', 'finaldata')
        region = os.getenv('AWS_REGION', 'us-west-2')
        
        return S3Config(
            bucket_name=bucket_name,
            prefix=prefix,
            region=region
        )
    
    def _load_mlflow_config(self) -> MLflowConfig:
        """Load MLflow configuration from environment variables."""
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 
                                  'Pytorch_CNN_from_Scratch_Pavement_Surface_Classification')
        
        return MLflowConfig(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name
        )
    
    def validate_all(self) -> bool:
        """Validate all configurations."""
        logger.info("Validating configuration...")
        
        # Validate data split ratios
        total_ratio = self.model.train_ratio + self.model.val_ratio + self.model.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            logger.error(f"Data split ratios must sum to 1.0, got {total_ratio}")
            return False
        
        # Validate S3 config
        if not self.s3.validate():
            return False
        
        # Validate MLflow config
        if not self.mlflow.validate():
            return False
        
        logger.info("✅ All configurations validated successfully")
        return True
    
    def get_description(self) -> str:
        """Get model description for MLflow logging."""
        return (
            "Secure production version of pavement classification model. "
            "Loads images from S3 bucket with proper credential management, "
            "applies class weighting for imbalanced data, and uses CNN architecture "
            "with grayscale 256×256 images. Implements secure coding practices "
            "with environment-based configuration and comprehensive validation."
        )


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance."""
    return config


def validate_environment() -> bool:
    """Validate that all required environment variables are set."""
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'MLFLOW_TRACKING_URI'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set these variables before running the model")
        return False
    
    return True


if __name__ == "__main__":
    # Test configuration
    if validate_environment() and config.validate_all():
        print("✅ Configuration is valid and ready to use")
        print(f"📁 S3 Bucket: {config.s3.bucket_name}")
        print(f"📊 MLflow URI: {config.mlflow.tracking_uri}")
        print(f"🎯 Batch Size: {config.model.batch_size}")
        print(f"📈 Epochs: {config.model.num_epochs}")
    else:
        print("❌ Configuration validation failed")
        exit(1)
