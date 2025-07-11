#!/usr/bin/env python3
"""
config_local.py

Configuration file for model_7_local.py
Modify these parameters as needed for your local setup.

For containerized deployment, also check docker-config.env for
Docker Compose configuration parameters (ports, volumes, etc.)
"""
import os

# ===================================================================
# DATA CONFIGURATION
# ===================================================================

# Local data path - automatically configured for container/local use
# For containerized training: Set TRAINING_DATA_PATH in docker-config.env
# For local training: Set this path directly or use DATA_PATH environment variable
# Expected structure: ./data/pavement_images/class_name/image.jpg
LOCAL_DATA_PATH = os.getenv("DATA_PATH", "/Users/remioyediji/CapstoneProject/finaldata")

# ===================================================================
# MLFLOW CONFIGURATION
# ===================================================================

# MLflow tracking server URI - automatically configured for container/local use
# For containerized training: Uses internal container network (set in docker-config.env)
# For local training: Uses localhost (set MLFLOW_HOST_PORT in docker-config.env)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server-local:5000")

# Experiment name in MLflow
EXPERIMENT_NAME = "Pytorch_CNN_from_Scratch_Pavement_Surface_Classification"

# Artifact root is handled by the MLflow server - no need to set it on client side
# ARTIFACT_ROOT = None  # Let MLflow server handle artifact storage

# ===================================================================
# TRAINING PARAMETERS
# ===================================================================

# Training hyperparameters - keep same as original for comparable results
BATCH_SIZE = 32
NUM_EPOCHS = 2  # Set to 2 for quick testing on CPU. For production, use 30 epochs
LEARNING_RATE = 0.001

# Data split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
SEED = 42

# ===================================================================
# MODEL CONFIGURATION
# ===================================================================

# Input image size (height, width)
INPUT_SIZE = (256, 256)

# Model architecture name for logging
ARCHITECTURE_NAME = "PavementNet (3xConv->Pool + AdaptivePool->FC)"

# ===================================================================
# VALIDATION
# ===================================================================

def validate_config():
    """Validate configuration parameters"""
    import os
    
    # Check data path exists
    if not os.path.exists(LOCAL_DATA_PATH):
        raise ValueError(f"Data path does not exist: {LOCAL_DATA_PATH}")
    
    # Check split ratios sum to 1.0
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}")
    
    # Check positive values
    if BATCH_SIZE <= 0:
        raise ValueError(f"Batch size must be positive, got {BATCH_SIZE}")
    
    if NUM_EPOCHS <= 0:
        raise ValueError(f"Number of epochs must be positive, got {NUM_EPOCHS}")
    
    if LEARNING_RATE <= 0:
        raise ValueError(f"Learning rate must be positive, got {LEARNING_RATE}")
    
    print("✅ Configuration validated successfully!")
    print(f"📁 Data path: {LOCAL_DATA_PATH}")
    print(f"🔗 MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"🎯 Batch size: {BATCH_SIZE}")
    print(f"📊 Epochs: {NUM_EPOCHS}")
    print(f"⚡ Learning rate: {LEARNING_RATE}")

if __name__ == "__main__":
    validate_config()
