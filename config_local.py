#!/usr/bin/env python3
"""
config_local.py

Configuration file for model_7_local.py
Modify these parameters as needed for your local setup.
"""

# ===================================================================
# DATA CONFIGURATION
# ===================================================================

# Local data path - CHANGE THIS to point to your local image data
# Expected structure: ./data/pavement_images/class_name/image.jpg
LOCAL_DATA_PATH = "./data/pavement_images"

# ===================================================================
# MLFLOW CONFIGURATION
# ===================================================================

# MLflow tracking server URI - points to local MLflow server
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Experiment name in MLflow
EXPERIMENT_NAME = "Pytorch_CNN_from_Scratch_Pavement_Surface_Classification"

# ===================================================================
# TRAINING PARAMETERS
# ===================================================================

# Training hyperparameters - keep same as original for comparable results
BATCH_SIZE = 32
NUM_EPOCHS = 30  # Same as original model_7_final
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
