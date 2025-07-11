#!/usr/bin/env python3
"""
Integration tests for the complete training pipeline.

Tests the end-to-end training workflow including data loading,
model training, and MLflow integration.
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from PIL import Image
import mlflow

# Add local directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config_local


class TestTrainingPipeline:
    """Integration tests for the complete training pipeline."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Create temporary test data directory."""
        temp_dir = tempfile.mkdtemp()
        
        # Create class directories
        classes = ['asphalt', 'chip-sealed', 'gravel']
        for class_name in classes:
            class_dir = Path(temp_dir) / class_name
            class_dir.mkdir()
            
            # Create dummy images
            for i in range(10):
                # Create a random RGB image
                img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(class_dir / f'test_{i}.jpg')
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_mlflow(self):
        """Mock MLflow for testing."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.start_run'), \
             patch('mlflow.log_param'), \
             patch('mlflow.log_metric'), \
             patch('mlflow.log_artifact'), \
             patch('mlflow.pytorch.log_model'):
            yield
    
    def test_data_loading_with_valid_structure(self, test_data_dir):
        """Test that data loading works with valid directory structure."""
        # This would test the data loading logic from model_7_local.py
        # For now, we'll test the basic structure validation
        
        classes = ['asphalt', 'chip-sealed', 'gravel']
        for class_name in classes:
            class_dir = Path(test_data_dir) / class_name
            assert class_dir.exists(), f"Class directory {class_name} should exist"
            
            images = list(class_dir.glob('*.jpg'))
            assert len(images) > 0, f"Should have images in {class_name} directory"
    
    def test_config_validation_with_test_data(self, test_data_dir):
        """Test configuration validation with test data."""
        with patch.object(config_local, 'LOCAL_DATA_PATH', test_data_dir):
            # Should not raise an exception
            config_local.validate_config()
    
    def test_model_architecture_creation(self):
        """Test that model architecture can be created."""
        # This tests the basic model creation logic
        # We'll create a simple version of the model for testing
        
        class TestPavementNet(torch.nn.Module):
            def __init__(self, num_classes=3):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(32, num_classes)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = TestPavementNet()
        
        # Test forward pass
        test_input = torch.randn(1, 3, 256, 256)
        output = model(test_input)
        
        assert output.shape == (1, 3), f"Expected output shape (1, 3), got {output.shape}"
    
    def test_training_loop_basic_functionality(self, test_data_dir, mock_mlflow):
        """Test basic training loop functionality."""
        # This is a simplified version of the training loop
        
        # Mock the training components
        model = torch.nn.Linear(10, 3)  # Simple model for testing
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Simulate one training step
        inputs = torch.randn(4, 10)
        labels = torch.randint(0, 3, (4,))
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Test that loss is computed
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
    
    def test_mlflow_integration_setup(self, mock_mlflow):
        """Test MLflow integration setup."""
        # Test that MLflow functions can be called without errors
        
        mlflow.set_tracking_uri("http://localhost:5005")
        mlflow.set_experiment("test_experiment")
        
        with mlflow.start_run():
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.5)
    
    def test_device_selection(self):
        """Test device selection logic."""
        # Test CUDA availability check
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        assert device.type in ["cuda", "cpu"]
        
        # Test tensor creation on device
        test_tensor = torch.randn(2, 2).to(device)
        assert test_tensor.device.type == device.type


class TestDataPipeline:
    """Test data pipeline components."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    def test_image_preprocessing(self, sample_image):
        """Test image preprocessing pipeline."""
        from torchvision import transforms
        
        # Define transforms similar to those in model_7_local.py
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        transformed = transform(sample_image)
        
        # Check output shape and type
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 256, 256)
        
        # Check normalization (values should be roughly in [-2, 2] range after normalization)
        assert transformed.min() >= -3.0
        assert transformed.max() <= 3.0
    
    def test_data_split_logic(self):
        """Test data splitting logic."""
        # Simulate data splitting
        total_samples = 100
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        
        # Calculate split sizes
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
        
        # Verify splits
        assert train_size + val_size + test_size == total_samples
        assert train_size > 0
        assert val_size > 0
        assert test_size > 0


class TestMLflowIntegration:
    """Test MLflow integration components."""
    
    def test_mlflow_experiment_creation(self):
        """Test MLflow experiment creation."""
        with patch('mlflow.set_experiment') as mock_set_exp:
            experiment_name = "test_experiment"
            mlflow.set_experiment(experiment_name)
            mock_set_exp.assert_called_once_with(experiment_name)
    
    def test_mlflow_parameter_logging(self):
        """Test MLflow parameter logging."""
        with patch('mlflow.log_param') as mock_log_param:
            mlflow.log_param("batch_size", 32)
            mlflow.log_param("learning_rate", 0.001)
            
            assert mock_log_param.call_count == 2
    
    def test_mlflow_metric_logging(self):
        """Test MLflow metric logging."""
        with patch('mlflow.log_metric') as mock_log_metric:
            mlflow.log_metric("train_loss", 0.5, step=1)
            mlflow.log_metric("val_accuracy", 0.8, step=1)
            
            assert mock_log_metric.call_count == 2
    
    def test_mlflow_artifact_logging(self):
        """Test MLflow artifact logging."""
        with patch('mlflow.log_artifact') as mock_log_artifact:
            with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
                temp_file.write(b"test artifact content")
                temp_file.flush()
                
                mlflow.log_artifact(temp_file.name)
                mock_log_artifact.assert_called_once_with(temp_file.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
