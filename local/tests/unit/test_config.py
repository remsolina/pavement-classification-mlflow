#!/usr/bin/env python3
"""
Unit tests for config_local.py

Tests configuration validation, parameter ranges, and environment variable handling.
"""

import pytest
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add local directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config_local


class TestConfigLocal:
    """Test cases for config_local.py"""
    
    def test_default_values(self):
        """Test that default configuration values are reasonable."""
        assert config_local.BATCH_SIZE > 0
        assert config_local.NUM_EPOCHS > 0
        assert config_local.LEARNING_RATE > 0
        assert config_local.TRAIN_RATIO + config_local.VAL_RATIO + config_local.TEST_RATIO == 1.0
        assert config_local.SEED >= 0
        
    def test_input_size_format(self):
        """Test that INPUT_SIZE is a valid tuple."""
        assert isinstance(config_local.INPUT_SIZE, tuple)
        assert len(config_local.INPUT_SIZE) == 2
        assert all(isinstance(x, int) and x > 0 for x in config_local.INPUT_SIZE)
        
    def test_split_ratios_sum_to_one(self):
        """Test that data split ratios sum to 1.0."""
        total = config_local.TRAIN_RATIO + config_local.VAL_RATIO + config_local.TEST_RATIO
        assert abs(total - 1.0) < 1e-6, f"Split ratios sum to {total}, not 1.0"
        
    def test_positive_hyperparameters(self):
        """Test that all hyperparameters are positive."""
        assert config_local.BATCH_SIZE > 0, "Batch size must be positive"
        assert config_local.NUM_EPOCHS > 0, "Number of epochs must be positive"
        assert config_local.LEARNING_RATE > 0, "Learning rate must be positive"
        
    def test_reasonable_hyperparameter_ranges(self):
        """Test that hyperparameters are in reasonable ranges."""
        assert 1 <= config_local.BATCH_SIZE <= 512, "Batch size should be reasonable"
        assert 1 <= config_local.NUM_EPOCHS <= 1000, "Epochs should be reasonable"
        assert 1e-6 <= config_local.LEARNING_RATE <= 1.0, "Learning rate should be reasonable"
        
    @patch.dict(os.environ, {'DATA_PATH': '/test/data/path'})
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        # Reload the module to pick up environment changes
        import importlib
        importlib.reload(config_local)
        
        assert config_local.LOCAL_DATA_PATH == '/test/data/path'
        
    @patch.dict(os.environ, {'MLFLOW_TRACKING_URI': 'http://test:1234'})
    def test_mlflow_uri_override(self):
        """Test that MLflow URI can be overridden."""
        import importlib
        importlib.reload(config_local)
        
        assert config_local.MLFLOW_TRACKING_URI == 'http://test:1234'
        
    def test_experiment_name_not_empty(self):
        """Test that experiment name is not empty."""
        assert config_local.EXPERIMENT_NAME
        assert len(config_local.EXPERIMENT_NAME.strip()) > 0
        
    def test_architecture_name_not_empty(self):
        """Test that architecture name is not empty."""
        assert config_local.ARCHITECTURE_NAME
        assert len(config_local.ARCHITECTURE_NAME.strip()) > 0


class TestConfigValidation:
    """Test cases for config validation function."""
    
    def test_validate_config_with_valid_data_path(self):
        """Test validation with valid data path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data structure
            for class_name in ['class1', 'class2']:
                class_dir = Path(temp_dir) / class_name
                class_dir.mkdir()
                # Create a dummy image file
                (class_dir / 'test.jpg').touch()
            
            with patch.object(config_local, 'LOCAL_DATA_PATH', temp_dir):
                # Should not raise an exception
                config_local.validate_config()
                
    def test_validate_config_with_invalid_data_path(self):
        """Test validation with invalid data path."""
        with patch.object(config_local, 'LOCAL_DATA_PATH', '/nonexistent/path'):
            with pytest.raises(ValueError, match="Data path does not exist"):
                config_local.validate_config()
                
    def test_validate_config_with_invalid_split_ratios(self):
        """Test validation with invalid split ratios."""
        with patch.object(config_local, 'TRAIN_RATIO', 0.5):
            with patch.object(config_local, 'VAL_RATIO', 0.3):
                with patch.object(config_local, 'TEST_RATIO', 0.3):  # Sum = 1.1
                    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
                        config_local.validate_config()
                        
    def test_validate_config_with_zero_batch_size(self):
        """Test validation with zero batch size."""
        with patch.object(config_local, 'BATCH_SIZE', 0):
            with pytest.raises(ValueError, match="Batch size must be positive"):
                config_local.validate_config()
                
    def test_validate_config_with_zero_epochs(self):
        """Test validation with zero epochs."""
        with patch.object(config_local, 'NUM_EPOCHS', 0):
            with pytest.raises(ValueError, match="Number of epochs must be positive"):
                config_local.validate_config()
                
    def test_validate_config_with_zero_learning_rate(self):
        """Test validation with zero learning rate."""
        with patch.object(config_local, 'LEARNING_RATE', 0):
            with pytest.raises(ValueError, match="Learning rate must be positive"):
                config_local.validate_config()


class TestConfigIntegration:
    """Integration tests for configuration."""
    
    def test_config_can_be_imported(self):
        """Test that config can be imported without errors."""
        import config_local
        assert hasattr(config_local, 'BATCH_SIZE')
        assert hasattr(config_local, 'LOCAL_DATA_PATH')
        assert hasattr(config_local, 'MLFLOW_TRACKING_URI')
        
    def test_config_validation_function_exists(self):
        """Test that validation function exists and is callable."""
        assert hasattr(config_local, 'validate_config')
        assert callable(config_local.validate_config)
        
    @patch('builtins.print')
    def test_validation_prints_success_message(self, mock_print):
        """Test that validation prints success message."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data structure
            (Path(temp_dir) / 'class1').mkdir()
            
            with patch.object(config_local, 'LOCAL_DATA_PATH', temp_dir):
                config_local.validate_config()
                
                # Check that success message was printed
                mock_print.assert_called()
                printed_args = [call.args[0] for call in mock_print.call_args_list]
                assert any("Configuration validated successfully" in arg for arg in printed_args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
