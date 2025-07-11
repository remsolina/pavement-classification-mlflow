#!/usr/bin/env python3
"""
data_loader.py

Secure and modular data loading functionality for S3-based image datasets.
Implements proper error handling, logging, and security practices.
"""

import boto3
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
import logging
from collections import defaultdict, Counter
import random
import numpy as np
from botocore.exceptions import ClientError, NoCredentialsError

from config import get_config

# Configure logging
logger = logging.getLogger(__name__)


class SecureS3ImageDataset(Dataset):
    """
    Secure S3-based image dataset with proper error handling and logging.
    
    Expected S3 structure:
        s3://<bucket>/<prefix>/<class_name>/image.jpg
    """
    
    def __init__(self, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.
        
        Args:
            transform: Optional image transformations to apply
        """
        super().__init__()
        self.config = get_config()
        self.transform = transform
        
        # Initialize S3 client with error handling
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.config.s3.access_key_id,
                aws_secret_access_key=self.config.s3.secret_access_key,
                region_name=self.config.s3.region
            )
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
        
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        # Load dataset metadata
        self._load_dataset_metadata()
    
    def _load_dataset_metadata(self) -> None:
        """Load dataset metadata from S3 with proper error handling."""
        logger.info(f"Loading dataset from S3: s3://{self.config.s3.bucket_name}/{self.config.s3.prefix}")
        
        try:
            # Use paginator for large datasets
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.config.s3.bucket_name,
                Prefix=self.config.s3.prefix.rstrip('/')
            )
            
            class_samples = defaultdict(list)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Check if this is an image file
                    if self._is_image_file(key):
                        class_name = self._extract_class_name(key)
                        if class_name:
                            class_samples[class_name].append(key)
            
            # Sort classes for consistency
            self.classes = sorted(class_samples.keys())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            # Create samples list
            for class_name, keys in class_samples.items():
                class_idx = self.class_to_idx[class_name]
                for key in keys:
                    self.samples.append((key, class_idx))
            
            logger.info(f"Found {len(self.classes)} classes: {self.classes}")
            logger.info(f"Total images: {len(self.samples)}")
            
            # Log class distribution
            self._log_class_distribution(class_samples)
            
        except ClientError as e:
            logger.error(f"AWS S3 error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset metadata: {e}")
            raise
    
    def _is_image_file(self, key: str) -> bool:
        """Check if the S3 key points to an image file."""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        return key.lower().endswith(image_extensions)
    
    def _extract_class_name(self, key: str) -> Optional[str]:
        """Extract class name from S3 key."""
        parts = key.split('/')
        if len(parts) >= 2:
            return parts[-2]  # Class name is the parent directory
        return None
    
    def _log_class_distribution(self, class_samples: Dict[str, List[str]]) -> None:
        """Log the distribution of samples across classes."""
        logger.info("Class distribution:")
        for class_name in sorted(class_samples.keys()):
            count = len(class_samples[class_name])
            logger.info(f"  {class_name}: {count} images")
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        s3_key, label = self.samples[idx]
        
        try:
            # Download image from S3
            response = self.s3_client.get_object(
                Bucket=self.config.s3.bucket_name,
                Key=s3_key
            )
            image_bytes = response['Body'].read()
            
            # Load image
            image = Image.open(BytesIO(image_bytes))
            
            # Ensure RGB mode for consistency
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except ClientError as e:
            logger.error(f"Error downloading image {s3_key}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing image {s3_key}: {e}")
            raise


class DataLoaderFactory:
    """Factory class for creating data loaders with proper splits."""
    
    def __init__(self):
        self.config = get_config()
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L")),  # Force grayscale
            transforms.Resize((280, 280)),                    # Resize to 280×280
            transforms.RandomCrop(256),                       # Random crop to 256×256
            transforms.RandomHorizontalFlip(),                # Random horizontal flip
            transforms.RandomRotation(10),                    # Random rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),                            # (1, 256, 256)
            transforms.Normalize((0.5,), (0.5,))              # Normalize
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L")),  # Force grayscale
            transforms.Resize((256, 256)),                    # Resize to 256×256
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
        """
        Create train, validation, and test data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader, class_names)
        """
        logger.info("Creating data loaders...")
        
        # Set random seed for reproducibility
        random.seed(self.config.model.random_seed)
        
        # Load full dataset without transforms for splitting
        full_dataset = SecureS3ImageDataset(transform=None)
        
        # Create stratified splits
        train_indices, val_indices, test_indices = self._create_stratified_splits(full_dataset)
        
        # Create datasets with appropriate transforms
        train_dataset = self._create_subset(full_dataset, train_indices, self.train_transform)
        val_dataset = self._create_subset(full_dataset, val_indices, self.test_transform)
        test_dataset = self._create_subset(full_dataset, test_indices, self.test_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Created data loaders - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, full_dataset.classes
    
    def _create_stratified_splits(self, dataset: SecureS3ImageDataset) -> Tuple[List[int], List[int], List[int]]:
        """Create stratified train/val/test splits."""
        logger.info("Creating stratified data splits...")
        
        # Group indices by class
        label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            label_to_indices[label].append(idx)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Split each class proportionally
        for label, indices in label_to_indices.items():
            random.shuffle(indices)
            n = len(indices)
            
            train_count = int(self.config.model.train_ratio * n)
            val_count = int(self.config.model.val_ratio * n)
            
            train_indices.extend(indices[:train_count])
            val_indices.extend(indices[train_count:train_count + val_count])
            test_indices.extend(indices[train_count + val_count:])
        
        # Shuffle the final splits
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)
        
        # Log split information
        self._log_split_distribution(train_indices, val_indices, test_indices, dataset)
        
        return train_indices, val_indices, test_indices
    
    def _create_subset(self, dataset: SecureS3ImageDataset, indices: List[int], 
                      transform: transforms.Compose) -> SecureS3ImageDataset:
        """Create a subset of the dataset with specific indices and transform."""
        subset = SecureS3ImageDataset(transform=transform)
        subset.classes = dataset.classes
        subset.class_to_idx = dataset.class_to_idx
        subset.samples = [dataset.samples[i] for i in indices]
        return subset
    
    def _log_split_distribution(self, train_indices: List[int], val_indices: List[int], 
                               test_indices: List[int], dataset: SecureS3ImageDataset) -> None:
        """Log the distribution of samples in each split."""
        def log_distribution(indices: List[int], split_name: str):
            labels = [dataset.samples[i][1] for i in indices]
            distribution = Counter(labels)
            logger.info(f"{split_name} distribution:")
            for label, count in distribution.items():
                class_name = dataset.classes[label]
                logger.info(f"  {class_name}: {count}")
        
        log_distribution(train_indices, "Train")
        log_distribution(val_indices, "Validation")
        log_distribution(test_indices, "Test")
    
    def calculate_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data."""
        logger.info("Calculating class weights for imbalanced data...")
        
        # Count samples per class in training set
        class_counts = torch.zeros(self.config.model.num_classes)
        
        for _, labels in train_loader:
            for label in labels:
                class_counts[label] += 1
        
        # Calculate inverse frequency weights
        weights = 1.0 / class_counts
        weights = weights / weights.sum()  # Normalize
        
        logger.info(f"Class weights: {weights}")
        return weights
