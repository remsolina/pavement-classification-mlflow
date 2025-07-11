#!/usr/bin/env python3
"""
model_7_local.py

Local replica of model_7_final.ipynb that uses local disk storage for images 
and connects to local MLflow environment. All transformations and model architecture 
remain exactly the same for comparable results.
"""

import os
import random
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np
import datetime
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from mlflow.tracking import MlflowClient

# Import local configuration
import config.config_local as config

# ===================================================================
# VALIDATE CONFIGURATION
# ===================================================================
config.validate_config()

# ---------------------------------------------------------------------------
# MLflow will use the server's configured artifact root automatically
# ---------------------------------------------------------------------------
# ===================================================================
# MLflow configuration - LOCAL ENVIRONMENT
# ===================================================================
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.EXPERIMENT_NAME)
mlflow.pytorch.autolog()

# ===================================================================
# Description for MLflow - Updated for local version
# ===================================================================
description = (
    "Local version of model_7 that loads images from local disk (instead of S3) "
    "and applies class weighting in the loss function to address class imbalance. "
    "Otherwise, it retains the EXACT same CNN architecture (PavementNet) and transformations "
    "of model_7. It processes grayscale images resized/cropped to 256×256 via data augmentation, "
    "with a 70/15/15 split, and logs metrics/artifacts to local MLflow."
)

# ===================================================================
# For reproducibility - SAME AS ORIGINAL
# ===================================================================
random.seed(config.SEED)
torch.manual_seed(config.SEED)

# ===================================================================
# LOCAL IMAGE DATASET - Replaces S3ImageDataset
# ===================================================================
class LocalImageDataset(Dataset):
    """
    Local version of S3ImageDataset.
    Expects a local directory structure of the form:
       <data_path>/<class_name>/image.jpg
    """
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.transform = transform

        self.samples = []
        self.classes = set()

        # Walk through all subdirectories (class folders)
        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                self.classes.add(class_name)
                
                # Find all image files in this class directory
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
                        self.samples.append((str(img_file), class_name))

        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Convert (file_path, class_name) => (file_path, class_index)
        self.samples = [(path, self.class_to_idx[c]) for (path, c) in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Load image from local disk
        image = Image.open(file_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ===================================================================
# EXACT SAME TRANSFORMS AS ORIGINAL - DO NOT CHANGE
# ===================================================================
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("L")),  # Force grayscale
    transforms.Resize((280, 280)),                    # Resize to 280×280
    transforms.RandomCrop(256),                       # Random crop to 256×256
    transforms.RandomHorizontalFlip(),                # Random horizontal flip
    transforms.RandomRotation(10),                    # Random rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),                            # (1, 256, 256)
    transforms.Normalize((0.5,), (0.5,))              # Normalize
])

test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("L")),  # Force grayscale
    transforms.Resize((256, 256)),                    # Resize to 256×256
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===================================================================
# Load the full dataset WITHOUT transform first, so we can split
# ===================================================================
print(f"Loading dataset from: {config.LOCAL_DATA_PATH}", flush=True)
full_dataset_no_transform = LocalImageDataset(
    data_path=config.LOCAL_DATA_PATH,
    transform=None
)
print("Classes found locally:", full_dataset_no_transform.classes, flush=True)
print("Total images found:", len(full_dataset_no_transform), flush=True)

# ===================================================================
# EXACT SAME 70/15/15 split across classes (stratified) AS ORIGINAL
# ===================================================================
label_to_indices = defaultdict(list)
for idx, (path, lbl_idx) in enumerate(full_dataset_no_transform.samples):
    label_to_indices[lbl_idx].append(idx)

train_indices = []
val_indices = []
test_indices = []

for lbl, indices in label_to_indices.items():
    random.shuffle(indices)
    n = len(indices)
    train_count = int(config.TRAIN_RATIO * n)
    val_count   = int(config.VAL_RATIO * n)
    # remainder -> test
    test_count  = n - train_count - val_count

    train_indices.extend(indices[:train_count])
    val_indices.extend(indices[train_count:train_count + val_count])
    test_indices.extend(indices[train_count + val_count:])

random.shuffle(train_indices)
random.shuffle(val_indices)
random.shuffle(test_indices)

# EXACT SAME helper function as original
def print_class_distribution(indices, dataset, subset_name):
    from collections import Counter
    labels = [dataset.samples[i][1] for i in indices]
    distribution = Counter(labels)
    print(f"{subset_name} distribution:", flush=True)
    for label, count in distribution.items():
        cls_name = dataset.classes[label]
        print(f"  {cls_name}: {count}", flush=True)
    print(flush=True)

print_class_distribution(train_indices, full_dataset_no_transform, "Train Set")
print_class_distribution(val_indices,  full_dataset_no_transform, "Validation Set")
print_class_distribution(test_indices,  full_dataset_no_transform, "Test Set")

# ===================================================================
# EXACT SAME subset creation function as original
# ===================================================================
def create_subset(dataset, indices, transform):
    # We'll create a copy that references only the subset's samples but with the new transform
    subset_ds = LocalImageDataset(dataset.data_path, transform=transform)
    subset_ds.classes       = dataset.classes
    subset_ds.class_to_idx  = dataset.class_to_idx
    # Filter only these indices
    subset_ds.samples       = [dataset.samples[i] for i in indices]
    return subset_ds

train_dataset = create_subset(full_dataset_no_transform, train_indices, train_transform)
val_dataset   = create_subset(full_dataset_no_transform, val_indices,   test_transform)
test_dataset  = create_subset(full_dataset_no_transform, test_indices,  test_transform)

print("Train dataset size:", len(train_dataset), flush=True)
print("Val dataset size:  ", len(val_dataset), flush=True)
print("Test dataset size: ", len(test_dataset), flush=True)

# ===================================================================
# EXACT SAME DataLoaders as original
# ===================================================================
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False)

# Quick shape check (same as original)
for images, labels in train_loader:
    print("Sample batch - images.shape:", images.shape, "labels.shape:", labels.shape)
    break

# ===================================================================
# EXACT SAME class weight calculation as original
# ===================================================================
train_labels = [full_dataset_no_transform.samples[i][1] for i in train_indices]
class_counts = np.bincount(train_labels)
print("Class sample counts (train subset):", class_counts, flush=True)

# Inverse frequency
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()  # optional normalization
print("Class weights (inverse-freq):", weights, flush=True)

# ===================================================================
# EXACT SAME CNN architecture from model_7 - DO NOT CHANGE
# ===================================================================
class PavementNet(nn.Module):
    def __init__(self):
        super(PavementNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # Input: (1,256,256) -> (32,256,256)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (64,256,256)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (128,256,256)

        self.pool = nn.MaxPool2d(2, 2)
        # After 3 x (conv+pool), 256 -> 128 -> 64 -> 32 in H/W
        # Then adaptive average pool to (8,8)
        self.adapt_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Fully connected layers
        # Flatten => 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(full_dataset_no_transform.classes))  # Dynamic number of classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.adapt_pool(x)    # (128,8,8)
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)           # logits
        return x

# ===================================================================
# EXACT SAME training setup as original
# ===================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PavementNet().to(device)

# Use weighted cross-entropy
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

train_losses = []
val_losses   = []
train_accuracies = []
val_accuracies   = []

# A distinctive run name
run_name = f"model_7_local_cnn_grayscale_classweight_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# ===================================================================
# EXACT SAME TRAINING LOOP AS ORIGINAL
# ===================================================================
with mlflow.start_run(run_name=run_name):
    # Log a description and hyperparams
    mlflow.set_tag("description", description)
    mlflow.log_param("num_epochs", config.NUM_EPOCHS)
    mlflow.log_param("batch_size", config.BATCH_SIZE)
    mlflow.log_param("learning_rate", config.LEARNING_RATE)
    mlflow.log_param("input_size", config.INPUT_SIZE)
    mlflow.log_param("architecture", config.ARCHITECTURE_NAME)
    mlflow.log_param("data_path", config.LOCAL_DATA_PATH)
    mlflow.log_param("class_weighting", "inverse_frequency")
    mlflow.log_param("train_ratio", config.TRAIN_RATIO)
    mlflow.log_param("val_ratio", config.VAL_RATIO)
    mlflow.log_param("test_ratio", config.TEST_RATIO)

    for epoch in range(config.NUM_EPOCHS):
        # ------------- TRAIN -------------
        model.train()
        running_loss = 0.0
        correct_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc  = correct_train / len(train_dataset)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # ------------- VALIDATION -------------
        model.eval()
        running_val_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc  = correct_val / len(val_dataset)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        # Logging
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}  "
              f"Train Loss: {epoch_train_loss:.4f}  Train Acc: {epoch_train_acc:.4f}  "
              f"Val Loss: {epoch_val_loss:.4f}  Val Acc: {epoch_val_acc:.4f}")

        mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

    # --------------------------------
    # Plot & log training curves
    plt.figure()
    plt.plot(range(1, config.NUM_EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, config.NUM_EPOCHS + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    loss_curve_path = "loss_curve.png"
    plt.savefig(loss_curve_path)
    mlflow.log_artifact(loss_curve_path)
    plt.close()

    plt.figure()
    plt.plot(range(1, config.NUM_EPOCHS + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, config.NUM_EPOCHS + 1), val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    acc_curve_path = "accuracy_curve.png"
    plt.savefig(acc_curve_path)
    mlflow.log_artifact(acc_curve_path)
    plt.close()

    # --------------------------------
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Classification report and confusion matrix
    class_names = full_dataset_no_transform.classes
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:\n", class_report, flush=True)

    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    mlflow.log_metric("test_accuracy", report_dict["accuracy"])

    # Per-class metrics
    for cls, metrics in report_dict.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"{cls}_precision", metrics.get("precision", 0))
            mlflow.log_metric(f"{cls}_recall", metrics.get("recall", 0))
            mlflow.log_metric(f"{cls}_f1-score", metrics.get("f1-score", 0))

    # Save & log classification report
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(class_report)
    mlflow.log_artifact(report_path)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # Log the final trained model to MLflow
    mlflow.pytorch.log_model(model, "model")

print(f"Training completed! Check your local MLflow server at http://localhost:5005", flush=True)
print(f"💡 Note: MLflow links above show internal URLs. Use http://localhost:5005 to access the UI from your host machine.", flush=True)
