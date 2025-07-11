# Model 7 Local Version

This is a local replica of `model_7_final.ipynb` that uses local disk storage for images and connects to a local MLflow environment. **All transformations and model architecture remain exactly the same** for comparable results.

## 🔄 Key Changes from Original

- **S3 → Local Storage**: Images loaded from local disk instead of S3
- **MLflow URI**: Points to local MLflow server (`http://localhost:5000`)
- **No AWS Dependencies**: Removed boto3 and AWS credentials
- **Easy Configuration**: Parameters can be modified in `config_local.py`

## 📁 Required Directory Structure

Your local image data should be organized as:

```
./data/pavement_images/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image3.jpg
│   ├── image4.jpg
│   └── ...
└── class3/
    ├── image5.jpg
    └── ...
```

## 🚀 Quick Start

### 1. Start Local MLflow Server

Make sure your local MLflow server is running:

```bash
./scripts/start_mlflow_local.sh
```

### 2. Configure Data Path

Edit `config_local.py` to point to your local image data:

```python
LOCAL_DATA_PATH = "./data/pavement_images"  # Change this path
```

### 3. Run Training

```bash
python model_7_local.py
```

## ⚙️ Configuration

All parameters can be easily modified in `config_local.py`:

- **Data path**: `LOCAL_DATA_PATH`
- **MLflow URI**: `MLFLOW_TRACKING_URI`
- **Training parameters**: `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`
- **Data splits**: `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`

## 🔒 Identical Components (DO NOT CHANGE)

These components are kept **exactly the same** as the original for comparable results:

### Transforms
```python
# Training transform
transforms.Compose([
    transforms.Lambda(lambda img: img.convert("L")),  # Force grayscale
    transforms.Resize((280, 280)),                    # Resize to 280×280
    transforms.RandomCrop(256),                       # Random crop to 256×256
    transforms.RandomHorizontalFlip(),                # Random horizontal flip
    transforms.RandomRotation(10),                    # Random rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Test transform
transforms.Compose([
    transforms.Lambda(lambda img: img.convert("L")),  # Force grayscale
    transforms.Resize((256, 256)),                    # Resize to 256×256
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

### Model Architecture (PavementNet)
- 3 Convolutional layers (32, 64, 128 channels)
- MaxPool2d after each conv layer
- AdaptiveAvgPool2d to (8, 8)
- 3 Fully connected layers (512, 256, num_classes)

### Training Setup
- Adam optimizer with lr=0.001
- Weighted CrossEntropyLoss (inverse frequency)
- 70/15/15 train/val/test split
- Same random seed (42) for reproducibility

## 📊 MLflow Logging

The script logs the same metrics and artifacts as the original:

- **Metrics**: train/val loss and accuracy per epoch, test accuracy, per-class precision/recall/f1
- **Parameters**: All hyperparameters and configuration
- **Artifacts**: Loss curves, accuracy curves, confusion matrix, classification report, trained model

## 🔍 Monitoring

Access your local MLflow UI at: http://localhost:5000

## 🐛 Troubleshooting

### Common Issues

1. **Data path not found**
   - Check that `LOCAL_DATA_PATH` in `config_local.py` points to the correct directory
   - Ensure the directory structure matches the expected format

2. **MLflow connection error**
   - Make sure local MLflow server is running: `./scripts/start_mlflow_local.sh`
   - Check that the server is accessible at `http://localhost:5000`

3. **Out of memory errors**
   - Reduce `BATCH_SIZE` in `config_local.py`
   - Use CPU instead of GPU if necessary

4. **Import errors**
   - Install required packages: `pip install torch torchvision mlflow scikit-learn seaborn matplotlib pillow`

### Validation

Run configuration validation:
```bash
python config_local.py
```

## 📈 Expected Results

Since all transformations and model architecture are identical to the original, you should expect:
- Similar training curves
- Comparable accuracy metrics
- Same model behavior and performance characteristics

The only differences should be in:
- Faster data loading (local vs S3)
- Different artifact storage location (local MLflow vs S3)
- No network dependencies
