# 🐳 Containerized MLflow Training

This setup runs both MLflow server and training scripts in Docker containers, eliminating artifact logging issues and providing a consistent environment.

## 🚀 Quick Start

### 1. Start MLflow Server
```bash
./start_mlflow_local.sh
```

### 2. Run Containerized Training
```bash
./run_training.sh train
```

### 3. View Results
Open http://localhost:5005 to see your experiments and artifacts!

## 📁 File Structure

```
local/
├── Dockerfile.training          # Training container definition
├── docker-compose.local.yml     # Updated with training service
├── requirements.txt             # Python dependencies
├── config_containerized.py      # Container-specific config
├── model_7_containerized.py     # Containerized training script
├── run_training.sh             # Training orchestration script
└── README_CONTAINERIZED.md     # This file
```

## 🛠️ Available Commands

### Training Commands
```bash
# Run default training
./run_training.sh train

# Run specific script
./run_training.sh train model_7_containerized.py

# Just build the training image
./run_training.sh build

# Clean up containers and images
./run_training.sh cleanup

# Show help
./run_training.sh help
```

### MLflow Server Commands
```bash
# Start MLflow server
./start_mlflow_local.sh

# Stop MLflow server (preserves data)
./stop_mlflow_local.sh

# Clean all data (destructive!)
./clean_all_data.sh
```

## 🔧 How It Works

### Container Architecture
1. **MLflow Server Container**: Runs MLflow UI and API
2. **MySQL Container**: Stores experiment metadata
3. **Training Container**: Runs your training scripts

### Network Communication
- All containers run in the same Docker network
- Training container connects to MLflow server via `http://mlflow-server-local:5000`
- Artifacts are shared via mounted volumes

### Volume Mapping
- `../mlflow-artifacts/` → Container artifacts storage
- `../data/` → Shared data directory
- `mysql_data_local` → MySQL database persistence

## ✅ Benefits

### ✅ **Artifact Logging Fixed**
- No more "Read-only file system" errors
- All containers share the same file system access
- Proper artifact server integration

### ✅ **Consistent Environment**
- Same Python/library versions everywhere
- No dependency conflicts with host system
- Reproducible across different machines

### ✅ **Easy Deployment**
- Everything runs with one command
- No need to install Python packages on host
- Self-contained and portable

### ✅ **Data Persistence**
- Experiments and artifacts preserved between runs
- MySQL database persists in Docker volume
- Easy backup and restore

## 🧪 Testing the Setup

### 1. Verify MLflow Server
```bash
curl http://localhost:5005
```

### 2. Run Test Training
```bash
./run_training.sh train
```

### 3. Check Artifacts
- Go to http://localhost:5005
- Navigate to your experiment
- Verify artifacts are logged correctly

## 🔍 Troubleshooting

### Training Container Won't Start
```bash
# Check if MLflow server is running
curl http://localhost:5005

# Rebuild training image
./run_training.sh build

# Check Docker logs
docker-compose -f docker-compose.local.yml logs training
```

### Artifact Logging Issues
```bash
# Check volume mounts
docker-compose -f docker-compose.local.yml config

# Verify network connectivity
docker-compose -f docker-compose.local.yml exec training ping mlflow-server-local
```

### Clean Start
```bash
# Stop everything
./stop_mlflow_local.sh

# Clean up training containers
./run_training.sh cleanup

# Start fresh
./start_mlflow_local.sh
./run_training.sh train
```

## 🎯 Next Steps

1. **Customize Training**: Modify `model_7_containerized.py` for your needs
2. **Add More Models**: Create additional training scripts
3. **GPU Support**: Update Dockerfile for CUDA if needed
4. **Production**: Use this as base for production deployments

## 📊 Expected Results

After running training, you should see:
- ✅ Experiment logged in MLflow UI
- ✅ Training curves artifact uploaded
- ✅ Model artifacts saved
- ✅ All metrics tracked properly
- ✅ No file system errors!
