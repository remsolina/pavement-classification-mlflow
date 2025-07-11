# MLflow Server Setup

This repository contains MLflow server setups for the Pavement Classification project, organized into separate local and production environments.

## 📁 Project Structure

```
mlflow-server/
├── local/                    # Local development setup
│   ├── model_7_local.py     # Local model training script
│   ├── config_local.py      # Local configuration
│   ├── README_local_model.md # Local setup documentation
│   ├── start_simple_mlflow.sh # Simple MLflow server start
│   ├── stop_simple_mlflow.sh  # Simple MLflow server stop
│   ├── docker-compose.local.yml # Docker local setup (optional)
│   └── Dockerfile.local     # Local Docker image (optional)
├── production/              # Production/AWS setup
│   ├── docker-compose.yml   # Production Docker setup
│   ├── Dockerfile           # Production Docker image
│   └── scripts/             # Production scripts
│       ├── setup_server.sh  # EC2 setup automation
│       ├── start_mlflow.sh  # Start production MLflow
│       └── check_s3_access.sh # S3 connectivity check
├── mlflow-artifacts/        # Local artifact storage (created)
├── mlruns/                  # Local runs storage (created)
└── README.md               # This file
```

## 🚀 Quick Start

### 🏠 Local Development (Recommended for Development)

**Simple Setup** (No Docker required):
```bash
cd local/
./start_simple_mlflow.sh

# Run your model training
python model_7_local.py

# Access MLflow UI at http://localhost:5005
```

**Models stored in**: `../mlflow-artifacts/` directory

### ☁️ Production/AWS Setup

For production deployment with S3 storage:
```bash
cd production/
./scripts/setup_server.sh
./scripts/start_mlflow.sh

# Access at http://<EC2_PUBLIC_IP>:5000
```

**Models stored in**: S3 bucket (`s3://mlflow-pavement-classification/`)

## 🔧 Configuration

### Local Development
- **Configuration**: `local/config_local.py`
- **MLflow URI**: `http://localhost:5005`
- **Backend Store**: File-based (`./mlruns/`)
- **Artifact Store**: Local directory (`./mlflow-artifacts/`)

### Production
- **MLflow URI**: `http://<EC2_PUBLIC_IP>:5000`
- **Backend Store**: MySQL database
- **Artifact Store**: S3 bucket

## 📊 Features

### Local Setup Benefits
- ✅ No AWS costs
- ✅ No internet required
- ✅ Fast artifact storage
- ✅ Easy debugging
- ✅ Perfect for development
- ✅ No Docker complications

### Production Setup Benefits
- ✅ Scalable MySQL backend
- ✅ S3 artifact storage
- ✅ Production-ready
- ✅ Team collaboration
- ✅ Persistent storage

## 🛠️ Switching Between Setups

### From Local to Production
1. Train and validate your model locally
2. Update data paths and MLflow URI for production
3. Deploy to EC2 using production setup

### From Production to Local
1. Download model artifacts from S3 if needed
2. Use local setup for development and debugging

## 📚 Documentation

- **Local Setup**: See `local/README_local_model.md`
- **Production Setup**: See `production/` directory
- **Model Training**: Use `local/model_7_local.py` for local development

## 🎯 Recommended Workflow

1. **Development**: Use local setup for model development and testing
2. **Validation**: Test with small datasets locally
3. **Production**: Deploy to AWS for full-scale training and team collaboration

Choose the setup that best fits your current needs!
