# 🚀 Quick Start Guide

## For New Users Testing This Project

### 📋 Prerequisites
- **Docker & Docker Compose** (for MLflow server)
- **Python 3.8+** (for model training)
- **Image Dataset** organized in folders by class

### 🗂️ Prepare Your Data
Organize your pavement images like this:
```
your_data_folder/
├── asphalt/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── chip-sealed/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── gravel/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### ⚡ 5-Minute Setup

```bash
# 1. Clone and navigate
git clone <this-repo>
cd mlflow-server/local/

# 2. Run setup (installs Python dependencies)
./setup.sh

# 3. Activate Python environment
source venv/bin/activate

# 4. Test setup
python test_setup.py

# 5. Update data path
# Edit config_local.py and change LOCAL_DATA_PATH to your data folder

# 6. Start MLflow server (takes ~30 seconds)
./start_mlflow_local.sh

# 7. Train the model
python model_7_local.py

# 8. View results
# Open http://localhost:5005 in your browser
```

### 🛑 When Done
```bash
# Stop MLflow server
./stop_mlflow_local.sh

# Deactivate Python environment
deactivate
```

### 🔧 Troubleshooting

**Docker not found?**
- Install Docker Desktop from https://docs.docker.com/get-docker/

**Python import errors?**
- Make sure you activated the virtual environment: `source venv/bin/activate`
- Re-run setup: `./setup.sh`

**MLflow UI not loading?**
- Wait 30-60 seconds for containers to start
- Check containers: `docker-compose -f docker-compose.local.yml ps`

**Data not found?**
- Update `LOCAL_DATA_PATH` in `config_local.py` to point to your image folder
- Make sure images are organized in class subfolders

### 📊 What You'll See

1. **Terminal**: Training progress with epoch metrics
2. **MLflow UI** (http://localhost:5005): 
   - Real-time training curves
   - Model parameters and metrics
   - Confusion matrix and classification report
   - Saved model artifacts

### 🎯 Expected Results

- **Training**: ~30 epochs, ~5-10 minutes (depending on dataset size)
- **Accuracy**: Varies based on your dataset quality
- **Artifacts**: Plots, model files, and reports saved automatically

### 📁 What Gets Created

```
mlflow-server/
├── mlflow-artifacts/     # Model files, plots, reports
├── mlruns/              # Experiment metadata
└── local/venv/          # Python virtual environment
```

All data is stored locally - no cloud dependencies required!

### 🚀 Next Steps

- Experiment with different parameters in `config_local.py`
- Try the production version in `production/` for AWS deployment
- Check `README.md` for detailed documentation

Happy training! 🎉
