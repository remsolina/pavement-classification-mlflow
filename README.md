# MLflow Local Pipeline: Quick Start & Development Guide

This project provides a fully containerized MLflow experiment tracking and training pipeline, orchestrated by Prefect. The training code has been converted from a Jupyter notebook (`model_7_final.ipynb`) to a Python script (`model_7_local.py`) for containerized execution. All major components (MLflow server, MySQL backend, training jobs) run in Docker containers for easy reproducibility and onboarding.

---

## 1. Prerequisites

- **Docker** and **Docker Compose** must be installed on your host machine.
  - [Install Docker](https://docs.docker.com/get-docker/)
- **Prefect** (v2.x recommended) must be installed on your host for orchestration:
  ```bash
  pip install prefect
  ```
  Or see [Prefect installation docs](https://docs.prefect.io/latest/getting-started/installation/)

---

## 2. Project Structure

```
local/
├── config/              # Configuration files (env, compose, Dockerfiles)
├── scripts/             # Prefect flow scripts (e.g., prefect_flow.py)
├── src/                 # Source code and requirements.txt
├── tests/               # Test files (unit/ and integration/)
├── Makefile             # Local development make commands
```

---

## 3. Setup & Running Everything

### 3.1. Clone the repository

```bash
git clone <repository-url>
cd pavement-classification-mlflow/local
```

### 3.2. Configure your data path and training parameters

**⚠️ IMPORTANT: You MUST update the workspace paths before running the pipeline!**

The configuration files contain hardcoded paths that need to be updated for your system:

1. **Edit `local/config/config_local.py`** and update the `LOCAL_DATA_PATH`:
   ```python
   # Change this line (around line 18):
   LOCAL_DATA_PATH = os.getenv("DATA_PATH", "/Users/remioyediji/CapstoneProject/finaldata")
   
   # To your actual data path, for example:
   LOCAL_DATA_PATH = os.getenv("DATA_PATH", "/path/to/your/dataset")
   ```

2. **Edit `local/config/docker-config.env`** and update **both** paths:
   ```bash
   # Change this line (around line 47):
   TRAINING_DATA_PATH=/Users/remioyediji/CapstoneProject/finaldata
   
   # To your actual data path:
   TRAINING_DATA_PATH=/path/to/your/dataset
   
   # AND change this line (around line 75):
   WORKSPACE_PATH=/Users/remioyediji/CapstoneProject/pavement-classification-mlflow
   
   # To your actual project root path:
   WORKSPACE_PATH=/path/to/your/pavement-classification-mlflow
   ```

3. **Alternative**: Set environment variables:
   ```bash
   export DATA_PATH="/path/to/your/dataset"
   export WORKSPACE_PATH="/path/to/your/pavement-classification-mlflow"
   ```

**Training Data Structure**: Your dataset must follow this directory structure:
```
TRAINING_DATA_PATH/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image3.jpg
│   ├── image4.jpg
│   └── ...
└── ...
```

**Training Parameters**: You can modify training hyperparameters in `config/config_local.py`:
- `NUM_EPOCHS = 2` (Should be 30, currently set to 2 for quick testing on CPU)
- `LEARNING_RATE = 0.001`
- `BATCH_SIZE = 32`
- `TRAIN_RATIO = 0.70`, `VAL_RATIO = 0.15`, `TEST_RATIO = 0.15`

### 3.3. Start Prefect Server

**Important:** Before running any flows, you must start the Prefect server:

```bash
# Start Prefect server (this starts the UI and API)
prefect server start

# Keep this running in a separate terminal
# The UI will be available at: http://localhost:4200
```

### 3.4. Run the Complete Pipeline

**Everything is orchestrated by Prefect!** The Prefect flow will automatically:

1. **Build the training Docker image**
2. **Start the MLflow infrastructure** (MySQL and MLflow server)
3. **Run unit tests** in the training container
4. **Execute training** and log results to MLflow

To run the complete pipeline (in a new terminal):

```bash
# From the project root directory
python local/scripts/prefect_flow.py
# or
prefect run -p local/scripts/prefect_flow.py
```

- MLflow UI: [http://localhost:5005](http://localhost:5005)
- Prefect UI: [http://localhost:4200](http://localhost:4200)

**Note:** You don't need to manually start Docker Compose or build images - Prefect handles everything!

**Configuration Changes**: If you modify training parameters (epochs, learning rate, etc.) in `config/config_local.py`, the Prefect flow will automatically rebuild the training image with your changes. No manual rebuilding required!

---

### 3.5. Alternative: Using Makefile Commands

If you prefer simpler commands, you can use the provided Makefile:

```bash
# Navigate to the local directory
cd local

# See all available commands
make help

# Run the complete pipeline
make train

# Check system status
make status

# Run health checks
make health-check

# Build training image only
make docker-build

# Start MLflow services only
make docker-up

# Clean everything
make clean-all
```

**Makefile Benefits:**
- **Simpler commands** - No need to remember long Docker/Prefect commands
- **Consistent workflow** - Same commands for everyone on the team
- **Built-in help** - Run `make help` to see all options
- **Health checks** - Easy system status verification

---

## 4. What the Prefect Flow Does

The `ml_pipeline` flow performs these tasks automatically:

1. **Build Training Image**: Creates the `config-training:latest` Docker image
2. **Start Infrastructure**: Launches MySQL and MLflow server containers
3. **Run Unit Tests**: Executes pytest in the training container
4. **Execute Training**: Runs the ML training script (`model_7_local.py`)
5. **Optional Cleanup**: Removes containers and volumes (if `do_cleanup=True`)

---

## 5. Manual Operations (Optional)

If you need to run components manually:

### Start MLflow Infrastructure Only
```bash
docker compose --env-file config/docker-config.env -f config/docker-compose.local.yml up -d
```

### Build Training Image Only
```bash
docker compose --env-file config/docker-config.env -f config/docker-compose.local.yml build training
```

### Run with Cleanup
```bash
python local/scripts/prefect_flow.py --do-cleanup
```

---

## 6. Monitoring and Logs

- **Prefect UI**: Monitor flow runs at [http://localhost:4200](http://localhost:4200) (after starting the server in step 3.3)
- **MLflow UI**: View experiment results at [http://localhost:5005](http://localhost:5005)
- **Flow Logs**: Check flow execution logs in the terminal output
- **Container Logs**:  
  ```bash
  docker compose -f config/docker-compose.local.yml logs
  ```

---

## 7. Troubleshooting

- **Data path not found**: 
  ```bash
  # Error: "Data path does not exist: /Users/remioyediji/CapstoneProject/finaldata"
  # Solution: Update LOCAL_DATA_PATH in local/config/config_local.py to your actual data path
  ```
- **Docker build context errors**: 
  ```bash
  # Error: "failed to compute cache key: failed to calculate checksum"
  # Solution: Update WORKSPACE_PATH in local/config/docker-config.env to your actual project root
  ```
- **Training data not found in container**: 
  ```bash
  # Error: "No such file or directory" when accessing training data
  # Solution: Update TRAINING_DATA_PATH in local/config/docker-config.env to your actual data path
  ```
- **Prefect server not running**: Make sure to start `prefect server start` before running flows
- **Port conflicts**: If ports 5005 (MLflow), 3308 (MySQL), or 4200 (Prefect) are in use, stop other services or change ports
- **Database connection**: Ensure MySQL is running and credentials are correct
- **Permission errors**:  
  ```bash
  chmod -R 777 mlflow-artifacts mlruns
  ```
- **Prefect version conflicts**:  
  ```bash
  pip install --upgrade prefect
  ```
- **Training image issues**: The Prefect flow automatically rebuilds the image when you run it, but you can manually rebuild:
  ```bash
  docker compose --env-file config/docker-config.env -f config/docker-compose.local.yml build training
  ```
- **For production training**: Increase `NUM_EPOCHS` in `config/config_local.py` from 2 to 30 (recommended) or higher for better model performance

---

## 8. Architecture Overview

- **MLflow Server**: Tracks experiments, stores artifacts (uses MySQL for backend).
- **Training Pipeline**: Data processing, model training/evaluation/registration (PyTorch, scikit-learn, pandas).
- **Orchestration**: All training and testing is run via Prefect flows, which manage Docker containers.
- **Docker Compose**: Orchestrates all services locally except the training container, which is started by Prefect.

---

## 9. Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Documentation](https://docs.docker.com/)
- [Prefect Documentation](https://docs.prefect.io/)

---

**Everything runs in containers, orchestrated by Prefect. No need for a Python virtual environment on the host.** 