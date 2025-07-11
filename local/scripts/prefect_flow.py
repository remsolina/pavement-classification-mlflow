# Prefect flow for ML pipeline orchestration
#
# This flow automatically sets up the infrastructure, builds the training image,
# runs tests, and executes training. It ensures everything is properly configured
# before proceeding with ML operations.

from prefect import flow, task, get_run_logger
import subprocess
import os
import sys
import time

# Add the project root to Python path for imports
# Running from host (from local/scripts/ directory)
project_root = "../.."
sys.path.insert(0, project_root)

# Import after adding to path
try:
    from local.config.config_local import EXPERIMENT_NAME
except ImportError:
    # Fallback for running from project root
    sys.path.insert(0, ".")
    from local.config.config_local import EXPERIMENT_NAME

@task
def build_training_image():
    logger = get_run_logger()
    logger.info("Building training Docker image...")
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    compose_file = os.path.join(project_root, "local", "config", "docker-compose.local.yml")
    env_file = os.path.join(project_root, "local", "config", "docker-config.env")
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Using compose file: {compose_file}")
    logger.info(f"Using env file: {env_file}")
    
    result = subprocess.run([
        "docker", "compose",
        "--env-file", env_file,
        "-f", compose_file,
        "build", "training"
    ], capture_output=True, text=True, cwd=project_root)
    
    if result.returncode != 0:
        logger.error(f"Training image build failed: {result.stderr}")
        raise RuntimeError("Training image build failed")
    logger.info("Training image built successfully")

@task
def start_mlflow_infrastructure():
    logger = get_run_logger()
    logger.info("Starting MLflow infrastructure (MySQL and MLflow server)...")
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    compose_file = os.path.join(project_root, "local", "config", "docker-compose.local.yml")
    env_file = os.path.join(project_root, "local", "config", "docker-config.env")
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Using compose file: {compose_file}")
    logger.info(f"Using env file: {env_file}")
    
    # Start the infrastructure
    result = subprocess.run([
        "docker", "compose",
        "--env-file", env_file,
        "-f", compose_file,
        "up", "-d"
    ], capture_output=True, text=True, cwd=project_root)
    
    if result.returncode != 0:
        logger.error(f"Infrastructure startup failed: {result.stderr}")
        raise RuntimeError("Infrastructure startup failed")
    
    logger.info("Infrastructure started successfully")
    
    # Wait for services to be ready
    logger.info("Waiting for services to be ready...")
    time.sleep(10)  # Give services time to start
    
    # Check if MLflow server is responding
    max_retries = 30
    for i in range(max_retries):
        try:
            result = subprocess.run([
                "curl", "-f", "http://localhost:5005/health"
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("MLflow server is ready")
                break
        except subprocess.TimeoutExpired:
            pass
        
        if i == max_retries - 1:
            logger.error("MLflow server failed to start within expected time")
            raise RuntimeError("MLflow server not ready")
        
        logger.info(f"Waiting for MLflow server... (attempt {i+1}/{max_retries})")
        time.sleep(2)

@task
def run_unit_tests():
    logger = get_run_logger()
    logger.info("Running unit tests in the training image...")
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    compose_file = os.path.join(project_root, "local", "config", "docker-compose.local.yml")
    env_file = os.path.join(project_root, "local", "config", "docker-config.env")
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Using compose file: {compose_file}")
    logger.info(f"Using env file: {env_file}")
    
    result = subprocess.run([
      "docker", "compose",
       "--env-file", env_file,
      "-f", compose_file,
      "run", "--rm", "--no-deps",
      "-e", "PYTHONPATH=/app/src:/app",
      "training",
      "python", "-m", "pytest", "/app/tests", "-v"
    ], capture_output=True, text=True, cwd=project_root)
    
    if result.returncode != 0:
        logger.error(f"Unit tests failed: {result.stderr}")
        raise RuntimeError("Unit tests failed")
    logger.info(result.stdout)

@task
def run_training():
    logger = get_run_logger()
    logger.info("Running training in the training image...")
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    compose_file = os.path.join(project_root, "local", "config", "docker-compose.local.yml")
    env_file = os.path.join(project_root, "local", "config", "docker-config.env")
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Using compose file: {compose_file}")
    logger.info(f"Using env file: {env_file}")
    
    result = subprocess.run([
        "docker", "compose",
        "--env-file", env_file,
        "-f", compose_file,
        "run", "--rm", "--no-deps",
        "-e", f"MLFLOW_EXPERIMENT_NAME={EXPERIMENT_NAME}",
        "-e", "PYTHONPATH=/app/src:/app",
        "training",
        "python", "src/model_7_local.py"
    ], capture_output=True, text=True, cwd=project_root)
    
    if result.returncode != 0:
        logger.error(f"Training failed: {result.stderr}")
        raise RuntimeError("Training failed")
    logger.info(result.stdout)

@task
def cleanup():
    logger = get_run_logger()
    logger.info("Cleaning up Docker containers and volumes...")
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    compose_file = os.path.join(project_root, "local", "config", "docker-compose.local.yml")
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Using compose file: {compose_file}")
    
    result = subprocess.run([
        "docker", "compose", "-f", compose_file, "down", "--remove-orphans", "-v"
    ], capture_output=True, text=True, cwd=project_root)
    
    if result.returncode != 0:
        logger.error(f"Cleanup failed: {result.stderr}")
        raise RuntimeError("Cleanup failed")
    logger.info(result.stdout)

@flow(name="MLflow Training Pipeline")
def ml_pipeline(do_cleanup: bool = False):
    """
    Complete ML pipeline that sets up infrastructure, builds training image,
    runs tests, and executes training.
    
    Args:
        do_cleanup: If True, cleans up containers and volumes after completion
    """
    # Infrastructure setup
    build_training_image()
    start_mlflow_infrastructure()
    
    # ML operations
    run_unit_tests()
    run_training()
    
    # Optional cleanup
    if do_cleanup:
        cleanup()

if __name__ == "__main__":
    ml_pipeline(do_cleanup=False) 