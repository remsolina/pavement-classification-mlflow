FROM python:3.9

WORKDIR /app

# Install MLflow and dependencies
RUN pip install cryptography
RUN pip install --no-cache-dir mlflow boto3 pymysql

# Expose MLflow UI port
EXPOSE 5000

# Start MLflow Server
CMD mlflow server \
    --backend-store-uri mysql+pymysql://mlflow_user:mlflow_pass@mlflow-mysql/mlflow \
    --default-artifact-root s3://mlflow-pavement-classification/ \
    --host 0.0.0.0 --port 5000
