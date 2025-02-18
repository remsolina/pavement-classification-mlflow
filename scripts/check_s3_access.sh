#!/bin/bash

echo "Checking S3 access..."
aws s3 ls s3://mlflow-pavement-classification/

if [ $? -eq 0 ]; then
    echo "✅ S3 access is working!"
else
    echo "❌ S3 access failed! Check IAM role permissions."
fi
