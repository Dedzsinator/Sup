#!/bin/bash

# Production server start script for spam detection microservice

set -e

echo "Starting Spam Detection Microservice..."

# Check if model exists, if not train one
if [ ! -d "/app/models" ] || [ -z "$(ls -A /app/models)" ]; then
    echo "No trained model found. Training a new model..."
    python train.py --generate-synthetic --synthetic-samples 5000 --log-level INFO
fi

# Start the server
echo "Starting FastAPI server..."
exec uvicorn src.simple_server:app \
    --host ${HOST:-0.0.0.0} \
    --port ${PORT:-8080} \
    --workers ${WORKERS:-1} \
    --log-level ${LOG_LEVEL:-info} \
    --access-log
