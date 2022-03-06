#!/bin/bash

# Initialize AWS credentials
chmod +x ./bashrc_install.sh && ./bashrc_install.sh

# Create datasets, train and evaluate the model on startup
python src/core/data/create_datasets.py
python src/core/ml/train_model.py
python src/core/ml/evaluate_model.py

# Start the FastAPI service
uvicorn --host 0.0.0.0 --port 8080 --log-level info --factory apis:create_app
