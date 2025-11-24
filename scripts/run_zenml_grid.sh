#!/usr/bin/env bash
set -e

#export MLFLOW_TRACKING_URI="http://localhost:5000"

python -m src.zenml_pipelines.run_yolo_pipeline_grid
