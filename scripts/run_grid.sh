#!/usr/bin/env bash
set -e

export MLFLOW_TRACKING_URI="http://localhost:5000"

runs=(
  "3 320 yolo_e3_320"
  "3 416 yolo_e3_416"
  "5 320 yolo_e5_320"
  "5 416 yolo_e5_416"
)

for r in "${runs[@]}"; do
  set -- $r
  epochs=$1
  imgsz=$2
  name=$3
  echo "[run_grid] epochs=$epochs imgsz=$imgsz exp_name=$name"
  python -m src.train_cv --epochs "$epochs" --imgsz "$imgsz" --exp-name "$name"
done
