@echo off
set MLFLOW_TRACKING_URI=http://localhost:5000

echo [run_grid] epochs=3 imgsz=320 exp_name=yolo_e3_320
python -m src.train_cv --epochs 3 --imgsz 320 --exp-name yolo_e3_320

echo [run_grid] epochs=3 imgsz=416 exp_name=yolo_e3_416
python -m src.train_cv --epochs 3 --imgsz 416 --exp-name yolo_e3_416

echo [run_grid] epochs=5 imgsz=320 exp_name=yolo_e5_320
python -m src.train_cv --epochs 5 --imgsz 320 --exp-name yolo_e5_320

echo [run_grid] epochs=5 imgsz=416 exp_name=yolo_e5_416
python -m src.train_cv --epochs 5 --imgsz 416 --exp-name yolo_e5_416
