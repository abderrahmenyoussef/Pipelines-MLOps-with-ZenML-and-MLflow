$env:MLFLOW_TRACKING_URI = "http://localhost:5000"

$runs = @(
    @{epochs=3; imgsz=320; name="yolo_e3_320"},
    @{epochs=3; imgsz=416; name="yolo_e3_416"},
    @{epochs=5; imgsz=320; name="yolo_e5_320"},
    @{epochs=5; imgsz=416; name="yolo_e5_416"}
)

foreach ($r in $runs) {
    Write-Host "[run_grid] epochs=$($r.epochs) imgsz=$($r.imgsz) exp_name=$($r.name)"
    python -m src.train_cv --epochs $r.epochs --imgsz $r.imgsz --exp-name $r.name
}
