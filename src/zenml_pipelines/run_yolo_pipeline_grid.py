from .yolo_training_pipeline import yolo_training_pipeline

GRID_CONFIGS = [
    {"epochs": 3, "imgsz": 320, "exp_name": "zenml_yolo_tiny_e3_320"},
    {"epochs": 3, "imgsz": 416, "exp_name": "zenml_yolo_tiny_e3_416"},
    {"epochs": 5, "imgsz": 320, "exp_name": "zenml_yolo_tiny_e5_320"},
    {"epochs": 5, "imgsz": 416, "exp_name": "zenml_yolo_tiny_e5_416"},
]


def main():
    """Lance une petite grille de runs de pipeline ZenML (TP5)."""
    for cfg in GRID_CONFIGS:
        print(f"[run_yolo_pipeline_grid] Lancement config : {cfg}")
        pipe = yolo_training_pipeline(
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            exp_name=cfg["exp_name"],
        )
        pipe.run()


if __name__ == "__main__":
    main()
