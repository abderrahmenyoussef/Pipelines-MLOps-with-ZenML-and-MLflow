from .yolo_training_pipeline import yolo_training_pipeline


def main():
    """Lance un run baseline du pipeline YOLO tiny via ZenML."""
    pipe = yolo_training_pipeline(
        epochs=3,
        imgsz=320,
        exp_name="zenml_yolo_tiny_baseline",
    )
    pipe.run()


if __name__ == "__main__":
    main()
