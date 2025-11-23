from zenml import pipeline
from src.zenml_steps.data_steps import prepare_tiny_coco_dataset
from src.zenml_steps.train_steps import train_yolo_tiny
from src.zenml_steps.eval_steps import summarize_yolo_experiment


@pipeline
def yolo_training_pipeline(
    epochs: int = 3,
    imgsz: int = 320,
    exp_name: str = "zenml_yolo_tiny",
):
    """Pipeline ZenML complet pour YOLO tiny.

    Steps :
    1) Préparer / vérifier le dataset tiny_coco (DVC).
    2) Entraîner YOLO tiny (logs MLflow).
    3) Afficher un résumé et pointer vers MLflow.
    """
    prepare_tiny_coco_dataset()
    exp = train_yolo_tiny(epochs=epochs, imgsz=imgsz, exp_name=exp_name)
    summarize_yolo_experiment(exp)
