from __future__ import annotations

"""
Script de lancement du pipeline ZenML en mode "baseline".

Dans les versions récentes de ZenML, l'appel au pipeline
lance directement un run et retourne un PipelineRunResponse.
Il NE FAUT PAS appeler .run() dessus.
"""

from .yolo_training_pipeline import yolo_training_pipeline


def main() -> None:
    """Lance un run baseline du pipeline YOLO tiny via ZenML."""
    # Cet appel déclenche directement l'exécution du pipeline.
    # Le run sera visible dans ZenML Server et MLflow.
    yolo_training_pipeline(
        epochs=3,
        imgsz=320,
        exp_name="zenml_yolo_tiny_baseline",
    )


if __name__ == "__main__":
    main()
