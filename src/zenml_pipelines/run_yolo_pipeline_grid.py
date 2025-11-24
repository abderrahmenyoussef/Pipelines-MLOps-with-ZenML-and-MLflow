from __future__ import annotations

"""
Script de lancement d'une "grille" de runs du pipeline ZenML.

On lance plusieurs exécutions du pipeline avec des hyperparamètres
différents (epochs, imgsz, nom d'expérience), sans appeler .run().
"""

from .yolo_training_pipeline import yolo_training_pipeline


def main() -> None:
    """Lance plusieurs runs du pipeline YOLO tiny avec différents paramètres."""

    # Liste de configurations à tester.
    configs = [
        {"epochs": 3, "imgsz": 320, "exp_name": "zenml_yolo_tiny_e3_320"},
        {"epochs": 5, "imgsz": 320, "exp_name": "zenml_yolo_tiny_e5_320"},
        {"epochs": 3, "imgsz": 416, "exp_name": "zenml_yolo_tiny_e3_416"},
        {"epochs": 5, "imgsz": 416, "exp_name": "zenml_yolo_tiny_e5_416"},
    ]

    for cfg in configs:
        print(f"[ZenML] Lancement pipeline avec paramètres : {cfg}")
        # Chaque appel lance un nouveau run de pipeline.
        yolo_training_pipeline(**cfg)


if __name__ == "__main__":
    main()
