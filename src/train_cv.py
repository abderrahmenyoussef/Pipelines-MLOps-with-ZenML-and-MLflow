import argparse
import os
from pathlib import Path

import mlflow
from mlflow import log_metric, log_param
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO tiny with MLflow tracking.")
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'epochs")
    parser.add_argument("--imgsz", type=int, default=320, help="Taille d'image (carrée)")
    parser.add_argument("--exp-name", type=str, default="cv_yolo_tiny", help="Nom d'expérience MLflow")
    parser.add_argument(
        "--data",
        type=str,
        default="configs/tiny_coco.yaml",
        help="Fichier data YAML pour YOLO",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Poids YOLOv8 à utiliser comme base",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    #mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("cv_yolo_tiny")

    with mlflow.start_run(run_name=args.exp_name):
        log_param("epochs", args.epochs)
        log_param("imgsz", args.imgsz)
        log_param("data_yaml", args.data)
        log_param("model", args.model)

        # Entraînement YOLO
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            project="runs/train",
            name=args.exp_name,
        )

        # On essaye d'extraire quelques métriques si disponibles
        try:
            metrics = results.results_dict  # ultralytics >=8.3
        except Exception:
            metrics = {}

        # Log de quelques métriques courantes si elles existent
        for key in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]:
            if key in metrics:
                log_metric(key, float(metrics[key]))

        # On logge au moins une métrique fictive si rien n'est dispo
        if not metrics:
            log_metric("training_finished", 1.0)

        # Log d'un lien vers le dossier de sortie YOLO
        run_dir = Path("runs/train") / args.exp_name
        mlflow.log_param("yolo_run_dir", str(run_dir))

        print(f"Entraînement terminé. Résultats YOLO dans {run_dir}")
        print("Consultez l'UI MLflow pour les métriques et artefacts.")


if __name__ == "__main__":
    main()
