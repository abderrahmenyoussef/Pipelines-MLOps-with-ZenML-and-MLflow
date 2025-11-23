import subprocess
from typing import Optional
from zenml import step


@step
def train_yolo_tiny(
    epochs: int = 3,
    imgsz: int = 320,
    exp_name: str = "zenml_yolo_tiny",
) -> Optional[str]:
    """Lance l'entraînement YOLO tiny en appelant src/train_cv.py.

    Les métriques et artefacts sont loggés dans MLflow (comme au TP4).
    Ce step renvoie le nom de l'expérience MLflow utilisée.
    """
    cmd = [
        "python",
        "-m",
        "src.train_cv",
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--exp-name",
        exp_name,
    ]

    print(f"[train_yolo_tiny] Commande : {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print("[train_yolo_tiny] Entraînement YOLO tiny terminé avec une ERREUR.")
    else:
        print("[train_yolo_tiny] Entraînement YOLO tiny terminé avec SUCCÈS.")

    return exp_name
