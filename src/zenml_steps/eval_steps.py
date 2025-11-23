from typing import Optional
from zenml import step


@step
def summarize_yolo_experiment(exp_name: Optional[str] = None) -> None:
    """Étape de résumé / évaluation simple pour le TP.

    L'analyse détaillée (mAP, précision, rappel, artefacts)
    se fait dans l'UI MLflow. Ici on se contente de rappeler
    le nom d'expérience et de pointer vers MLflow.
    """
    print("[summarize_yolo_experiment] Résumé du run YOLO tiny.")
    if exp_name:
        print(f"  - Nom d'expérience MLflow : {exp_name}")
    print("  - Ouvrez l'UI MLflow (http://localhost:5000) pour comparer les runs.")
