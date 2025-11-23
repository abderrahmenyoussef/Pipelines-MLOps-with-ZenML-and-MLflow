import subprocess
from zenml import step


@step
def prepare_tiny_coco_dataset() -> None:
    """Prépare / vérifie le dataset tiny_coco via DVC.

    - si un remote DVC est configuré : dvc pull pour récupérer les données
    - sinon : le script ne fait qu'afficher un message.
    """
    print("[prepare_tiny_coco_dataset] Vérification du dataset tiny_coco avec DVC...")
    try:
        subprocess.run(["dvc", "pull"], check=False)
    except FileNotFoundError:
        print("[prepare_tiny_coco_dataset] DVC non trouvé dans l'environnement.")
