# zenml-cv-yolo

Projet MLOps de détection d'objets (YOLO tiny) avec :

- **DVC** pour le versioning de données (tiny COCO "person")
- **MLflow + MinIO** pour le tracking d'expériences et artefacts
- **ZenML + ZenML Server** pour définir et visualiser des pipelines
- TP4 : exécution directe du script `train_cv.py`
- TP5 : exécution via un pipeline **ZenML**

## 1. Démarrer l'infrastructure (MLflow + MinIO + ZenML Server)

```bash
docker compose up -d
```

- MLflow UI : http://localhost:5000  
- MinIO console : http://localhost:9001  
- ZenML Server : http://localhost:8080  

## 2. Environnement Python local

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: adapter
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Initialiser ZenML et se connecter au serveur

Depuis la racine du projet :

```bash
# Se connecter au ZenML Server
zenml login http://localhost:8080

# Initialiser le repo ZenML (si pas déjà fait)
zenml init

# Créer une stack locale pointant vers MLflow + artefacts locaux
zenml orchestrator register local_orch --type=local || true
zenml artifact-store register local_artifacts --type=local --path=./artifacts || true
zenml experiment-tracker register mlflow_tracker --type=mlflow         --tracking_uri=http://localhost:5000 || true

zenml stack register local_stack         -o local_orch -a local_artifacts -e mlflow_tracker || true
zenml stack set local_stack
```

## 4. Dataset tiny COCO (person)

```bash
python tools/make_tiny_person_from_coco128.py
# et/ou dvc pull selon le TP
```

## 5. Lancer le pipeline ZenML (baseline)

```bash
bash scripts/run_zenml_pipeline.sh
```

## 6. Lancer une grille de pipelines ZenML

```bash
bash scripts/run_zenml_grid.sh
```

## 7. Où regarder ?

- **ZenML Server** : visualisation des pipelines, runs, stacks  
- **MLflow UI** : métriques, artefacts, comparaisons de runs  
- **MinIO** : stockage objet (si utilisé comme remote DVC)
