# zenml-cv-yolo

Projet MLOps de détection d'objets (YOLO tiny) avec :

- **DVC** pour le versioning de données (tiny COCO "person")
- **MLflow + MinIO** pour le tracking d'expériences et artefacts
- **ZenML + ZenML Server** pour définir et visualiser des pipelines
- TP4 : exécution directe du script `train_cv.py`
- TP5 : exécution via un pipeline **ZenML**

---

## 1. Démarrer l'infrastructure (MLflow + MinIO + ZenML Server)

```bash
docker compose up -d
````

* MLflow UI : [http://localhost:5000](http://localhost:5000)
* MinIO console : [http://localhost:9001](http://localhost:9001)
* ZenML Server : [http://localhost:8080](http://localhost:8080)

---

## 2. Environnement Python local

Dans ce projet, le **code des pipelines** est exécuté dans votre environnement Python local
(venv) mais **toute la configuration ZenML + stockage des artefacts** est centralisée
dans le **ZenML Server** (Docker) et MinIO (bucket S3).

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: adapter
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Initialiser ZenML et se connecter au serveur

Depuis la racine du projet :

```bash
# Se connecter au ZenML Server (dans Docker)
zenml connect http://localhost:8080

# Initialiser le repo ZenML (si pas déjà fait)
zenml init

# Vérifier les stacks disponibles sur le serveur
zenml stack list

# Sélectionner la stack préconfigurée (créée par l'enseignant)
zenml stack set mlflow_stack

# (Optionnel) Vérifier la stack active
zenml stack describe
```

La stack `mlflow_stack` est **préconfigurée dans le ZenML Server** (par l’enseignant) avec :

* un **orchestrateur local** (exécution des steps dans votre venv local) ;
* un **artifact store S3** sur **MinIO** (bucket `zenml-artifacts`) pour les artefacts ZenML ;
* un **experiment tracker MLflow** pointant vers le service MLflow dans Docker.

> ⚠️ En TP, **vous ne créez pas de stack** (`register`), vous utilisez uniquement celle
> qui est déjà définie côté serveur.

---

## 4. Dataset tiny COCO (person)

Toujours depuis la racine du projet :

```bash
python tools/make_tiny_person_from_coco128.py
# et/ou dvc pull selon le TP
```

Le dataset minimal `tiny_coco` est ensuite utilisé par le script d’entraînement YOLO
et par les steps du pipeline ZenML.

---

## 5. Lancer le pipeline ZenML (baseline)

Pipeline ZenML "baseline" (YOLO tiny, hyperparamètres par défaut) :

```bash
bash scripts/run_zenml_pipeline.sh
# ou
python -m src.zenml_pipelines.run_yolo_pipeline_baseline
```

Ce pipeline :

* exécute les **steps ZenML** (préparation des données, entraînement, évaluation, …) ;
* enregistre les **runs de pipeline** dans le **ZenML Server** ;
* loggue les **métriques & artefacts de training** dans **MLflow**.

---

## 6. Lancer une grille de pipelines ZenML

Pour lancer une **grille de runs** (variations sur `imgsz`, `epochs`, etc.) :

```bash
bash scripts/run_zenml_grid.sh
# ou
python -m src.zenml_pipelines.run_yolo_pipeline_grid
```

Ce script lance plusieurs exécutions du pipeline avec des hyperparamètres différents.

---

## 7. Où regarder ?

* **ZenML Server** ([http://localhost:8080](http://localhost:8080))

  * Vue **pipelines** : définition des pipelines ZenML.
  * Vue **runs** : liste des exécutions (baseline + grille).
  * Vue **stacks** : stack `mlflow_stack` (orchestrateur, artifact store MinIO, MLflow tracker).

* **MLflow UI** ([http://localhost:5000](http://localhost:5000))

  * Liste des runs (projet YOLO tiny).
  * Métriques (`mAP@50`, précision, rappel, etc.).
  * Artefacts : images de résultats, matrices de confusion, poids du modèle, …

* **MinIO** ([http://localhost:9001](http://localhost:9001))

  * Bucket `zenml-artifacts` : artefacts ZenML (tous les outputs des steps / pipelines).
  * (Éventuellement) autres buckets utilisés comme **remote DVC** ou pour MLflow.

---

## 8. (Annexe) Configuration initiale dans le conteneur ZenML Server

> Cette section est uniquement pour la **configuration initiale** réalisée par
> l’enseignant / admin. Les étudiants n’ont PAS à exécuter ces commandes.

1. Dans la console MinIO ([http://localhost:9001](http://localhost:9001)), créer un **bucket** :

   * nom : `zenml-artifacts`

2. Ouvrir un shell dans le conteneur ZenML Server (nom à adapter selon le `docker-compose`) :

```bash
docker exec -it zenml-server bash
```

3. Dans le conteneur, exécuter :

```bash
# 1) Experiment tracker MLflow (MLflow est un autre service Docker, accessible via "mlflow")
zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri=http://mlflow:5000 \
    --tracking_token="dummy-token"

# 2) Secret MinIO (identifiants du service MinIO)
zenml secret create minio_zenml_secret \
    --aws_access_key_id='minio' \
    --aws_secret_access_key='minio12345'

# 3) Artifact store ZenML sur MinIO (bucket zenml-artifacts)
#    Attention : depuis le conteneur, MinIO est accessible via le hostname "minio"
zenml artifact-store register minio_artifacts \
    --flavor=s3 \
    --path='s3://zenml-artifacts' \
    --authentication_secret=minio_zenml_secret \
    --client_kwargs='{"endpoint_url": "http://minio:9000", "region_name": "us-east-1"}'

# 4) Orchestrateur local (exécution des steps sur la machine où le pipeline est lancé)
zenml orchestrator register local_orch --flavor=local

# 5) Stack complète : orchestrateur local + artifact store MinIO + MLflow tracker
zenml stack register mlflow_stack \
    -o local_orch -a minio_artifacts -e mlflow_tracker

# 6) Définir cette stack comme stack par défaut dans le serveur
zenml stack set mlflow_stack
```

Après cette configuration :

* la stack `mlflow_stack` est visible dans l’UI ZenML Server,
* les étudiants n’ont plus qu’à faire :

  * `zenml connect http://localhost:8080`
  * `zenml init`
  * `zenml stack set mlflow_stack`
  * puis lancer les pipelines.

---

