# TP5 – Pipelines MLOps avec ZenML, MLflow et AWS S3

Ce document décrit **toutes les étapes** que j’ai suivies pour réaliser la version **AWS S3** du TP :

- création et configuration du bucket S3  
- configuration de `docker-compose` (MLflow + ZenML Server)  
- configuration des credentials AWS côté Docker **et** côté machine locale  
- création et activation d’une stack ZenML (`mlflow_stack`) basée sur S3 + MLflow  
- génération du dataset `tiny_coco`  
- exécution du pipeline **baseline** et du pipeline **grid** avec ZenML  
- vérification des artefacts dans S3, des runs dans ZenML et MLflow  


---

## 1. Pré-requis

- Python 3.11 + environnement virtuel (`conda` ou `venv`)
- Docker + Docker Compose
- Compte AWS avec accès S3

---

## 2. Création et configuration du bucket S3

Sur la console AWS, j’ai créé un bucket :

* **Nom du bucket** : `mlops-dvc-abdou`
* Région : `us-east-1`

Dans ce bucket, j’utilise deux préfixes principaux :

* `mlflow-artifacts/` – pour les artefacts MLflow
* `zenml-artifacts/` – pour les artefacts ZenML

**Capture 1 – Dossiers S3**

![Création des dossiers mlflow-artifacts et zenml-artifacts dans S3](./captures/1.png)

---

## 3. Configuration des credentials AWS

### 3.1. Variables d’environnement sur la machine locale

Avant de lancer les pipelines ZenML (orchestrateur local), j’exporte les variables AWS dans mon terminal :

```bash
export AWS_ACCESS_KEY_ID=<MON_AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<MON_AWS_SECRET_ACCESS_KEY>
export AWS_SESSION_TOKEN=<MON_AWS_SESSION_TOKEN>   # si nécessaire
export AWS_DEFAULT_REGION=us-east-1
```

Pour éviter de les retaper, on peut aussi utiliser un fichier `aws.env` :

```env
# aws.env (non versionné dans Git)
AWS_ACCESS_KEY_ID=<MON_AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<MON_AWS_SECRET_ACCESS_KEY>
AWS_SESSION_TOKEN=<MON_AWS_SESSION_TOKEN>
AWS_DEFAULT_REGION=us-east-1
```

Puis :

```bash
set -a
source aws.env
set +a
```

### 3.2. Credentials côté services Docker (MLflow & ZenML Server)

Dans le fichier `docker-compose.yml`, j’ai ajouté les variables AWS dans les services **mlflow** et **zenml-server** pour qu’ils puissent eux aussi accéder à S3.

Exemple pour `zenml-server` :

```yaml
zenml-server:
  image: zenmldocker/zenml-server:0.92.0
  environment:
    ZENML_STORE_URL: mysql://zenml:zenmlpass@zenml-db:3306/zenml
    ZENML_DEFAULT_USER_NAME: admin
    ZENML_DEFAULT_USER_PASSWORD: zenml
    AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
    AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
    AWS_SESSION_TOKEN: ${AWS_SESSION_TOKEN}
    AWS_DEFAULT_REGION: ${AWS_DEFAULT_REGION}
  # ...
```

Exemple pour `mlflow` (artifacts S3) :

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.16.0
  command: >
    mlflow server --host 0.0.0.0 --port 5000
    --backend-store-uri sqlite:///mlflow.db
    --artifacts-destination s3://mlops-dvc-abdou/mlflow-artifacts
  environment:
    AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
    AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
    AWS_SESSION_TOKEN: ${AWS_SESSION_TOKEN}
    AWS_DEFAULT_REGION: ${AWS_DEFAULT_REGION}
  # ...
```

---

## 4. Lancement de l’infrastructure Docker

Depuis la racine du projet :

```bash
docker compose up -d
docker compose ps
```

Services utilisés :

* `zenml-server` exposé sur `http://localhost:8080`
* `mlflow` exposé sur `http://localhost:5000`

Connexion du client ZenML au serveur :

```bash
zenml login --url http://localhost:8080
```

Écran de confirmation :

**Capture 2 – Device ajouté dans ZenML**

![Connexion du client au ZenML server](./captures/2.png)

---

## 5. Initialisation ZenML (côté client)

Dans mon environnement virtuel (ici `agents`) :

```bash
# Connexion au serveur ZenML
zenml login --url http://localhost:8080

# Initialisation du repo ZenML dans le projet
zenml init
```

Vérification des stacks :

```bash
zenml stack list
```

---

## 6. Création de la stack ZenML `mlflow_stack`

### 6.1. Enregistrement des composants

1. **Experiment Tracker MLflow**

```bash
zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri=http://localhost:5000 \
    --tracking_token="dummy-token"
```

2. **Artifact Store S3**

```bash
zenml artifact-store register s3_artifacts \
    --flavor=s3 \
    --path=s3://mlops-dvc-abdou/zenml-artifacts
```

3. **Orchestrateur local**

```bash
zenml orchestrator register local_orch --flavor=local
```

### 6.2. Enregistrement et activation de la stack

```bash
zenml stack register mlflow_stack \
    -o local_orch \
    -a s3_artifacts \
    -e mlflow_tracker

zenml stack set mlflow_stack
```

Vérification :

```bash
zenml stack list
```

**Capture 3 – Stack mlflow_stack active**

![Stack mlflow\_stack avec local\_orch, mlflow\_tracker, s3\_artifacts](./captures/3.png)

---

## 7. Génération du dataset `tiny_coco`

Depuis la racine du projet :

```bash
python tools/make_tiny_person_from_coco128.py
```

Le script crée un dataset minimal à partir de COCO, dans `data/tiny_coco`.

---

## 8. Exécution du pipeline **baseline** ZenML

Le TP fournit un script qui lance un pipeline ZenML `yolo_training_pipeline` pour un run baseline YOLO tiny.

```bash
bash scripts/run_zenml_pipeline.sh
# (équivalent Python)
# python -m src.zenml_pipelines.run_yolo_pipeline_baseline
```

Ce pipeline :

1. Vérifie / prépare le dataset `tiny_coco`
2. Lance `train_cv.py` pour entraîner YOLOv8 tiny (`epochs=3`, `imgsz=320`, expérience `zenml_yolo_tiny_baseline`)
3. Logge automatiquement les résultats dans MLflow
4. Stocke les artefacts dans :

   * `s3://mlops-dvc-abdou/zenml-artifacts/` (ZenML)
   * `s3://mlops-dvc-abdou/mlflow-artifacts/` (MLflow)

**Capture 4 – Baseline dans ZenML**

![Run du pipeline yolo\_training\_pipeline (baseline) dans ZenML](./captures/4.png)

**Capture 5 – Baseline dans MLflow**

![Run zenml\_yolo\_tiny\_baseline dans l'expérience cv\_yolo\_tiny](./captures/5.png)

**Capture 6 – Artefacts ZenML sur S3**

![Contenu de s3://mlops-dvc-abdou/zenml-artifacts](./captures/6.png)

**Capture 7 – Artefacts MLflow sur S3**

![Contenu de s3://mlops-dvc-abdou/mlflow-artifacts](./captures/7.png)

---

## 9. Exécution du **grid search** (plusieurs runs ZenML)

Le TP prévoit un script pour lancer plusieurs runs du pipeline avec des hyperparamètres différents (grille sur `epochs` et `imgsz`).

```bash
bash scripts/run_zenml_grid.sh
# (équivalent Python)
# python -m src.zenml_pipelines.run_yolo_pipeline_grid
```

Ce script déclenche plusieurs exécutions de `yolo_training_pipeline` avec des configurations du type :

* `zenml_yolo_tiny_e3_320`
* `zenml_yolo_tiny_e5_320`
* `zenml_yolo_tiny_e3_416`
* `zenml_yolo_tiny_e5_416`
* etc.

Chaque run est :

* tracé dans ZenML
* loggé dans MLflow (expérience `cv_yolo_tiny`)
* stocké dans S3 via `s3_artifacts` et `mlflow-artifacts/`.

**Capture 8 – Pipeline en cours dans ZenML**

![Run en cours du pipeline yolo\_training\_pipeline](./captures/8.png)

**Capture 9 – Liste des runs de pipeline (grid + baseline) dans ZenML**

![Plusieurs runs yolo\_training\_pipeline complétés avec la stack mlflow\_stack](./captures/9.png)

**Capture 10 – Tous les runs dans MLflow**

![Expérience cv\_yolo\_tiny avec les runs baseline + grille](./captures/10.png)

---

## 10. Analyse des résultats dans MLflow

Pour comparer les runs :

1. Ouvrir MLflow : `http://localhost:5000`
2. Aller dans l’expérience `cv_yolo_tiny`
3. Cocher plusieurs runs (baseline + runs de la grille)
4. Cliquer sur **Compare**
5. Observer :

   * les paramètres (`epochs`, `imgsz`, …)
   * les métriques (`mAP50`, `mAP50-95`, etc.)
   * les artefacts (poids du modèle, courbes de loss, labels, etc.)

Ces observations sont utilisées dans le compte-rendu pour :

* choisir le **meilleur run** (compromis entre performance et coût d’entraînement)
* discuter d’un éventuel passage en **Staging / modèle de référence**.

> NB : Nous avons effectué cette analyse comparative dans MLflow ; le compte-rendu synthétique et la décision de promotion se trouvent dans le template de décision : [`reports/templates/decision_template_zenml.md`](reports/templates/decision_template_zenml.md).

---

## 11. Résumé de l’architecture MLOps

* **ZenML Server**

  * héberge les **pipelines**, les **stacks** et les **runs**
  * accessible sur `http://localhost:8080`

* **Stack `mlflow_stack`**

  * orchestrateur : `local_orch` (exécution des steps sur la machine locale)
  * artifact store : `s3_artifacts` (`s3://mlops-dvc-abdou/zenml-artifacts`)
  * experiment tracker : `mlflow_tracker` (`http://localhost:5000`)

* **MLflow**

  * interface d’expérimentation (tracking des runs, métriques, paramètres, artefacts)
  * utilisation de S3 (`mlflow-artifacts/`) pour stocker les modèles et fichiers générés

* **S3**

  * stockage centralisé et persistant des artefacts :

    * côté ZenML (`zenml-artifacts/`)
    * côté MLflow (`mlflow-artifacts/`)

Cette architecture permet de :

* rejouer les pipelines facilement (mêmes steps, même stack)
* comparer les expériences dans MLflow
* conserver tous les artefacts dans le cloud (S3)
* préparer une future intégration CI/CD/CT (ex. GitLab CI) en automatisant les scripts :

  * `bash scripts/run_zenml_pipeline.sh`
  * `bash scripts/run_zenml_grid.sh`

---
