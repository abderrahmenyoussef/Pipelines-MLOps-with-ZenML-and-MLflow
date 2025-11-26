# zenml-cv-yolo (version AWS S3)

Projet MLOps de détection d'objets (YOLO tiny) avec :

* **DVC** pour le versioning de données (tiny COCO "person")
* **MLflow + AWS S3** pour le tracking d'expériences et artefacts
* **ZenML + ZenML Server** pour définir et visualiser des pipelines
* TP4 : exécution directe du script `train_cv.py`
* TP5 : exécution via un pipeline **ZenML**

---

## 1. Démarrer l'infrastructure (MLflow + ZenML Server + AWS S3)

Avant de démarrer, créer un fichier **aws.env** :

```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxx
AWS_SESSION_TOKEN=xxxxxxxxxxxxxxxxxxxx
AWS_DEFAULT_REGION=us-east-1
```

Puis lancer l'infrastructure :

```bash
docker compose up -d
```

* MLflow UI : [http://localhost:5000](http://localhost:5000)
* ZenML Server : [http://localhost:8080](http://localhost:8080)

> Les artefacts MLflow et ZenML seront stockés dans **AWS S3** (bucket à définir).

---

## 2. Environnement Python local

Les **pipelines ZenML s’exécutent dans votre environnement Python local**,
mais la configuration globale (stack, metadata, artefacts) est centralisée via :

* ZenML Server (dans Docker)
* S3 AWS (artefacts ZenML et MLflow)

Commandes pour créer votre venv :

```bash
python3 -m venv .venv
source .venv/bin/activate          # Adapter sous Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Initialiser ZenML et se connecter au serveur

Depuis la racine du projet :

```bash
# Connexion au ZenML Server
zenml connect http://localhost:8080

# Initialiser ZenML localement
zenml init

# Lister les stacks disponibles sur le serveur
zenml stack list

# Sélectionner la stack fournie par l’enseignant
zenml stack set mlflow_stack

# (Optionnel)
zenml stack describe
```

La stack `mlflow_stack` est préconfigurée dans le serveur avec :

* orchestrateur local
* artifact store **S3 AWS**
* experiment tracker MLflow

> ⚠️ Les étudiants **ne créent pas de stack**, ils utilisent celle du serveur.

---

## 4. Dataset tiny COCO (person)

```bash
python tools/make_tiny_person_from_coco128.py
# ou dvc pull selon le TP
```

Le dataset `tiny_coco` est utilisé pour :

* l’entraînement YOLO
* les steps ZenML

---

## 5. Lancer le pipeline ZenML (baseline)

```bash
bash scripts/run_zenml_pipeline.sh
# ou
python -m src.zenml_pipelines.run_yolo_pipeline_baseline
```

Ce pipeline :

* exécute les steps ZenML (préparation / entraînement / évaluation)
* loggue tous les artefacts sur **AWS S3**
* enregistre les runs dans :

  * ZenML Server
  * MLflow (métriques + artefacts)

---

## 6. Lancer une grille de pipelines ZenML

```bash
bash scripts/run_zenml_grid.sh
# ou
python -m src.zenml_pipelines.run_yolo_pipeline_grid
```

Lance plusieurs runs avec des variations d’hyperparamètres.

---

## 7. Où regarder ?

### **ZenML Server** — [http://localhost:8080](http://localhost:8080)

* Pipelines
* Runs
* Stacks (`mlflow_stack` → S3 + MLflow + orchestrateur local)

### **MLflow UI** — [http://localhost:5000](http://localhost:5000)

* Runs YOLO tiny
* Métirques (`mAP@50`, précision, rappel…)
* Artefacts → stockés physiquement dans **S3**

### **AWS S3**

* Bucket `mlflow-artifacts` (ou autre)
* Bucket `zenml-artifacts` (ou autre)
* Contient :

  * métriques
  * images de résultats
  * poids YOLO entraînés
  * JSON de configuration
  * outputs ZenML

---

## 8. (Annexe) Configuration initiale côté ZenML Server (enseignant)

> Cette section est **réservée à l’enseignant / admin**.
> Les étudiants **n’exécutent rien de tout ceci**.

1. Créer 2 buckets AWS S3 :

* `mlflow-artifacts`
* `zenml-artifacts`

2. Entrer dans le conteneur ZenML Server :

```bash
docker exec -it zenml-server bash
```

3. Configurer MLflow comme tracker :

```bash
zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri=http://mlflow:5000 \
    --tracking_token="dummy-token"
```

4. Créer le secret AWS :

```bash
zenml secret create aws_s3_secret \
    --aws_access_key_id="$AWS_ACCESS_KEY_ID" \
    --aws_secret_access_key="$AWS_SECRET_ACCESS_KEY" \
    --aws_session_token="$AWS_SESSION_TOKEN"
```

5. Artifact store S3 :

```bash
zenml artifact-store register s3_artifacts \
    --flavor=s3 \
    --path='s3://VOTRE_BUCKET/zenml-artifacts' \
    --authentication_secret=aws_s3_secret
```

6. Orchestrateur local :

```bash
zenml orchestrator register local_orch --flavor=local
```

7. Stack complète :

```bash
zenml stack register mlflow_stack \
    -o local_orch -a s3_artifacts -e mlflow_tracker
```

8. Définir la stack par défaut :

```bash
zenml stack set mlflow_stack
```