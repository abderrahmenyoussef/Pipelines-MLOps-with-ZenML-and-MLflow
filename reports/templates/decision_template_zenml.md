# TP5 – Décision de promotion (Pipeline ZenML CV YOLO)

## 1. Contexte

- Pipeline : `yolo_training_pipeline`
- Dataset : `tiny_coco` (person)
- Outils : ZenML + MLflow + DVC + MinIO

## 2. Runs de pipeline testés

| Run | epochs | imgsz | exp_name                     | mAP@50 | precision | recall |
|-----|--------|-------|------------------------------|--------|-----------|--------|
|  1  |        |       |                              |        |           |        |
|  2  |        |       |                              |        |           |        |
|  3  |        |       |                              |        |           |        |
|  4  |        |       |                              |        |           |        |

## 3. Analyse rapide

- Meilleur run (critère principal) :
- Compromis précision / rappel :
- Stabilité / reproductibilité (seed, variance) :
- Observations sur les artefacts (images, courbes, matrices de confusion) :

## 4. Décision

- Modèle **proposé pour Staging / NON-promu** :
- Justification (performance, risques, coût, temps d'entraînement) :

## 5. Rôle du pipeline ZenML (réflexion courte)

- Intérêt d'avoir structuré le flux en steps + pipeline :
- Ce que ZenML apporte par rapport au script brut (TP4) :
- Idées pour une future intégration CI/CD/CT avec GitLab CI :
