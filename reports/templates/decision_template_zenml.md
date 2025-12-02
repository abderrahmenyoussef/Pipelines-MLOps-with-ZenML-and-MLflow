# TP5 – Décision de promotion (Pipeline ZenML CV YOLO)

## 1. Contexte

- Pipeline : `yolo_training_pipeline`
-- Dataset : `tiny_coco` (person)
-- Outils : ZenML + MLflow + DVC + S3 bucket

## 2. Runs de pipeline testés

| Run | epochs | imgsz | exp_name                     | mAP@50 | precision | recall |
|-----|--------|-------|------------------------------|--------|-----------|--------|
|  1  | 3      | 320   | `zenml_yolo_tiny_baseline`   | 0.2408 | 0.0075    | 0.6452 |
|  2  | 3      | 320   | `zenml_yolo_tiny_e3_320`     | 0.2408 | 0.0075    | 0.6452 |
|  3  | 3      | 416   | `zenml_yolo_tiny_e3_416`     | 0.2713 | 0.0080    | 0.7742 |
|  4  | 5      | 320   | `zenml_yolo_tiny_e5_320`     | 0.2600 | 0.0082    | 0.6774 |
|  5  | 5      | 416   | `zenml_yolo_tiny_e5_416`     | 0.3301 | 0.0080    | 0.7742 |

> Notes : les valeurs listées sont les métriques finales (dernier epoch) extraites des `results.csv` de chaque run. Les précisions sont arrondies à 4 décimales.

## 3. Analyse rapide

- **Meilleur run (critère principal)** : `zenml_yolo_tiny_e5_416` (mAP@50 = 0.3301). Ce run obtient la meilleure mAP@50 parmi les essais.
- **Compromis précision / rappel** : les runs avec `imgsz=416` (surtout `e5_416`) montrent un meilleur rappel (~0.77) et une mAP supérieure, tandis que la précision absolue reste très faible (valeurs très basses autour de 0.007–0.008) — il faudra vérifier l'interprétation de la colonne "precision(B)" (échelle / métrique) et la cohérence des annotations.
- **Stabilité / reproductibilité** : tous les runs utilisent `seed: 0` et `deterministic: true` dans `args.yaml`, ce qui facilite la reproductibilité locale. Plusieurs runs (baseline et `e3_320`) montrent des métriques identiques — vérifier si ce sont des reruns ou si la configuration était identique.
- **Observations sur les artefacts** :
	- Les modèles et weights sont sauvegardés dans les répertoires `runs/train/<exp_name>/weights` (vérifier fichiers `.pt`).
	- Les résultats (courbes, images de validation) sont générés (`plots: true`) — utile pour inspection qualitative.

## 4. Décision

- **Modèle proposé pour Staging** : `zenml_yolo_tiny_e5_416`.
- **Justification** :
	- Meilleure mAP@50 (0.3301) sur le jeu de validation.
	- Rappel élevé (≈0.77), indiquant que le modèle détecte bien les instances (moins de faux négatifs).
	- Configuration raisonnable (5 epochs, `imgsz=416`) — coût d'entraînement modéré sur dataset tiny.
	- Tous les runs sont reproductibles (seed fixé), facilitant promotion vers un environnement contrôlé.

## 5. Rôle du pipeline ZenML (réflexion courte)

- **Intérêt d'avoir structuré le flux en steps + pipeline** : facilite la traçabilité des trials, réexécution partielle (caching), et la collecte automatique des artefacts et métadonnées.
-- **Ce que ZenML apporte par rapport au script brut (TP4)** : orchestration des steps, intégration avec MLflow pour le tracking, supports pour stockage d'artefacts (S3 bucket / DVC), et meilleures pratiques reproductibles.
-- **Idées pour une future intégration CI/CD/CT** :
	- Ajouter un test automatique de non-régression des métriques (p.ex. mAP@50 must be >= seuil) dans GitLab CI.
	- Automatiser l'enregistrement des artefacts vers un S3 bucket et validation des poids (checksum) avant promotion.
	- Lancer runs de validation plus longs (10–20 epochs) et/ou hyperparameter search contrôlé (ZenML + Optuna) avant promotion en production.

---


