# NYC Taxi Tip Predictor — TP Python avancé, Master 1 IABD

Prédiction du pourcentage de pourboire `tip_pct = tip_amount / fare_amount` sur les courses Yellow Taxi 2023. Le dépôt couvre le pipeline (parties 1 à 3), l'entraînement LightGBM suivi dans MLflow, le registre, DVC, l'API FastAPI et une image Docker.

Interpréteur utilisé pour les mesures ci-dessous : Python 3.10.11 (`C:\VENV\machine-learning\Scripts\python.exe`). La chaîne d'intégration vise Python 3.11. Cette chaîne n'a pas été exécutée sur GitHub depuis cette machine.

## Scores

Commande, depuis la racine du dépôt, le 9 octobre 2026 :

```
python -m src.models.train --mode full --n_rows 100000
```

Fenêtre conservée après filtre 2023 : `2023-01-01 00:00:00` → `2023-01-02 13:35:06`. 100 000 lignes chargées, 72 735 en entraînement, 18 751 en validation. Découpage chronologique. Fichier : `reports/metrics.json` (date `2026-10-09T13:14:36Z`).

| Modèle | RMSE | MAE | R² | RMSE carte |
|---|---:|---:|---:|---:|
| baseline_mean | 0.143131 | 0.123788 | −0.001740 | 0.125168 |
| baseline_linear | 0.097639 | 0.069525 | 0.533839 | 0.105965 |
| grid_001 | 0.085689 | 0.052029 | 0.640960 | 0.100172 |
| embedding_lightgbm | 0.085746 | 0.053449 | 0.640486 | 0.100168 |

`grid_001` est le run promu : `09ac8b9ff161415bb8f1042ec8287421`, registre `TipPredictor` version 1, stage Production (`reports/registry.json`). Couverture de l'intervalle 10 %–90 % sur la validation : 0.858674 (toutes les courses), 0.807757 (carte seule). Largeur moyenne : 0.181195.

## Installation

```
pip install -r requirements.txt
```

## Données

```
python scripts/download_data.py
```

Janvier 2023 est présent dans `data/raw/` (ignoré par Git). Les fichiers traités versionnés par DVC sont des fixtures, pas les douze mois.

## Entraînement et tests

```
python -m src.models.train --mode full --n_rows 100000
pytest tests/ --collect-only -q
```

Le 10 octobre 2026, `pytest tests/ --collect-only -q` a collecté **253** tests.

## API

```
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Exemple, course du schéma (`pu` 161, `do` 236, tarif 12,5, paiement carte). Réponse mesurée le 10 octobre 2026, modèle version 1, aussi bien en local que dans le conteneur :

```
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"passenger_count\":1,\"trip_distance\":2.5,\"pickup_datetime\":\"2023-01-15T08:30:00\",\"dropoff_datetime\":\"2023-01-15T08:48:00\",\"pu_location_id\":161,\"do_location_id\":236,\"payment_type\":1,\"fare_amount\":12.5}"
```

```json
{"tip_pct_predicted":0.23720673223660585,"confidence_interval":[0.07625590919334785,0.32692497017648364],"model_version":"1"}
```

Santé : `GET /health` renvoie `{"status":"ok","model_version":"1"}` quand les artefacts de `reports/` sont chargés.

Charge mesurée avec `scripts/load_test.py` (locust n'est pas installé), fichier `reports/load_test.json` :

- 100 requêtes/s demandées pendant 5 s (500 requêtes) : 13,85 requêtes/s atteintes, p50 2,78 s, p95 3,40 s, 500 réponses 200.
- 10 requêtes/s pendant 3 s : 10,06 requêtes/s, p50 0,076 s, p95 0,086 s, 30 réponses 200.

## MLflow

```
mlflow ui --port 5000
```

Export léger : `reports/mlflow_runs.csv` et les captures `docs/img/mlflow_runs_rmse.png`, `docs/img/mlflow_ui_rmse.png`. Le dossier `mlruns/` n'est pas versionné.

## Docker

```
docker build -t tip-api .
docker run -p 8000:8000 tip-api
```

`docker images tip-api` affiche **403MB** après cette construction.

Le `docker build` copie dans l'image les artefacts déjà écrits dans `reports/` : `model.txt`, `preprocessor.joblib`, `quantile_10.txt`, `quantile_90.txt` et `registry.json`. L'image n'installe pas le client MLflow. Au démarrage, l'API tente le registre puis sert ces fichiers. La version vient de `registry.json`.

```
docker compose up --build
```

Cette commande démarre l'API, MLflow sur PostgreSQL, Prometheus (scrape de `/metrics`) et Grafana. Le tableau de bord `Tip API` est sur http://localhost:3000 (compte local `admin` / `admin`).

Le 10 octobre 2026, `docker compose up -d --build` a abouti (code 0). Contrôles ensuite : `/health` de l'API (version 1), la prédiction ci-dessus, `/health` de MLflow (`OK`), cible Prometheus `tip-api` à l'état `up`, tableau Grafana `tip-api` affichant le compteur de requêtes et les quantiles de latence.

## Application

```
streamlit run app.py
```

La page Résultats lit `reports/benchmarks.json` (produit par `python scripts/measure_benchmarks.py`, Python 3.10.11, le 10 octobre 2026). Sur 20 000 lignes synthétiques, la comparaison iterrows / NumPy donne un speedup de 414,7 (moyennes 4 999 ms et 12,1 ms, 3 répétitions). Sur 200 000 lignes de janvier 2023, `optimize_dtypes` réduit la mémoire de 40 400 128 à 13 000 352 octets (gain 67,8 %). La page Partie 4 lit `reports/metrics.json`.
