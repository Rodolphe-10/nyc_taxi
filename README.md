# NYC Taxi Tip Predictor — TP Python Avancé Master 1 IABD

## Installation
pip install -r requirements.txt

## Téléchargement des données
python scripts/download_data.py

## Lancer le pipeline complet
python scripts/part1_pipeline.py

## Lancer l'API
uvicorn src.api.main:app --reload

## Lancer les tests
pytest tests/ --cov=src --cov-report=html

## Lancer MLflow UI
mlflow ui --port 5000

## Docker
docker-compose up --build
