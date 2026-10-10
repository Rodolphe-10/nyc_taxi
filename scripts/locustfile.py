"""Charge de l'API : locust -f scripts/locustfile.py --host http://127.0.0.1:8000.

Objectif de l'énoncé : 100 requêtes par seconde sur POST /predict.
"""

from locust import HttpUser, constant, task


_TRIP = {
    "passenger_count": 1,
    "trip_distance": 2.5,
    "pickup_datetime": "2023-01-15T08:30:00",
    "dropoff_datetime": "2023-01-15T08:48:00",
    "pu_location_id": 161,
    "do_location_id": 236,
    "payment_type": 1,
    "fare_amount": 12.5,
}


class TipUser(HttpUser):
    """Envoie des courses valides sans temps d'attente."""

    wait_time = constant(0)

    @task
    def predict(self) -> None:
        self.client.post("/predict", json=_TRIP)
