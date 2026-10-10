"""Schémas Pydantic v2 des requêtes et réponses de prédiction."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

_EXAMPLE = {
    "passenger_count": 1,
    "trip_distance": 2.5,
    "pickup_datetime": "2023-01-15T08:30:00",
    "dropoff_datetime": "2023-01-15T08:48:00",
    "pu_location_id": 161,
    "do_location_id": 236,
    "payment_type": 1,
    "fare_amount": 12.5,
}


class TripPredictionRequest(BaseModel):
    """Course à scorer.

    ``fare_amount`` n'est pas dans la liste de l'énoncé. Le modèle prédit
    ``tip_amount / fare_amount`` et le prétraitement filtre le tarif, donc
    le champ est requis. ``dropoff_datetime`` est facultatif : l'énoncé
    demande de rejeter une course de plus de 12 heures, sans fournir
    d'heure d'arrivée. S'il est absent, ce contrôle est ignoré.
    """

    model_config = ConfigDict(json_schema_extra={"examples": [_EXAMPLE]})

    passenger_count: int = Field(ge=1, le=8)
    trip_distance: float = Field(gt=0, le=100)
    pickup_datetime: datetime
    pu_location_id: int = Field(ge=1, le=265)
    do_location_id: int = Field(ge=1, le=265)
    payment_type: Literal[1, 2, 3, 4]
    fare_amount: float = Field(gt=0, le=200)
    dropoff_datetime: Optional[datetime] = None

    @model_validator(mode="after")
    def check_distance_and_duration(self) -> "TripPredictionRequest":
        """Rejette une course trop courte ou trop longue."""
        if self.trip_distance < 0.1:
            raise ValueError("La distance doit être d'au moins 0,1 mile.")
        if self.dropoff_datetime is None:
            return self
        duration = self.dropoff_datetime - self.pickup_datetime
        if duration <= timedelta(0):
            raise ValueError("L'heure d'arrivée doit être postérieure au départ.")
        if duration > timedelta(hours=12):
            raise ValueError("La durée de la course dépasse 12 heures.")
        return self


class TripPredictionResponse(BaseModel):
    """Pourcentage de pourboire prédit, avec un intervalle ordonné."""

    tip_pct_predicted: float = Field(ge=0, le=1)
    confidence_interval: list[float] = Field(min_length=2, max_length=2)
    model_version: str

    @model_validator(mode="after")
    def check_interval(self) -> "TripPredictionResponse":
        """L'intervalle est ordonné et chaque borne est un pourcentage."""
        low, high = self.confidence_interval
        if not 0 <= low <= 1 or not 0 <= high <= 1:
            raise ValueError("Les bornes de l'intervalle doivent être entre 0 et 1.")
        if low > high:
            raise ValueError("L'intervalle de confiance doit être ordonné.")
        return self
