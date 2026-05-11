"""Pattern Observer pour instrumenter les exécutions du pipeline."""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


# Événement standard envoyé par les étapes du pipeline aux observateurs.
@dataclass
class PipelineEvent:
    """Représente un point de vie d'une étape (avant/après exécution)."""

    step_name: str
    event_type: str
    timestamp: float
    input_shape: tuple
    output_shape: tuple
    duration_ms: float
    metadata: Dict = field(default_factory=dict)

    def __str__(self):
        return (
            f"[{self.event_type.upper()}] {self.step_name} | "
            f"in={self.input_shape} out={self.output_shape} | {self.duration_ms:.1f}ms"
        )


# Interface Observer : chaque observateur implémente update(event).
class PipelineObserver(ABC):
    """Contrat de base des observateurs du pipeline."""

    @abstractmethod
    def update(self, event: PipelineEvent) -> None:
        """Traite un événement émis par le pipeline."""

    def on_pipeline_start(self, n_steps: int) -> None:
        """Hook optionnel appelé au démarrage du pipeline."""

    def on_pipeline_end(self, total_duration_ms: float) -> None:
        """Hook optionnel appelé à la fin du pipeline."""


# Observer de log : écrit les événements dans un fichier + console.
class LoggingObserver(PipelineObserver):
    """Observer chargé de journaliser l'activité du pipeline."""

    def __init__(self, log_file="pipeline.log", level=logging.INFO):
        self._logger = logging.getLogger(f"PipelineLogger:{log_file}")
        self._logger.setLevel(level)
        self._logger.propagate = False

        # Évite la duplication des handlers si plusieurs instances réutilisent le même logger.
        if not self._logger.handlers:
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
            self._logger.addHandler(stream_handler)

    def update(self, event: PipelineEvent) -> None:
        if event.event_type == "after_run":
            self._logger.info(str(event))
        elif event.event_type == "before_run":
            self._logger.debug("Démarrage de '%s'...", event.step_name)

    def on_pipeline_start(self, n_steps) -> None:
        self._logger.info("Pipeline démarré — %s étapes", n_steps)

    def on_pipeline_end(self, total_duration_ms) -> None:
        self._logger.info("Pipeline terminé en %.1fms", total_duration_ms)


# Observer de métriques : conserve les mesures de performance en mémoire.
class MetricsObserver(PipelineObserver):
    """Observer qui collecte les métriques techniques des étapes."""

    def __init__(self):
        self._metrics: List[Dict] = []
        self._pipeline_start: float = None
        self._global_summary: Dict[str, Any] = {}

    def update(self, event: PipelineEvent) -> None:
        if event.event_type != "after_run":
            return

        input_rows = event.input_shape[0] if event.input_shape else 0
        output_rows = event.output_shape[0] if event.output_shape else 0
        input_cols = event.input_shape[1] if event.input_shape and len(event.input_shape) >= 2 else None
        output_cols = event.output_shape[1] if event.output_shape and len(event.output_shape) >= 2 else None
        rows_dropped = input_rows - output_rows
        pct_rows_kept = (output_rows / input_rows * 100) if input_rows > 0 else 0.0

        self._metrics.append(
            {
                "step": event.step_name,
                "input_rows": input_rows,
                "output_rows": output_rows,
                "input_cols": input_cols,
                "output_cols": output_cols,
                "duration_ms": event.duration_ms,
                "rows_dropped": rows_dropped,
                "pct_rows_kept": pct_rows_kept,
                "timestamp": event.timestamp,
            }
        )

    def on_pipeline_start(self, n_steps) -> None:
        self._pipeline_start = time.time()
        self._global_summary = {"n_steps": n_steps}

    def on_pipeline_end(self, total_duration_ms) -> None:
        self._global_summary["pipeline_duration_ms"] = total_duration_ms
        self._global_summary["steps_recorded"] = len(self._metrics)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self._metrics)

    def total_duration_ms(self) -> float:
        return float(sum(metric["duration_ms"] for metric in self._metrics))

    def slowest_step(self) -> str:
        if not self._metrics:
            return ""
        return max(self._metrics, key=lambda m: m["duration_ms"])["step"]

    def to_json(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self._metrics, file, ensure_ascii=False, indent=2)


# Observer d'alerte : détecte les pertes de lignes anormales entre étapes.
class AlertObserver(PipelineObserver):
    """Observer qui déclenche des alertes sur chute de volume."""

    def __init__(self, threshold_pct=30.0, alert_callback=None):
        self.threshold_pct = threshold_pct
        self.alert_callback = alert_callback
        self._alerts: List[Dict] = []
        self._logger = logging.getLogger(self.__class__.__name__)

    def update(self, event: PipelineEvent) -> None:
        if event.event_type != "after_run":
            return

        input_rows = event.input_shape[0] if event.input_shape else 0
        output_rows = event.output_shape[0] if event.output_shape else 0
        if input_rows <= 0:
            return

        pct_lost = (input_rows - output_rows) / input_rows * 100
        if pct_lost > self.threshold_pct:
            alert = {
                "step": event.step_name,
                "pct_lost": pct_lost,
                "input_rows": input_rows,
                "output_rows": output_rows,
                "timestamp": event.timestamp,
            }
            self._alerts.append(alert)
            self._logger.warning(
                "ALERTE : '%s' a supprimé %.1f%% des lignes !",
                event.step_name,
                pct_lost,
            )
            if self.alert_callback:
                self.alert_callback(alert)

    def has_alerts(self) -> bool:
        return len(self._alerts) > 0

    def alerts(self) -> List[Dict]:
        return self._alerts


# Sujet du pattern Observer : gère la liste des observateurs et les notifications.
class PipelineSubject:
    """Sujet central qui notifie tous les observateurs inscrits."""

    def __init__(self):
        self._observers: List[PipelineObserver] = []
        self._logger = logging.getLogger(self.__class__.__name__)

    def attach(self, observer: PipelineObserver) -> None:
        if observer in self._observers:
            self._logger.warning("Observateur déjà attaché")
            return
        self._observers.append(observer)

    def detach(self, observer: PipelineObserver) -> None:
        if observer not in self._observers:
            self._logger.warning("Observateur non trouvé")
            return
        self._observers.remove(observer)

    def notify(self, event: PipelineEvent) -> None:
        for observer in list(self._observers):
            try:
                observer.update(event)
            except Exception as exc:  # pragma: no cover - robustesse runtime
                self._logger.error("Échec d'un observateur: %s", exc)

    def notify_start(self, n_steps: int) -> None:
        for observer in list(self._observers):
            try:
                observer.on_pipeline_start(n_steps)
            except Exception as exc:  # pragma: no cover - robustesse runtime
                self._logger.error("Échec on_pipeline_start observateur: %s", exc)

    def notify_end(self, total_duration_ms: float) -> None:
        for observer in list(self._observers):
            try:
                observer.on_pipeline_end(total_duration_ms)
            except Exception as exc:  # pragma: no cover - robustesse runtime
                self._logger.error("Échec on_pipeline_end observateur: %s", exc)

    def observer_count(self) -> int:
        return len(self._observers)


if __name__ == "__main__":
    from src.core.metaclasses import BasePipelineStep

    # Étape fictive 1 : pas de transformation.
    class FakeStep1(BasePipelineStep):
        name = "chargement"

        def run(self, df):
            return df

    # Étape fictive 2 : simule une suppression de 50% des lignes.
    class FakeStep2(BasePipelineStep):
        name = "nettoyage"

        def run(self, df):
            return df.head(int(len(df) * 0.5))

    # Étape fictive 3 : pas de transformation.
    class FakeStep3(BasePipelineStep):
        name = "encodage"

        def run(self, df):
            return df

    subject = PipelineSubject()
    logging_obs = LoggingObserver()
    metrics_obs = MetricsObserver()
    alert_obs = AlertObserver(threshold_pct=30.0)

    subject.attach(logging_obs)
    subject.attach(metrics_obs)
    subject.attach(alert_obs)

    BasePipelineStep.set_subject(subject)

    df = pd.DataFrame({"col": range(10000)})
    steps = [FakeStep1(), FakeStep2(), FakeStep3()]

    subject.notify_start(len(steps))
    t0 = time.time()
    for step in steps:
        df = step.execute(df)
    subject.notify_end((time.time() - t0) * 1000)

    print(metrics_obs.summary())
    print(f"Étape la plus lente : {metrics_obs.slowest_step()}")
    print(f"Alertes : {alert_obs.alerts()}")
