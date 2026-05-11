"""Tests unitaires du pattern Observer appliqué au pipeline."""

import time

import pandas as pd

from src.core.metaclasses import BasePipelineStep
from src.pipeline.observers import (
    AlertObserver,
    LoggingObserver,
    MetricsObserver,
    PipelineEvent,
    PipelineObserver,
    PipelineSubject,
)


def test_pipeline_subject_attach_detach(tmp_path):
    subject = PipelineSubject()
    obs = LoggingObserver(log_file=str(tmp_path / "pipeline.log"))
    subject.attach(obs)
    assert subject.observer_count() == 1
    subject.detach(obs)
    assert subject.observer_count() == 0


def test_metrics_observer_collects_data():
    metrics_obs = MetricsObserver()
    event1 = PipelineEvent("step_a", "after_run", time.time(), (100, 3), (90, 3), 12.5, {})
    event2 = PipelineEvent("step_b", "after_run", time.time(), (90, 3), (80, 4), 20.0, {})

    metrics_obs.update(event1)
    metrics_obs.update(event2)
    df = metrics_obs.summary()

    assert len(df) == 2
    assert {"step", "duration_ms", "rows_dropped"}.issubset(df.columns)


def test_alert_observer_triggers():
    alert_obs = AlertObserver(threshold_pct=30.0)
    event = PipelineEvent("clean", "after_run", time.time(), (100, 2), (50, 2), 5.0, {})

    alert_obs.update(event)
    assert alert_obs.has_alerts() is True
    assert alert_obs.alerts()[0]["pct_lost"] > 30.0


def test_alert_observer_no_trigger():
    alert_obs = AlertObserver(threshold_pct=30.0)
    event = PipelineEvent("clean", "after_run", time.time(), (100, 2), (95, 2), 5.0, {})

    alert_obs.update(event)
    assert alert_obs.has_alerts() is False


def test_notify_calls_all_observers():
    class CountingObserver(PipelineObserver):
        def __init__(self):
            self.calls = 0

        def update(self, event: PipelineEvent) -> None:
            self.calls += 1

    subject = PipelineSubject()
    o1, o2, o3 = CountingObserver(), CountingObserver(), CountingObserver()
    subject.attach(o1)
    subject.attach(o2)
    subject.attach(o3)

    event = PipelineEvent("step", "after_run", time.time(), (10, 1), (10, 1), 1.0, {})
    subject.notify(event)

    assert o1.calls == 1
    assert o2.calls == 1
    assert o3.calls == 1


def test_duplicate_attach_warning(tmp_path):
    subject = PipelineSubject()
    obs = LoggingObserver(log_file=str(tmp_path / "pipeline.log"))
    subject.attach(obs)
    subject.attach(obs)
    assert subject.observer_count() == 1


def test_execute_notifies_subject():
    class FakeStep(BasePipelineStep):
        name = "fake_step"

        def run(self, df):
            return df

    subject = PipelineSubject()
    metrics_obs = MetricsObserver()
    subject.attach(metrics_obs)
    BasePipelineStep.set_subject(subject)

    data = pd.DataFrame({"x": [1, 2, 3]})
    result = FakeStep().execute(data)

    assert isinstance(result, pd.DataFrame)
    assert len(metrics_obs.summary()) >= 1
    BasePipelineStep.set_subject(None)


def test_metrics_observer_slowest_step():
    metrics_obs = MetricsObserver()
    metrics_obs.update(PipelineEvent("step_fast", "after_run", time.time(), (10, 2), (10, 2), 5.0, {}))
    metrics_obs.update(PipelineEvent("step_slow", "after_run", time.time(), (10, 2), (9, 2), 30.0, {}))
    metrics_obs.update(PipelineEvent("step_mid", "after_run", time.time(), (9, 2), (9, 2), 10.0, {}))

    assert metrics_obs.slowest_step() == "step_slow"
