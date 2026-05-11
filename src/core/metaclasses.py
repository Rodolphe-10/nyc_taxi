"""Métaclasse de validation des étapes du pipeline."""

# Imports autorisés pour cette partie.
import inspect
import time

import pandas as pd


# Métaclasse qui impose le contrat des étapes de pipeline à la définition.
class PipelineStepMeta(type):
    """Métaclasse qui valide les classes d'étapes de pipeline."""

    def __new__(mcs, name, bases, namespace):
        # On laisse passer la classe de base elle-même sans validation.
        if name == "BasePipelineStep":
            return super().__new__(mcs, name, bases, namespace)

        # On ne valide que les classes qui héritent (directement ou indirectement)
        # de BasePipelineStep.
        inherits_base_step = any(
            hasattr(base, "__mro__") and any(parent.__name__ == "BasePipelineStep" for parent in base.__mro__)
            for base in bases
        )
        if not inherits_base_step:
            return super().__new__(mcs, name, bases, namespace)

        # Vérification de l'attribut de classe "name".
        step_name = namespace.get("name")
        if not isinstance(step_name, str) or not step_name.strip():
            raise TypeError(f"[{name}] doit définir un attribut de classe 'name: str'")

        # Vérification de la présence de run() et de son caractère callable.
        run_method = namespace.get("run")
        if run_method is None or not callable(run_method):
            raise TypeError(
                f"[{name}] doit implémenter la méthode run(self, df: pd.DataFrame) -> pd.DataFrame"
            )

        # Vérification stricte de la signature run(self, df).
        params = list(inspect.signature(run_method).parameters.values())
        has_valid_signature = (
            len(params) == 2 and params[0].name == "self" and params[1].name == "df"
        )
        if not has_valid_signature:
            raise TypeError(
                f"[{name}].run() doit avoir la signature run(self, df: pd.DataFrame) -> pd.DataFrame"
            )

        return super().__new__(mcs, name, bases, namespace)


# Classe de base de toutes les étapes du pipeline.
class BasePipelineStep(metaclass=PipelineStepMeta):
    """Classe abstraite mère des étapes du pipeline."""

    name: str = ""
    version: str = "1.0.0"
    _subject = None

    @classmethod
    def set_subject(cls, subject) -> None:
        """Injecte le sujet Observer partagé par les étapes du pipeline."""
        cls._subject = subject

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Méthode métier à implémenter dans chaque sous-classe."""
        raise NotImplementedError("Implémentez run() dans la sous-classe")

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Template Method : encadre run() avec instrumentation Observer."""
        t_start = time.time()
        input_shape = df.shape

        if self._subject:
            from src.pipeline.observers import PipelineEvent

            before_event = PipelineEvent(
                step_name=self.name,
                event_type="before_run",
                timestamp=time.time(),
                input_shape=input_shape,
                output_shape=None,
                duration_ms=0.0,
                metadata={},
            )
            self._subject.notify(before_event)

        result = self.run(df)

        duration_ms = (time.time() - t_start) * 1000
        output_shape = result.shape

        if self._subject:
            from src.pipeline.observers import PipelineEvent

            after_event = PipelineEvent(
                step_name=self.name,
                event_type="after_run",
                timestamp=time.time(),
                input_shape=input_shape,
                output_shape=output_shape,
                duration_ms=duration_ms,
                metadata={},
            )
            self._subject.notify(after_event)

        return result

    def __repr__(self) -> str:
        """Représentation lisible de l'étape."""
        return f"<PipelineStep name='{self.name}' version='{self.version}'>"

    def __str__(self) -> str:
        """Nom lisible de l'étape."""
        return self.name


# Démonstration rapide du comportement de la métaclasse.
if __name__ == "__main__":
    # Cas 1 — Classe valide.
    class CleaningStep(BasePipelineStep):
        name = "cleaning_step"

        def run(self, df):
            return df

    print(f"[OK] CleaningStep créée avec succès : {CleaningStep()!r}")

    # Cas 2 — Classe sans attribut name.
    try:
        class MissingNameStep(BasePipelineStep):
            def run(self, df):
                return df
    except TypeError as exc:
        print(f"[ERREUR ATTENDUE] {exc}")

    # Cas 3 — Classe sans méthode run().
    try:
        class MissingRunStep(BasePipelineStep):
            name = "missing_run_step"
    except TypeError as exc:
        print(f"[ERREUR ATTENDUE] {exc}")

    # Cas 4 — Classe avec mauvaise signature de run().
    try:
        class WrongSignatureStep(BasePipelineStep):
            name = "wrong_signature_step"

            def run(self):
                return None
    except TypeError as exc:
        print(f"[ERREUR ATTENDUE] {exc}")
