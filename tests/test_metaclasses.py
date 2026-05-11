"""Tests unitaires de la métaclasse PipelineStepMeta."""

import pytest

from src.core.metaclasses import BasePipelineStep


# Vérifie qu'une étape correcte est acceptée et bien représentée.
def test_valid_step_creation():
    class ValidStep(BasePipelineStep):
        name = "valid_step"

        def run(self, df):
            return df

    instance = ValidStep()
    assert "valid_step" in repr(instance)
    assert str(instance) == "valid_step"


# Vérifie qu'une étape sans name est rejetée avec le bon message.
def test_missing_name_raises_typeerror():
    with pytest.raises(
        TypeError,
        match=r"^\[MissingNameStep\] doit définir un attribut de classe 'name: str'$",
    ):
        class MissingNameStep(BasePipelineStep):
            def run(self, df):
                return df


# Vérifie qu'une étape sans run() est rejetée.
def test_missing_run_raises_typeerror():
    with pytest.raises(
        TypeError,
        match=r"^\[MissingRunStep\] doit implémenter la méthode run\(self, df: pd\.DataFrame\) -> pd\.DataFrame$",
    ):
        class MissingRunStep(BasePipelineStep):
            name = "missing_run_step"


# Vérifie qu'une étape avec une mauvaise signature run() est rejetée.
def test_wrong_run_signature_raises_typeerror():
    with pytest.raises(
        TypeError,
        match=r"^\[WrongRunSignatureStep\]\.run\(\) doit avoir la signature run\(self, df: pd\.DataFrame\) -> pd\.DataFrame$",
    ):
        class WrongRunSignatureStep(BasePipelineStep):
            name = "wrong_signature_step"

            def run(self):
                return None


# Vérifie que l'implémentation de base de run() lève bien NotImplementedError.
def test_run_raises_not_implemented():
    class CleaningStep(BasePipelineStep):
        name = "cleaning_step"
        run = BasePipelineStep.run

    step = CleaningStep()
    with pytest.raises(NotImplementedError, match=r"^Implémentez run\(\) dans la sous-classe$"):
        step.run(None)


# Vérifie que plusieurs étapes valides coexistent correctement.
def test_multiple_valid_steps():
    class LoaderStep(BasePipelineStep):
        name = "loader_step"

        def run(self, df):
            return df

    class EncoderStep(BasePipelineStep):
        name = "encoder_step"

        def run(self, df):
            return df

    class TransformerStep(BasePipelineStep):
        name = "transformer_step"

        def run(self, df):
            return df

    assert LoaderStep.name == "loader_step"
    assert EncoderStep.name == "encoder_step"
    assert TransformerStep.name == "transformer_step"
    assert issubclass(LoaderStep, BasePipelineStep)
    assert issubclass(EncoderStep, BasePipelineStep)
    assert issubclass(TransformerStep, BasePipelineStep)
