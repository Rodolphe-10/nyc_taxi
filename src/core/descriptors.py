"""Descripteurs de validation pour la configuration du pipeline."""


# Classe de base qui implémente le protocole descripteur.
class Validator:
    """Descripteur de base pour valider une valeur à l'affectation."""

    # Enregistre les noms public/privé de l'attribut lors de la définition.
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    # Retourne le descripteur depuis la classe, ou la valeur depuis l'instance.
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    # Valide puis stocke la valeur dans l'attribut privé de l'instance.
    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    # Méthode à implémenter par les sous-classes.
    def validate(self, value):
        raise NotImplementedError

    # Représentation pratique du descripteur.
    def __repr__(self):
        public_name = getattr(self, "public_name", "?")
        return f"<{self.__class__.__name__}({public_name})>"


# Valide qu'une valeur numérique est strictement positive selon un seuil.
class Positive(Validator):
    """Valide qu'une valeur est un nombre > min_value."""

    # Initialise le seuil minimal strict.
    def __init__(self, min_value=0):
        self.min_value = min_value

    # Vérifie le type numérique puis la contrainte de seuil.
    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{self.public_name} doit être un nombre, reçu {type(value).__name__}"
            )
        if value <= self.min_value:
            raise ValueError(
                f"{self.public_name} doit être > {self.min_value}, reçu {value}"
            )


# Valide qu'une valeur appartient à une liste de choix autorisés.
class OneOf(Validator):
    """Valide qu'une valeur appartient à un ensemble de choix."""

    # Stocke les choix permis.
    def __init__(self, *choices):
        self.choices = choices

    # Vérifie l'appartenance de la valeur aux choix déclarés.
    def validate(self, value):
        if value not in self.choices:
            raise ValueError(
                f"{self.public_name} doit être parmi {self.choices}, reçu {value!r}"
            )


# Valide strictement le type attendu pour la valeur.
class TypedAttr(Validator):
    """Valide le type d'une valeur avec isinstance."""

    # Stocke le type attendu.
    def __init__(self, expected_type):
        self.expected_type = expected_type

    # Vérifie que la valeur est du type attendu.
    def validate(self, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.public_name} doit être de type {self.expected_type.__name__}, "
                f"reçu {type(value).__name__}"
            )


# Valide un nombre flottant borné dans un intervalle fermé.
class BoundedFloat(Validator):
    """Valide qu'une valeur numérique est dans [low, high]."""

    # Vérifie la cohérence des bornes puis les stocke.
    def __init__(self, low, high):
        if low >= high:
            raise ValueError("low doit être strictement inférieur à high")
        self.low = low
        self.high = high

    # Vérifie le type numérique puis l'appartenance à l'intervalle.
    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{self.public_name} doit être un nombre, reçu {type(value).__name__}"
            )
        if not (self.low <= value <= self.high):
            raise ValueError(
                f"{self.public_name} doit être dans [{self.low}, {self.high}], reçu {value}"
            )


# Valide une chaîne non vide (après trim) et non None.
class NotEmpty(Validator):
    """Valide qu'une chaîne de caractères n'est pas vide."""

    # Vérifie que la valeur est une chaîne non vide après suppression des espaces.
    def validate(self, value):
        if not isinstance(value, str) or len(value.strip()) == 0:
            raise ValueError(f"{self.public_name} ne peut pas être vide")


# Configuration de modèle basée sur les descripteurs de validation.
class ModelConfig:
    """Encapsule les hyperparamètres du modèle avec validations automatiques."""

    learning_rate = BoundedFloat(0.0, 1.0)
    n_estimators = Positive(min_value=10)
    objective = OneOf("regression", "binary", "multiclass")
    max_depth = TypedAttr(int)
    model_name = NotEmpty()
    random_state = Positive(min_value=0)

    # Affecte tous les champs via les descripteurs.
    def __init__(
        self,
        learning_rate,
        n_estimators,
        objective,
        max_depth,
        model_name,
        random_state=42,
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.max_depth = max_depth
        self.model_name = model_name
        self.random_state = random_state

    # Représentation lisible de la configuration complète.
    def __repr__(self):
        return (
            "ModelConfig("
            f"learning_rate={self.learning_rate}, "
            f"n_estimators={self.n_estimators}, "
            f"objective={self.objective!r}, "
            f"max_depth={self.max_depth}, "
            f"model_name={self.model_name!r}, "
            f"random_state={self.random_state})"
        )

    # Convertit la configuration en dictionnaire pour le logging.
    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "objective": self.objective,
            "max_depth": self.max_depth,
            "model_name": self.model_name,
            "random_state": self.random_state,
        }


# Démonstration rapide du fonctionnement des descripteurs.
if __name__ == "__main__":
    # Cas 1 — Configuration valide.
    cfg = ModelConfig(
        learning_rate=0.05,
        n_estimators=500,
        objective="regression",
        max_depth=6,
        model_name="LightGBM_v1",
    )
    print(cfg)
    print(cfg.to_dict())

    # Cas 2 — learning_rate hors bornes.
    try:
        ModelConfig(
            learning_rate=1.5,
            n_estimators=500,
            objective="regression",
            max_depth=6,
            model_name="LightGBM_v1",
        )
    except ValueError as exc:
        print(f"[ERREUR ATTENDUE] {exc}")

    # Cas 3 — n_estimators trop petit.
    try:
        ModelConfig(
            learning_rate=0.05,
            n_estimators=5,
            objective="regression",
            max_depth=6,
            model_name="LightGBM_v1",
        )
    except ValueError as exc:
        print(f"[ERREUR ATTENDUE] {exc}")

    # Cas 4 — objective invalide.
    try:
        ModelConfig(
            learning_rate=0.05,
            n_estimators=500,
            objective="clustering",
            max_depth=6,
            model_name="LightGBM_v1",
        )
    except ValueError as exc:
        print(f"[ERREUR ATTENDUE] {exc}")

    # Cas 5 — max_depth de mauvais type.
    try:
        ModelConfig(
            learning_rate=0.05,
            n_estimators=500,
            objective="regression",
            max_depth=6.5,
            model_name="LightGBM_v1",
        )
    except TypeError as exc:
        print(f"[ERREUR ATTENDUE] {exc}")

    # Cas 6 — model_name vide.
    try:
        ModelConfig(
            learning_rate=0.05,
            n_estimators=500,
            objective="regression",
            max_depth=6,
            model_name="",
        )
    except ValueError as exc:
        print(f"[ERREUR ATTENDUE] {exc}")
