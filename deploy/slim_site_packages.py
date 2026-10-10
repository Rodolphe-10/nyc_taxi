"""Allège le site-packages de l'image API.

LightGBM importe ``scipy.sparse`` au chargement, mais une course au format
tableau dense ne construit jamais de matrice creuse. Un module de remplacement
suffit pour l'inférence et retire environ 140 Mo.
"""

from __future__ import annotations

import shutil
import site
from pathlib import Path

_STUB = '''\
"""Remplacement minimal de scipy.sparse pour une prédiction dense."""


class spmatrix:
    """Type de matrice creuse, jamais produit par l'API."""


class csr_matrix(spmatrix):
    """Matrice CSR factice."""


class csc_matrix(spmatrix):
    """Matrice CSC factice."""


def hstack(*_args: object, **_kwargs: object) -> None:
    """Les matrices creuses ne sont pas servies par cette image."""
    raise RuntimeError("scipy est absent de l'image API.")
'''


def main() -> None:
    """Supprime SciPy et les tests Pandas, puis pose le module factice."""
    root = Path(site.getusersitepackages())
    shutil.rmtree(root / "scipy", ignore_errors=True)
    shutil.rmtree(root / "scipy.libs", ignore_errors=True)
    for dist in root.glob("scipy-*.dist-info"):
        shutil.rmtree(dist, ignore_errors=True)
    shutil.rmtree(root / "pandas" / "tests", ignore_errors=True)
    sparse = root / "scipy" / "sparse"
    sparse.mkdir(parents=True)
    (root / "scipy" / "__init__.py").write_text("", encoding="utf-8")
    (sparse / "__init__.py").write_text(_STUB, encoding="utf-8")


if __name__ == "__main__":
    main()
