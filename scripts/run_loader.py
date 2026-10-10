"""Charge janvier 2023 via LoaderFactory et affiche df.info() (exercice 1.4)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.loaders import LoaderFactory

logger = logging.getLogger(__name__)

DEFAULT_PARQUET = _ROOT / "data" / "raw" / "yellow_tripdata_2023-01.parquet"


def main() -> int:
    """Charge le Parquet de janvier 2023 et affiche son schéma."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not DEFAULT_PARQUET.is_file():
        logger.error(
            "Fichier introuvable : %s. "
            "Téléchargez-le avec : python scripts/download_data.py --months 1",
            DEFAULT_PARQUET,
        )
        return 1

    loader = LoaderFactory.create(str(DEFAULT_PARQUET))
    df = loader.run()
    df.info()
    logger.info("%s lignes, %s colonnes", f"{len(df):,}", df.shape[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
