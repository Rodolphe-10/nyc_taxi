"""Télécharge les Parquet Yellow Taxi (enveloppe de src.optim.async_io)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.optim.async_io import DEFAULT_RAW_DIR, download_year_async

logger = logging.getLogger(__name__)


def _month(value: str) -> int:
    """Valide un numéro de mois entre 1 et 12."""
    try:
        month = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"mois invalide : {value}") from exc
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"mois hors plage : {month} (attendu entre 1 et 12)")
    return month


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments. Défaut : janvier 2023."""
    parser = argparse.ArgumentParser(
        description="Télécharge les courses Yellow Taxi au format Parquet."
    )
    parser.add_argument("--year", type=int, default=2023, help="Année (défaut : 2023).")
    parser.add_argument(
        "--months",
        type=_month,
        nargs="+",
        default=[1],
        help="Mois à télécharger, de 1 à 12 (défaut : 1).",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=DEFAULT_RAW_DIR,
        help="Dossier de destination (défaut : data/raw).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Nombre de téléchargements simultanés (défaut : 4).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Délai maximum par fichier, en secondes (défaut : 1800).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Lance le téléchargement asynchrone et renvoie 0 si tous les mois réussissent."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = _ROOT / dest
    logger.info(
        "Téléchargement Yellow Taxi %s, mois %s → %s",
        args.year,
        args.months,
        dest,
    )
    results = asyncio.run(
        download_year_async(
            year=args.year,
            dest_dir=str(dest),
            months=args.months,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
        )
    )
    failed = [r for r in results if not r.get("success")]
    if failed:
        for item in failed:
            logger.error("Mois %s : %s", item.get("month"), item.get("error"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
