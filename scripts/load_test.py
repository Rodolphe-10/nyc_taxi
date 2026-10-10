"""Mesure la latence de POST /predict.

Commande, depuis la racine du dépôt, API déjà démarrée :

    python scripts/load_test.py --url http://127.0.0.1:8000/predict --rps 100 --seconds 5

Le script tente d'émettre le débit demandé. Le débit réellement atteint
et les latences sont écrits dans reports/load_test.json.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
_BODY = json.dumps(_TRIP).encode("utf-8")
_ROOT = Path(__file__).resolve().parent.parent


def _one(url: str) -> tuple[float, int]:
    request = urllib.request.Request(
        url,
        data=_BODY,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        exc.read()
        status = exc.code
    return time.perf_counter() - started, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--rps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--output", default="reports/load_test.json")
    args = parser.parse_args()

    total = int(args.rps * args.seconds)
    interval = 1.0 / args.rps
    latencies: list[float] = []
    statuses: dict[str, int] = {}
    started = time.perf_counter()
    next_send = started
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for _ in range(total):
            now = time.perf_counter()
            if next_send > now:
                time.sleep(next_send - now)
            futures.append(pool.submit(_one, args.url))
            next_send += interval
        for future in as_completed(futures):
            latency, status = future.result()
            latencies.append(latency)
            key = str(status)
            statuses[key] = statuses.get(key, 0) + 1
    elapsed = time.perf_counter() - started
    ordered = sorted(latencies)

    def percentile(q: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    report = {
        "url": args.url,
        "requested_rps": args.rps,
        "seconds": args.seconds,
        "requests": len(latencies),
        "elapsed_s": elapsed,
        "achieved_rps": len(latencies) / elapsed if elapsed else 0.0,
        "latency_p50_s": percentile(0.50),
        "latency_p95_s": percentile(0.95),
        "latency_p99_s": percentile(0.99),
        "statuses": statuses,
        "command": (
            f"python scripts/load_test.py --url {args.url} "
            f"--rps {args.rps} --seconds {args.seconds}"
        ),
    }
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = _ROOT / destination
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
