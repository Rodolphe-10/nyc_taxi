# syntax=docker/dockerfile:1
# Étape 1 — dépendances de l'API
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user --timeout 300 --retries 15 -r requirements-api.txt
COPY deploy/slim_site_packages.py /tmp/slim_site_packages.py
RUN python /tmp/slim_site_packages.py

# Étape 2 — exécution, utilisateur non root
FROM python:3.11-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser \
    reports/preprocessor.joblib \
    reports/model.txt \
    reports/quantile_10.txt \
    reports/quantile_90.txt \
    reports/registry.json \
    ./reports/
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
