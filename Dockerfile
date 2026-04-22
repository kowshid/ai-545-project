# ============================================================
# Insurance Charge Predictor — FastAPI container
# Runs on Hugging Face Spaces (Docker SDK), Kubernetes, and locally
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install deps first for layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code + config
COPY params.yml .
COPY src ./src

# Trained artifact + champion metadata (both may be missing at build time;
# the app falls back to demo mode if model.pkl is absent)
COPY models ./models
COPY registry ./registry

COPY newrelic.ini .

# API port for local and Kubernetes deployment
EXPOSE 8000

# Runtime environment
ENV HOME=/tmp \
    API_SERVER_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Entrypoint auto-wraps the API with the New Relic agent if
# NEW_RELIC_LICENSE_KEY is set; otherwise starts the API directly.
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]