# ============================================================
# Insurance Charge Predictor — Streamlit app container
# ============================================================
FROM python:3.11-slim

# Faster, smaller pip installs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (curl for healthchecks, build tools kept minimal)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code and configuration
COPY params.yml .
COPY src ./src
# The trained pipeline. This file is optional at build time —
# if it is missing, the app falls back to demo mode automatically.
COPY models ./models
COPY newrelic.ini .

EXPOSE 8501

# Streamlit needs a writable config dir for its usage stats
ENV HOME=/tmp \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Healthcheck for Kubernetes / Fly.io
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Entry script adds optional New Relic instrumentation if
# NEW_RELIC_LICENSE_KEY is provided, otherwise starts Streamlit directly.
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
