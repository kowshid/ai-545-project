#!/usr/bin/env bash
# Entrypoint that starts FastAPI, optionally wrapped with
# the New Relic Python agent if a license key is provided.
set -euo pipefail

APP_MODULE="src.api:app"
PORT="${API_SERVER_PORT:-8000}"

if [[ -n "${NEW_RELIC_LICENSE_KEY:-}" ]]; then
  echo "[entrypoint] NEW_RELIC_LICENSE_KEY detected — enabling New Relic agent"
  export NEW_RELIC_CONFIG_FILE="${NEW_RELIC_CONFIG_FILE:-/app/newrelic.ini}"
  export NEW_RELIC_APP_NAME="${NEW_RELIC_APP_NAME:-insurance-charge-predictor}"
  export NEW_RELIC_LOG="${NEW_RELIC_LOG:-stderr}"
  exec newrelic-admin run-program \
    uvicorn "$APP_MODULE" \
      --host 0.0.0.0 \
      --port "$PORT"
else
  echo "[entrypoint] No NEW_RELIC_LICENSE_KEY — starting without New Relic"
  exec uvicorn "$APP_MODULE" \
    --host 0.0.0.0 \
    --port "$PORT"
fi
