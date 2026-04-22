#!/usr/bin/env bash
# Entrypoint that starts Streamlit, optionally wrapped with
# the New Relic Python agent if a license key is provided.
set -euo pipefail

APP_PATH="src/app.py"
PORT="${STREAMLIT_SERVER_PORT:-8501}"

if [[ -n "${NEW_RELIC_LICENSE_KEY:-}" ]]; then
  echo "[entrypoint] NEW_RELIC_LICENSE_KEY detected — enabling New Relic agent"
  export NEW_RELIC_CONFIG_FILE="${NEW_RELIC_CONFIG_FILE:-/app/newrelic.ini}"
  export NEW_RELIC_APP_NAME="${NEW_RELIC_APP_NAME:-insurance-charge-predictor}"
  export NEW_RELIC_LOG="${NEW_RELIC_LOG:-stderr}"
  export NEW_RELIC_LOG_LEVEL="${NEW_RELIC_LOG_LEVEL:-info}"
  exec newrelic-admin run-program \
    streamlit run "$APP_PATH" \
      --server.port "$PORT" \
      --server.address 0.0.0.0
else
  echo "[entrypoint] No NEW_RELIC_LICENSE_KEY — starting without New Relic"
  exec streamlit run "$APP_PATH" \
    --server.port "$PORT" \
    --server.address 0.0.0.0
fi