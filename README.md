## Insurance Charge Predictor - End-to-End MLOps Pipeline

This project implements a complete MLOps workflow for a non-trivial regression problem:
predicting medical insurance charges from demographic and lifestyle features.

Core capabilities included:

- DVC data + pipeline versioning
- Multi-model training (3 algorithms)
- Hyperparameter tuning and model comparison
- MLflow experiment tracking
- Champion model registry (`registry/champion.json`)
- FastAPI inference service in Docker
- Kubernetes deployment manifests
- GitHub Actions CI/CD automation

## Problem and Dataset

- **Problem type:** Regression
- **Target:** `charges`
- **Dataset:** Medical insurance dataset (`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`)
- **Source URL:** public mirror configured in `params.yml`

## Repository Structure

```text
.
├── .github/workflows/ci-cd.yml
├── .dvc/config
├── dvc.yaml
├── data/
├── k8s/
├── models/
├── registry/
├── scripts/
├── src/
│   ├── api.py          # FastAPI inference service
│   ├── app.py          # Streamlit dashboard (optional UI)
│   ├── train.py        # train/tune/compare 3 candidate models
│   ├── register.py     # promote winner to champion registry
│   └── utils.py
├── tests/test_model.py
├── Dockerfile
├── Makefile
├── params.yml
└── requirements.txt
```

## Model Development

`src/train.py` trains and tunes at least 3 algorithms:

1. `RandomForestRegressor`
2. `DecisionTreeRegressor`
3. `AdaBoostRegressor`

For each model:

- `GridSearchCV` performs hyperparameter tuning.
- MLflow logs params and metrics.
- Test metrics include `R2`, `MAE`, `RMSE`.

Artifacts produced:

- `models/candidates/<model>.pkl`
- `models/metrics_comparison.json` (leaderboard)
- `models/metrics.json` (winner metrics)
- `models/model.pkl` (best model for inference)

## DVC Pipeline

DVC stages are defined in `dvc.yaml`:

1. `prepare_data` - download and prepare dataset
2. `train` - train/tune/compare models
3. `register` - promote champion

Run:

```bash
dvc repro
```

## Champion Registry

`src/register.py` reads `models/metrics_comparison.json` and updates
`registry/champion.json` only when the new top model improves champion `R2`.

## FastAPI Inference Service

`src/api.py` exposes:

- `GET /health`
- `POST /predict`

The service automatically loads `models/model.pkl` (champion model).

Run locally:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42,
    "sex": "female",
    "bmi": 26.4,
    "children": 2,
    "smoker": "no",
    "region": "northeast"
  }'
```

## Docker

Build and run:

```bash
docker build -t insurance-app:v1 .
docker run --rm -p 8000:8000 insurance-app:v1
```

## Kubernetes (Minikube)

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
minikube service insurance-app-service --url
```

Default container/service port: `8000`

## CI/CD (GitHub Actions)

Workflow in `.github/workflows/ci-cd.yml` does:

1. Install dependencies and run tests
2. Run `dvc repro` to reproduce full data->train->register pipeline
3. Upload model + metrics + registry artifacts
4. Build and push Docker image to GHCR

## Local Quick Start

```bash
make venv
source .venv/bin/activate
make install
make dvc-repro
make api
```

Optional:

- `make app` for Streamlit dashboard
- `make mlflow-ui` for experiment UI

## Notes

- New Relic is optional; set `NEW_RELIC_LICENSE_KEY` to enable agent wrapping.
- Secret manifests under `k8s/` use placeholders only.