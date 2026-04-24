---
title: Insurance Charge Predictor
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# Insurance Charge Predictor

An end-to-end MLOps demo that trains a regression model on the classic
Kaggle **Medical Cost Personal Dataset**, tracks experiments with MLflow,
promotes a champion model via a lightweight JSON registry, and serves
predictions through a Streamlit UI — packaged as a Docker image and
deployable to **Hugging Face Spaces**, **Kubernetes**, or **Fly.io** from
the same CI pipeline.

> Status: deployed to Hugging Face Spaces via GitHub Actions on every push
> to `main`. Kubernetes manifests are included for running the same image
> on `minikube` or any managed cluster.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quickstart — Running Locally](#quickstart--running-locally)
3. [How the Project Works — Local to Prod](#how-the-project-works--local-to-prod)
4. [File-by-File Reference](#file-by-file-reference)
5. [Core MLOps Components](#core-mlops-components)
    - [Data Versioning Pipeline (DVC)](#data-versioning-pipeline-dvc)
    - [Experiment Tracking (MLflow + Dummy Registry)](#experiment-tracking-mlflow--dummy-registry)
    - [Deployment Pipeline (Docker + Streamlit UI)](#deployment-pipeline-docker--streamlit-ui)
    - [Orchestration (Kubernetes)](#orchestration-kubernetes)
    - [CI/CD Pipeline (GitHub Actions)](#cicd-pipeline-github-actions)
6. [Observability (New Relic)](#observability-new-relic)
7. [Environment Variables & Secrets](#environment-variables--secrets)
8. [Shortcomings & Known Limitations](#shortcomings--known-limitations)
9. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
.
├── Dockerfile                     # Container image definition
├── Makefile                       # macOS/Linux task runner
├── README.md
├── ci-cd.yml                      # GitHub Actions workflow
├── newrelic.ini                   # New Relic agent config
├── params.yml                     # Central project configuration
├── requirements.txt               # Pinned Python dependencies
├── .streamlit/
│   ├── config.toml                # Streamlit theme
├── data/
│   ├── insurance.csv              # Raw dataset (1,338 rows)
│   └── insurance.csv.dvc          # DVC pointer for the CSV
│
├── k8s/
│   ├── deployment.yaml            # Deployment: 2 replicas, probes, resource limits
│   ├── service.yaml               # NodePort service on 30085 → 8501
│   └── newrelic-secret.example.yaml
│
├── models/
│   ├── metrics.json               # Challenger metrics + leaderboard
│   └── model.pkl                  # Serialized sklearn pipeline (champion)
│
├── registry/
│   └── champion.json              # Dummy registry — the deployed champion
│
├── scripts/
│   ├── entrypoint.sh              # Container entrypoint 
│   └── make.ps1                   # Windows PowerShell equivalent of the Makefile
│
├── src/
│   ├── __init__.py
│   ├── app.py                     # Streamlit frontend
│   ├── register.py                # Challenger → Champion promotion
│   ├── train.py                   # Multi-model training + MLflow logging
│   └── utils.py                   # Shared helpers
│
└── tests/
    └── test_model.py              # Smoke tests
```

---

## Quickstart — Running Locally

### Prerequisites

- Python **3.11** (matches CI and the container)
- `make` (macOS/Linux). On Windows use `.\scripts\make.ps1 <target>`
- Docker
- minikube
- kubectl

### Five-command setup

```bash
make venv             # create .venv
source .venv/bin/activate
make install          # pip install -r requirements.txt
make train            # train all candidates, pick challenger, log to MLflow
make register         
make app              # launch Streamlit on http://localhost:8501
```

### Inspect experiments

```bash
make mlflow-ui        # MLflow UI at http://127.0.0.1:5000
```

You'll see one parent `training_session` run with five nested child runs
(one per candidate in `params.yml`). The parent run is tagged with
`challenger_name` and `challenger_type`.

### Run tests

```bash
make test
```

### Run the container locally

```bash
make docker-build
make docker-run       # http://localhost:8501
```

### Windows

Replace every `make <target>` with `.\scripts\make.ps1 <target>`. The
PowerShell script mirrors every target in the Makefile.

---

## How the Project Works — Local to Prod

```
┌─────────────────┐   git push    ┌──────────────────────┐
│  Local dev      │ ────────────▶ │  GitHub Actions      │
│  commit changes │               │  (ci-cd.yml)         │
│                 │               │                      │
└─────────────────┘               │  ┌────────────────┐  │
                                  │  │ 1. train       │  │
                                  │  │ 2. register    │  │
                                  │  │ 3. build-image │  │
                                  │  │ 4. deploy-hf   │  │
                                  │  └────────────────┘  │
                                  └──────────┬───────────┘
                                             │
                       ┌─────────────────────┼──────────────────────┐
                       ▼                     ▼                      ▼
             ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
             │ Hugging Face     │  │ GHCR image       │  │ (optional)       │
             │ Space            │  │ ghcr.io/<user>/  │  │ Kubernetes       │
             │ (production)     │  │ insurance-app    │  │ deployment       │
             └──────────────────┘  └──────────────────┘  └──────────────────┘
```

### Stage 1 — Local training

`python -m src.train` loads `params.yml`, downloads the insurance CSV
(cached under `data/`), splits 80/20, trains every model listed under
`training.candidates`, evaluates each on R² / MAE / RMSE, and selects
the **challenger** by `selection_metric` (default: R², higher is better).

The challenger is serialized to `models/model.pkl`; a full leaderboard
is written to `models/metrics.json`. Every candidate is logged to
MLflow as a nested run under a parent `training_session` run.

### Stage 2 — Registration

`python -m src.register` compares the fresh challenger's
`selection_metric` against the current `registry/champion.json`. The
challenger is only promoted if it wins. Otherwise the previous champion
stays in place and the new model is discarded (the `.pkl` is kept but
not referenced by `champion.json`).

### Stage 3 — Containerization

`Dockerfile` copies `src/`, `models/`, `registry/`, `params.yml`, and
`newrelic.ini` into a `python:3.11-slim` image. The champion model is
**baked into the image** at build time — the container is fully
self-contained and doesn't need to hit S3/MLflow/DVC at startup.

### Stage 4 — Deployment

The CI pipeline pushes the image to GHCR (`ghcr.io/<owner>/insurance-app`)
**and** deploys to Hugging Face Spaces by force-pushing a clean branch
to `huggingface.co/spaces/<user>/insurance-charge-predictor`. HF reads
the `Dockerfile` and rebuilds the Space automatically.

At runtime, `scripts/entrypoint.sh` checks for `NEW_RELIC_LICENSE_KEY`:

- Present → wraps Streamlit with `newrelic-admin run-program` (full APM)
- Missing → starts Streamlit directly (no telemetry)

### Stage 5 — Serving

The Streamlit app (`src/app.py`) exposes three tabs:

- **Dashboard** — filterable dataset view with histograms, scatter, and
  box plots.
- **Predict Charges** — form input → `model.predict()` → dollar amount.
  If no model is available it falls back to a hand-coded formula
  (`demo_predict` in `utils.py`), so the UI is never broken on a fresh
  clone.
- **About Model** — champion card with R²/MAE/RMSE, raw `champion.json`,
  and telemetry status.

Every page view emits an `AppPageView` event to New Relic; every
prediction emits an `InsurancePrediction` event with age/BMI/smoker/etc.
plus `predicted_charge`, `mode`, and `latency_ms`.

---

## File-by-File Reference

### Configuration

| File               | Purpose                                                                                                                                                                                                                                       |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `params.yml`       | **Single source of truth.** Dataset URL, feature ranges (age 18–64, BMI 15–55, etc.), target column, candidate models with hyperparameters, MLflow experiment name, output paths, app port. Both `train.py` and `app.py` read from this file. |
| `requirements.txt` | Pinned deps: pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2, streamlit 1.39.0, mlflow 2.17.2, matplotlib, pyyaml, requests, joblib, pytest, newrelic 10.3.0.                                                                                  |
| `config.toml`      | Streamlit theme (light base, `#1f77b4` primary). Must live at `.streamlit/config.toml` to take effect.                                                                                                                                        |
| `newrelic.ini`     | APM agent settings. License key + app name come from environment variables at runtime — nothing sensitive is committed.                                                                                                                       |

### Python source (`src/`)

| File          | Purpose                                                                                                                                                                                                                                                                                           |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `utils.py`    | Shared helpers: `project_root()`, `load_params()`, `download_data()` (cached fetch of insurance CSV), `load_dataset()` (normalizes string cols to lowercase), `load_model()`/`save_model()` (pickle), and `demo_predict()` — a hand-coded formula used as a fallback when no `.pkl` is available. |
| `train.py`    | Trains every candidate in `params.yml`, evaluates with R²/MAE/RMSE, logs each as a nested MLflow run, picks the challenger by `selection_metric`, writes `model.pkl` and `metrics.json`.                                                                                                          |
| `register.py` | Reads `metrics.json`, compares challenger vs. the existing `champion.json`, promotes only if better. Writes champion with a UTC timestamp and a `version` (epoch seconds).                                                                                                                        |
| `app.py`      | Streamlit UI. Three tabs, caches the dataset and the model via `st.cache_data` / `st.cache_resource`, records custom events to New Relic.                                                                                                                                                         |
| `__init__.py` | Marks `src/` as a package so `python -m src.train` works.                                                                                                                                                                                                                                         |

### Dev ergonomics

| File                    | Purpose                                                                                                        |
|-------------------------|----------------------------------------------------------------------------------------------------------------|
| `Makefile`              | All dev shortcuts: `venv`, `install`, `train`, `test`, `app`, `mlflow-ui`, `docker-*`, `k8s-*`, `fly-*`.       |
| `scripts/make.ps1`      | Windows PowerShell equivalent. Same targets, same semantics.                                                   |
| `scripts/entrypoint.sh` | Container entrypoint. Conditionally wraps Streamlit with the New Relic agent based on `NEW_RELIC_LICENSE_KEY`. |

### Containerization & CI/CD

| File         | Purpose                                                                                                                                                                                            |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Dockerfile` | `python:3.11-slim` base. Installs deps, copies app + model + config, exposes 8501, sets `HOME=/tmp` (HF Spaces FS constraint), adds a healthcheck against `/_stcore/health`, runs `entrypoint.sh`. |
| `ci-cd.yml`  | Four sequential GitHub Actions jobs: `train` → `register` → `build-image` → `deploy-hf`. Artifacts flow between jobs via `upload-artifact`/`download-artifact`.                                    |

### Kubernetes (`k8s/`)

| File                           | Purpose                                                                                                                                                                                 |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `deployment.yaml`              | 2 replicas, pulls image from GHCR, mounts `NEW_RELIC_LICENSE_KEY` from a Secret (optional), readiness + liveness probes on `/_stcore/health`, resources: 100m–1000m CPU, 256Mi–1Gi RAM. |
| `service.yaml`                 | `NodePort` service exposing port 8501 on node port 30085.                                                                                                                               |
| `newrelic-secret.example.yaml` | Template for the NR Secret. Real secret created via `kubectl create secret generic newrelic-secret --from-literal=license_key=...`.                                                     |

### Data & artifacts

| Path                     | Purpose                                                            |
|--------------------------|--------------------------------------------------------------------|
| `data/insurance.csv`     | Raw dataset (1,338 rows, 7 columns).                               |
| `data/insurance.csv.dvc` | DVC pointer tracking the hash of the CSV.                          |
| `models/model.pkl`       | Serialized challenger `sklearn.pipeline.Pipeline`.                 |
| `models/metrics.json`    | Challenger R²/MAE/RMSE + full leaderboard.                         |
| `registry/champion.json` | Dummy registry — the single source of truth for "what's deployed." |
| `tests/test_model.py`    | Smoke tests.                                                       |

---

## Core MLOps Components

### Data Versioning Pipeline (DVC)

The repo is **DVC-initialized**: `data/insurance.csv.dvc` is the
content-hash pointer that git tracks instead of the CSV itself. The
intent is:

- The actual `insurance.csv` stays out of git (large binary, churn-prone).
- Its hash is committed, so a checkout of any past commit maps
  deterministically to the exact dataset that was used.
- `dvc pull` would fetch the file from a configured remote (S3, GCS,
  Azure Blob, or an SSH endpoint).

**Caveat — see [Shortcomings](#shortcomings--known-limitations).** In
this project DVC is partially wired: the `.dvc` pointer exists but
`train.py` bypasses it and re-downloads from a public GitHub raw URL
defined in `params.yml`. A full DVC pipeline would add `dvc.yaml` with a
`train` stage that declares `data/insurance.csv` as a dependency and
`models/model.pkl` + `models/metrics.json` as outputs, giving you
cache-aware `dvc repro` and lineage graphs.

### Experiment Tracking (MLflow + Dummy Registry)

Two distinct layers handle "which experiments ran" vs. "which model is
deployed":

**MLflow (experiment tracking).** `train.py` uses a local file-store
backend (`./mlruns`) and logs, for every candidate in `params.yml`:

- Parameters: model type, all hyperparameters, split config.
- Metrics: R², MAE, RMSE.
- Artifacts: the fitted `sklearn.Pipeline` under `model/`.
- Tags: `challenger_name`, `challenger_type` on the parent run.

Runs are grouped under a parent `training_session` run so comparing
candidates in the MLflow UI is one click. `make mlflow-ui` starts the
UI against `./mlruns`.

**Dummy JSON registry (promotion).** Instead of using MLflow's Model
Registry (which needs a real tracking server), this project uses a
single file: `registry/champion.json`. `register.py` is the gatekeeper —
it reads `metrics.json`, compares to the existing champion, and only
overwrites the champion JSON if the new model wins on the configured
metric. The serving layer (`app.py`) reads `champion.json` to display
metadata ("R²=0.86 · MAE=$2,450 · registered 2026-04-22T17:05:12Z") but
always loads the actual weights from `models/model.pkl`.

Why this works for a demo:

- Zero infra — no tracking server, no database, no S3.
- Fully git-trackable — you can diff champion versions historically.
- Easy to swap for real MLflow Registry once you have a tracking server
  (replace the JSON read/write with `mlflow.register_model` calls).

### Deployment Pipeline (Docker + Streamlit UI)

> **Terminology note:** this project exposes a Streamlit **web UI**, not
> a REST API. There is no `/predict` HTTP endpoint you can `curl`.
> The container bundles the trained model with a web frontend.

**Dockerfile strategy** — keep the container self-contained:

1. Start from `python:3.11-slim` for a small base.
2. Install `curl` (needed by `HEALTHCHECK`), then wipe apt cache.
3. Copy `requirements.txt` first so the pip layer caches across builds.
4. Copy app code + model + registry **after** deps, so changing
   `app.py` doesn't invalidate the pip cache.
5. Set `HOME=/tmp` — Hugging Face Spaces mounts a read-only FS but
   allows writes to `/tmp`, and Streamlit/MLflow expect a writable home.
6. `HEALTHCHECK` against `/_stcore/health` — Streamlit's built-in
   health endpoint, also used by the Kubernetes probes.
7. `CMD ["/entrypoint.sh"]` — one entrypoint, two modes (NR-wrapped or
   plain Streamlit) depending on whether the license key is present.

**Image tags.** CI produces two tags on every push to `main`:

- `ghcr.io/<owner>/insurance-app:latest` — the moving head.
- `ghcr.io/<owner>/insurance-app:<commit-sha>` — immutable. Use this in
  production to avoid accidental rollbacks via `latest`.

### Orchestration (Kubernetes)

Kubernetes is an **alternative serving surface** to Hugging Face Spaces.
HF Spaces gives you one managed container — simple but with no
replicas, no rolling updates, no autoscaling. The `k8s/` directory lets
you run the same image anywhere you have a cluster:

**What the manifests provide:**

- **`deployment.yaml`** — 2 replicas behind a `Deployment` controller.
  If a pod crashes, Kubernetes replaces it. Rolling updates are
  zero-downtime (`kubectl rollout restart deployment/…`).
- **Probes** — `readinessProbe` keeps a pod out of the Service until
  it's actually healthy; `livenessProbe` restarts a pod that stops
  responding. Both hit `/_stcore/health` on port 8501.
- **Secret management** — `NEW_RELIC_LICENSE_KEY` is injected from a
  Kubernetes `Secret` via `secretKeyRef` with `optional: true`, so pods
  start cleanly even without the secret.
- **Resource governance** — `requests: 100m CPU / 256Mi RAM`,
  `limits: 1000m CPU / 1Gi RAM`. The scheduler uses requests for
  placement; the kernel enforces limits.
- **`service.yaml`** — A `NodePort` on 30085 exposes port 8501. In a
  real cluster you'd add an `Ingress` or switch to a `LoadBalancer`
  service type.

**Local workflow with minikube:**

```bash
make k8s-start                      # minikube up
make k8s-load                       # build image inside minikube
make k8s-apply                      # apply + wait for rollout
make k8s-url                        # get a browser URL
make k8s-scale N=4                  # bump replicas to 4
make k8s-rollout                    # rolling restart
make k8s-clean                      # tear down
```

For a real cluster, replace `k8s-load` with pulling from GHCR (the
image CI already published), create the NR secret with
`kubectl create secret generic newrelic-secret --from-literal=license_key=...`,
then `kubectl apply -f k8s/`.

### CI/CD Pipeline (GitHub Actions)

`ci-cd.yml` defines four sequential jobs. Each job runs in a fresh
Ubuntu runner; artifacts flow between jobs via
`actions/upload-artifact` + `actions/download-artifact`.

#### Job 1 — `train`

```
checkout → setup-python 3.11 → pip install → pytest → python -m src.train
→ upload model.pkl + metrics.json + mlruns/ as "model-bundle"
```

If `pytest` fails, the whole pipeline aborts before any artifact is
produced.

#### Job 2 — `register`

```
download "model-bundle" → python -m src.register
→ cat registry/champion.json → upload registry/champion.json as "registry"
```

This is where the champion/challenger decision happens. Note that on a
brand-new repo, the first run has no previous champion, so the
challenger is always promoted on first run.

#### Job 3 — `build-image`

```
download both artifacts → docker buildx build
→ push ghcr.io/<owner>/insurance-app:latest
→ push ghcr.io/<owner>/insurance-app:<sha>
```

Uses the built-in `GITHUB_TOKEN` to authenticate against GHCR. The
image name is lowercased defensively (`${IMAGE_NAME,,}`) because Docker
requires lowercase repo names but GitHub usernames can be mixed case.

#### Job 4 — `deploy-hf`

Three sub-steps:

1. **Sync HF Space secrets** — posts `NEW_RELIC_LICENSE_KEY` (as a
   secret) and `NEW_RELIC_APP_NAME` (as a variable) to the HF Spaces
   API. If the NR key isn't configured, this step emits a warning and
   continues.
2. **Build a clean orphan branch** — creates a `hf-deploy` branch from
   scratch, adds the code + model files with `git add -f` (the `models/`
   dir is usually gitignored locally, so the `-f` is necessary).
3. **Force-push to HF Space** — pushes `hf-deploy:main` to
   `huggingface.co/spaces/<user>/insurance-charge-predictor` using the
   HF token. HF detects the push, re-reads the Dockerfile, and rebuilds.

**Required GitHub secrets:**

| Secret                  | Purpose                                                                                   |
|-------------------------|-------------------------------------------------------------------------------------------|
| `HF_TOKEN`              | Hugging Face write token for pushing to the Space.                                        |
| `HF_USER`               | Hugging Face username that owns the Space.                                                |
| `NEW_RELIC_LICENSE_KEY` | Ingest license key (optional — pipeline continues gracefully without it).                 |
| `NEW_RELIC_APP_NAME`    | App name shown in the New Relic UI (optional — defaults to `insurance-charge-predictor`). |

### Observability (New Relic)

Instrumentation is opt-in — the project runs fine without New Relic.

Three layers of telemetry when enabled:

- **APM (automatic)** — `newrelic-admin run-program` in `entrypoint.sh`
  wraps the Streamlit process, giving you transaction traces, error
  collection, distributed tracing, and thread profiling.
- **Custom events (manual)** — `app.py` calls
  `newrelic.agent.record_custom_event("InsurancePrediction", …)` on
  every prediction with fields: age, BMI, children, sex, smoker,
  region, `predicted_charge`, `mode` (`champion` / `model` /
  `demo` / `demo_fallback`), `latency_ms`.
- **Page views** — each Streamlit render emits an `AppPageView` event
  with `has_model` and `has_champion` flags, useful for confirming
  telemetry is flowing on a freshly-deployed build.

Query examples in New Relic NRQL:

```sql
-- Prediction volume in the last hour
SELECT count(*)
FROM InsurancePrediction SINCE 1 hour ago

-- p95 latency by mode
SELECT percentile(latency_ms, 95)
FROM InsurancePrediction FACET mode

-- Smoker vs non-smoker predicted charge
SELECT average(predicted_charge)
FROM InsurancePrediction FACET smoker
```

---

## Environment Variables & Secrets

| Variable                    | Scope   | Purpose                                                                |
|-----------------------------|---------|------------------------------------------------------------------------|
| `NEW_RELIC_LICENSE_KEY`     | Runtime | Enables NR APM + custom events when set.                               |
| `NEW_RELIC_APP_NAME`        | Runtime | App name in the NR UI. Default: `insurance-charge-predictor`.          |
| `NEW_RELIC_CONFIG_FILE`     | Runtime | Path to `newrelic.ini`. Default: `/app/newrelic.ini` in the container. |
| `STREAMLIT_SERVER_PORT`     | Runtime | Defaults to 8501. HF Spaces *requires* 8501.                           |
| `STREAMLIT_SERVER_HEADLESS` | Runtime | `true` in the container — suppresses the "open browser" prompt.        |
| `HOME`                      | Runtime | Set to `/tmp` in the container because HF Spaces has a read-only FS.   |
| `HF_TOKEN`                  | CI      | Hugging Face write token (GitHub secret).                              |
| `HF_USER`                   | CI      | Hugging Face username (GitHub secret).                                 |

---

## Shortcomings & Known Limitations

Honest assessment of the gaps — most are deliberate trade-offs for a
demo project, but worth fixing for a real production system.

### Data & pipeline

- **DVC is only half-wired.** `data/insurance.csv.dvc` exists but
  there's no `dvc.yaml` defining a pipeline, no configured remote, and
  `train.py` bypasses DVC entirely by re-downloading the CSV from a
  GitHub raw URL on every run. CI has no `dvc pull` step. A real
  setup would have `dvc repro` drive training and a remote (S3, GCS)
  storing the data blobs.
- **No data validation.** There's no schema check (Pandera, Great
  Expectations) before training. If the upstream CSV ever changed
  columns or types, the failure would surface deep inside scikit-learn.
- **Fixed random seed gives false reproducibility confidence.** The
  split is reproducible, but the upstream dataset URL isn't pinned —
  if the GitHub raw file ever changes, training results shift silently.

### Modeling & registry

- **Dummy registry, not real.** `registry/champion.json` is
  git-trackable but has no model lineage, no approval workflow, no
  stage transitions (staging → production), no rollback. MLflow Model
  Registry or SageMaker Model Registry would give you all of that.
- **MLflow is file-backend-only.** `./mlruns/` is a local directory.
  It's uploaded as a CI artifact, but there's no long-running tracking
  server, so cross-run comparisons across different CI runs aren't
  possible in one UI.
- **Register script doesn't verify the model file.** It compares JSON
  metrics only. A corrupted `model.pkl` would still be promoted as
  long as `metrics.json` is valid.
- **No drift monitoring.** There's no tracking of input distribution
  shifts or prediction drift over time, and no alert if predictions
  suddenly skew.
- **Candidate space is tiny and hardcoded.** Five models, each with
  one hyperparameter configuration. No HPO (Optuna, grid search, or
  MLflow's sweep integrations).
- **Single test file.** `tests/test_model.py` is a smoke test. No
  data contract tests, no model-quality regression tests, no
  integration tests against the running container.

### Serving

- **Streamlit is a UI, not an API.** There's no REST endpoint that
  another service can POST JSON to. Adding a FastAPI `/predict` route
  alongside the UI would make this a proper prediction service —
  useful for batch scoring, mobile clients, or being called by other
  microservices.
- **Model is baked into the image.** You can't hot-swap a new model
  without rebuilding and redeploying the container. A production
  setup would load from an object store at startup (or via a sidecar)
  so promoting a model is just updating a pointer.
- **No auth on the UI.** Anyone who can reach the URL can run
  predictions. Fine for a demo, not fine for PII-sensitive inputs.
- **No rate limiting.** A single user hammering the predict button
  drives up your NR event count and, on a paid deployment, your bill.

### Kubernetes

- **Hardcoded image reference.** `k8s/deployment.yaml` has
  `ghcr.io/kowshid/insurance-app:latest` hardcoded — not templated via
  Helm or Kustomize. You can't easily deploy a different user's fork.
- **`:latest` tag in production.** The manifest uses `:latest`, which
  defeats the purpose of the immutable `:<sha>` tag CI also produces.
  Swap to a Kustomize overlay that pins `:<sha>` per environment.
- **No HorizontalPodAutoscaler.** Replicas are fixed at 2. A real
  deployment would scale on CPU or on a custom metric like
  requests-per-second.
- **NodePort, not Ingress.** Fine for minikube, not ideal for a
  real cluster. No TLS termination, no hostname routing.
- **No PodDisruptionBudget.** Cluster maintenance events (node
  drains) can take both replicas down simultaneously.
- **No NetworkPolicy.** Any pod in the cluster can talk to the
  insurance-app pods on 8501.

### CI/CD

- **HF secret sync is best-effort.** The `curl -fsS` calls to HF's
  secrets API fail silently if the HF token is missing (the job emits
  a warning and exits 0). A real pipeline would fail loudly.
- **No staging environment.** Every push to `main` goes straight to
  the production HF Space. No canary, no manual approval gate.
- **No image vulnerability scanning.** The built image isn't scanned
  with Trivy / Grype / Snyk before being promoted.
- **No SBOM.** No software bill of materials is generated.
- **CI doesn't push to K8s.** The pipeline builds and pushes the
  image but doesn't run `kubectl apply`. K8s deploys are manual.

### Observability

- **Browser instrumentation may not work.** `newrelic.ini` has
  `browser_monitoring.auto_instrument = true`, but Streamlit uses a
  custom React frontend — the auto-injection may not reach every page.
- **No structured application logs.** `print()` statements in
  training are ad-hoc; there's no JSON logging with request IDs.

---

## Troubleshooting

**`make train` fails with "No candidates configured"** — check that
`params.yml` has entries under `training.candidates`. The file loader
is strict.

**Streamlit shows "No `model.pkl` found yet"** — you haven't trained
yet. Run `make train`, then refresh. Demo mode will still work.

**HF Space won't rebuild after a push** — check the Space's build logs
in the HF UI. Most common cause: the image exceeds HF's size limit or
requires a paid runtime. This image is ~800 MB which is within limits.

**K8s pods stuck in `ImagePullBackOff`** — the GHCR package is
private by default. Either make the package public in GitHub's
repository → Packages settings, or create an `imagePullSecret` and
uncomment the relevant block in `deployment.yaml`.

**New Relic shows no data** — confirm `NEW_RELIC_LICENSE_KEY` is set
(check the Space's Settings → Variables). The `Observability` section
in the app's "About Model" tab will show whether the agent is
initialized.

**"Access denied" pushing to GHCR in CI** — make sure the workflow
has `permissions: { packages: write }` on the `build-image` job. It
does by default in this file, but a renamed job needs it explicitly.