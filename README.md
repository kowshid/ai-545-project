# Insurance Charge Predictor

An end-to-end MLOps demo that trains a regression model on the
[Kaggle Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
and serves predictions through a Streamlit web app.

The project is deliberately built around a single workflow that runs the **same
way on a Mac (M4 Air)** and on **Windows PowerShell**, through a `Makefile` +
`make.ps1` twin. Everything happens from the terminal — no point-and-click.

**Stack:** scikit-learn · MLflow · Docker · Kubernetes (minikube) · GitHub
Actions · Fly.io (free public hosting) · New Relic (optional APM).

---

## 1. What you get

- `params.yml` — single source of truth for feature ranges, model params, paths.
- `src/train.py` — trains a scikit-learn `Pipeline` (`ColumnTransformer` +
  `RandomForestRegressor`), logs everything to MLflow, saves `models/model.pkl`.
- `src/app.py` — Streamlit UI with three tabs (Dashboard, Predict Charges,
  About Model) and dataset filters in the sidebar.
- `Dockerfile` — reproducible image; the entrypoint auto-enables the New Relic
  agent when `NEW_RELIC_LICENSE_KEY` is present.
- `k8s/*.yaml` — 2-replica `Deployment` with readiness/liveness probes and a
  `NodePort` `Service`, plus an example New Relic secret.
- `.github/workflows/ci-cd.yml` — test → train → build & push to GHCR →
  deploy to Fly.io.
- `fly.toml` — Fly.io app config for the public URL.
- `Makefile` + `scripts/make.ps1` — one-liner targets for every step.

---

## 2. Repository layout

```
insurance-charge-predictor/
├── .github/workflows/ci-cd.yml
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── newrelic-secret.example.yaml
├── scripts/
│   ├── entrypoint.sh        # Docker entrypoint (adds New Relic if configured)
│   └── make.ps1             # Windows PowerShell task runner
├── src/
│   ├── __init__.py
│   ├── app.py               # Streamlit app
│   ├── train.py             # training script
│   └── utils.py             # shared helpers
├── tests/
│   └── test_model.py
├── Dockerfile
├── Makefile                 # macOS / Linux task runner
├── fly.toml                 # Fly.io deployment config
├── newrelic.ini             # New Relic Python agent config
├── params.yml               # central configuration
├── requirements.txt
└── README.md
```

---

## 3. Prerequisites

| Tool | macOS (Apple Silicon) | Windows (PowerShell) |
|---|---|---|
| Python 3.11 | `brew install python@3.11` | Install from [python.org](https://www.python.org/downloads/release/python-3119/) — check "Add to PATH" |
| Docker | [Docker Desktop for Mac (Apple Silicon)](https://docs.docker.com/desktop/install/mac-install/) | [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) |
| minikube | `brew install minikube` | `winget install Kubernetes.minikube` |
| kubectl | `brew install kubectl` | `winget install Kubernetes.kubectl` |
| flyctl (Fly.io) | `brew install flyctl` | `iwr https://fly.io/install.ps1 -useb \| iex` |
| Git | `brew install git` | `winget install Git.Git` |

Verify everything is on `PATH`:

```bash
python3 --version && docker --version && minikube version && kubectl version --client && flyctl version
```

```powershell
py -3.11 --version; docker --version; minikube version; kubectl version --client; flyctl version
```

---

## 4. First-time setup (both platforms)

```bash
# macOS / Linux
git clone https://github.com/<you>/insurance-charge-predictor.git
cd insurance-charge-predictor

make venv
source .venv/bin/activate
make install
```

```powershell
# Windows PowerShell
git clone https://github.com/<you>/insurance-charge-predictor.git
cd insurance-charge-predictor

.\scripts\make.ps1 venv
.\.venv\Scripts\Activate.ps1
.\scripts\make.ps1 install
```

> If PowerShell refuses to run the script, run this once in an admin prompt:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

---

## 5. Train the model (MLflow)

```bash
# Mac / Linux
make train
make mlflow-ui          # http://127.0.0.1:5000
```

```powershell
# Windows
.\scripts\make.ps1 train
.\scripts\make.ps1 mlflow-ui
```

What this does:

1. Downloads `insurance.csv` into `data/` (first run only).
2. Builds a `ColumnTransformer(OneHotEncoder on sex/smoker/region) → RandomForestRegressor` pipeline.
3. Logs params, R²/MAE/RMSE and the model to `./mlruns/`.
4. Writes `models/model.pkl` and `models/metrics.json` for the Streamlit app.

Re-run any time you tweak `params.yml`.

---

## 6. Run the app locally

```bash
make app
# or
streamlit run src/app.py
```

```powershell
.\scripts\make.ps1 app
```

Open <http://localhost:8501>. Without `models/model.pkl` the app shows an
**info banner** and uses the built-in formula predictor. With the pkl present,
the banner turns into a green **success** and real model predictions are served.

---

## 7. Run the tests

```bash
make test
```

```powershell
.\scripts\make.ps1 test
```

Set `SKIP_TRAIN_TEST=1` to skip the end-to-end training test (e.g. in a fast
pre-commit hook).

---

## 8. Docker

```bash
make docker-build
make docker-run                     # serves on http://localhost:8501
```

```powershell
.\scripts\make.ps1 docker-build
.\scripts\make.ps1 docker-run
```

Behind the scenes:

```bash
docker build -t insurance-app:v1 .
docker run --rm -p 8501:8501 --name insurance-app insurance-app:v1
```

The image contains the trained `model.pkl` so it can predict immediately.

---

## 9. Kubernetes (minikube)

This mirrors the lecture PDF step-by-step.

```bash
# Mac / Linux                    Windows
make k8s-start                   .\scripts\make.ps1 k8s-start
make k8s-load                    .\scripts\make.ps1 k8s-load
make k8s-apply                   .\scripts\make.ps1 k8s-apply
make k8s-url                     .\scripts\make.ps1 k8s-url
```

Expected output after `k8s-apply`:

```text
deployment.apps/insurance-app-deployment created
service/insurance-app-service created
deployment "insurance-app-deployment" successfully rolled out
```

`make k8s-url` prints the public minikube URL — open it in your browser.

**Scale up / rolling restart:**

```bash
make k8s-scale N=4
make k8s-rollout
```

```powershell
.\scripts\make.ps1 k8s-scale -N 4
.\scripts\make.ps1 k8s-rollout
```

**Clean up:**

```bash
make k8s-clean
minikube stop
```

---

## 10. Deploy to a public URL (Fly.io — free)

Fly.io is the easiest fully-CLI, free host that runs your Dockerfile as-is.
You'll get a URL like `https://insurance-charge-predictor.fly.dev`.

### One-time setup

```bash
flyctl auth signup         # or `flyctl auth login` if you have an account
flyctl auth token          # copy the token → you'll store it as a GitHub secret later
```

The first launch picks a **globally unique** app name. If
`insurance-charge-predictor` is taken, edit `fly.toml` → `app = "..."`.

```bash
make fly-launch            # same on Windows: .\scripts\make.ps1 fly-launch
make fly-deploy            #                  .\scripts\make.ps1 fly-deploy
make fly-status            #                  .\scripts\make.ps1 fly-status
```

After `fly-deploy` finishes you'll see:

```text
✔ Machine <id> is now running
Hostname:    insurance-charge-predictor.fly.dev
```

That is your **public URL**.

### New Relic on Fly.io

```bash
make fly-secrets NR_KEY=YOUR_NEW_RELIC_LICENSE_KEY
flyctl deploy                 # redeploy so the env is picked up
```

```powershell
.\scripts\make.ps1 fly-secrets -NrKey "YOUR_NEW_RELIC_LICENSE_KEY"
flyctl deploy
```

---

## 11. CI/CD with GitHub Actions

Workflow file: `.github/workflows/ci-cd.yml`

Stages:

1. **test-and-train** — installs deps, runs pytest, trains the model, uploads
   `model.pkl`, `metrics.json`, and `mlruns/` as workflow artifacts.
2. **build-and-push** — builds the Docker image (with the fresh `model.pkl`
   baked in) and pushes to `ghcr.io/<owner>/insurance-app:{sha,latest}`.
   Uses the auto-generated `GITHUB_TOKEN`, no extra secret needed.
3. **deploy** — runs `flyctl deploy --remote-only --now` when pushing to `main`.
   Requires a single repo secret: `FLY_API_TOKEN`.

### Add the Fly.io token to GitHub

```bash
flyctl auth token            # prints the token
```

Repo → **Settings → Secrets and variables → Actions → New repository secret**

- Name: `FLY_API_TOKEN`
- Value: `<the token>`

(Optional) Also add `NEW_RELIC_LICENSE_KEY` if you want the pipeline to push
it as a Fly.io secret automatically — tweak the workflow to include
`flyctl secrets set` before the deploy step.

From then on every push to `main` will:

1. retrain the model on the CI runner,
2. rebuild and push the image to GHCR,
3. deploy to your public Fly.io URL.

---

## 12. New Relic (APM)

Three ways to plug in, from simplest to most advanced:

1. **Local / Docker / Fly.io** — set `NEW_RELIC_LICENSE_KEY` at runtime.
   `scripts/entrypoint.sh` wraps Streamlit with `newrelic-admin run-program`
   automatically.

   ```bash
   # Local docker test
   docker run --rm -p 8501:8501 \
     -e NEW_RELIC_LICENSE_KEY=xxxxx \
     -e NEW_RELIC_APP_NAME=insurance-charge-predictor \
     insurance-app:v1
   ```

2. **Kubernetes** — create the secret once, the `Deployment` already wires it
   up with `optional: true` so the pod runs fine without it:

   ```bash
   kubectl create secret generic newrelic-secret \
     --from-literal=license_key=YOUR_NEW_RELIC_LICENSE_KEY
   ```

3. **Cluster-level** — install the official New Relic Helm chart:

   ```bash
   helm repo add newrelic https://helm-charts.newrelic.com
   helm install newrelic-bundle newrelic/nri-bundle \
     --set global.licenseKey=YOUR_KEY \
     --set global.cluster=my-minikube \
     --namespace newrelic --create-namespace
   ```

---

## 13. Full end-to-end run, on one screen

### macOS (Apple Silicon)

```bash
git clone https://github.com/<you>/insurance-charge-predictor.git
cd insurance-charge-predictor
make venv && source .venv/bin/activate && make install
make train
make test
make app                           # -> http://localhost:8501

# Docker sanity
make docker-build && make docker-run

# Kubernetes sanity
make k8s-start && make k8s-load && make k8s-apply
make k8s-url                       # open URL in browser
make k8s-scale N=4
make k8s-clean && minikube stop

# Public deployment
make fly-launch                    # first time only
make fly-deploy
make fly-status                    # prints https://<app>.fly.dev

# CI/CD
gh auth login                      # optional: GitHub CLI
gh secret set FLY_API_TOKEN -b"$(flyctl auth token)"
git push origin main               # CI takes it from here
```

### Windows PowerShell

```powershell
git clone https://github.com/<you>/insurance-charge-predictor.git
cd insurance-charge-predictor
.\scripts\make.ps1 venv
.\.venv\Scripts\Activate.ps1
.\scripts\make.ps1 install
.\scripts\make.ps1 train
.\scripts\make.ps1 test
.\scripts\make.ps1 app

.\scripts\make.ps1 docker-build
.\scripts\make.ps1 docker-run

.\scripts\make.ps1 k8s-start
.\scripts\make.ps1 k8s-load
.\scripts\make.ps1 k8s-apply
.\scripts\make.ps1 k8s-url
.\scripts\make.ps1 k8s-scale -N 4
.\scripts\make.ps1 k8s-clean
minikube stop

.\scripts\make.ps1 fly-launch
.\scripts\make.ps1 fly-deploy
.\scripts\make.ps1 fly-status

gh auth login
gh secret set FLY_API_TOKEN -b (flyctl auth token)
git push origin main
```

---

## 14. Troubleshooting

- **Streamlit shows the info banner even after training.** The pkl lives in
  `models/model.pkl`. Re-run `make train` and confirm the file exists.
- **`minikube image build` is slow on Apple Silicon.** That's normal the first
  time — the whole image is copied into the VM. Subsequent builds are cached.
- **Pod stuck in `ImagePullBackOff`.** The image wasn't loaded into minikube.
  Run `make k8s-load` again; the Deployment already uses
  `imagePullPolicy: IfNotPresent`.
- **`fly deploy` says app already exists.** Change `app` in `fly.toml` to a
  globally unique name and rerun `make fly-launch` (which now just updates the
  config) then `make fly-deploy`.
- **Windows: "running scripts is disabled on this system".** Run
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

---

## 15. License / attribution

Dataset: [Medical Cost Personal Dataset — mirichoi0218 on Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
(redistributed via the `stedy/Machine-Learning-with-R-datasets` mirror for
automated CI). The code in this repo is yours to use and extend.
