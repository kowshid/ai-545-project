# ============================================================
# make.ps1 — Windows PowerShell task runner
# Mirrors the Makefile targets so Windows users get the same UX.
#
# Usage:
#   .\scripts\make.ps1 <target> [-N <n>] [-NrKey <key>]
#
# Examples:
#   .\scripts\make.ps1 help
#   .\scripts\make.ps1 venv
#   .\scripts\make.ps1 install
#   .\scripts\make.ps1 train
#   .\scripts\make.ps1 app
#   .\scripts\make.ps1 api
#   .\scripts\make.ps1 docker-build
#   .\scripts\make.ps1 docker-run
#   .\scripts\make.ps1 k8s-start
#   .\scripts\make.ps1 k8s-load
#   .\scripts\make.ps1 k8s-apply
#   .\scripts\make.ps1 k8s-url
#   .\scripts\make.ps1 k8s-scale -N 4
#   .\scripts\make.ps1 fly-launch
#   .\scripts\make.ps1 fly-deploy
#   .\scripts\make.ps1 fly-secrets -NrKey "xxxxxxxxxxxx"
# ============================================================

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = "help",

    [int]$N = 0,

    [string]$NrKey = ""
)

$ErrorActionPreference = "Stop"

# Variables
$IMAGE = "insurance-app:v1"
$APP   = "src/app.py"
$PORT  = "8501"
$API_PORT = "8000"

# Helper: fail if a command is missing
function Assert-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Error "Required command '$name' is not on PATH."
    }
}

# Pick the Python launcher: prefer `py -3.11`, then `python`, then `python3`
function Get-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py -3.11 --version *> $null
            return @("py", "-3.11")
        } catch {}
        return @("py")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { return @("python") }
    if (Get-Command python3 -ErrorAction SilentlyContinue) { return @("python3") }
    Write-Error "Python 3.11 was not found. Install it from https://www.python.org/downloads/"
}

function Show-Help {
@"
Common targets:
  help              - show this help
  venv              - create a Python 3.11 virtualenv in .venv
  install           - install dependencies into the venv (uses active venv if set)
  train             - train the model (logs to MLflow)
  register          - promote best model to champion registry
  dvc-repro         - run full DVC pipeline (prepare/train/register)
  mlflow-ui         - open the MLflow UI at http://127.0.0.1:5000
  test              - run pytest
  app               - run the Streamlit app locally
  api               - run the FastAPI service locally
  docker-build      - build the Docker image ($IMAGE)
  docker-run        - run the container on port $PORT
  docker-stop       - stop the running container
  k8s-start         - start minikube
  k8s-load          - build the image inside minikube
  k8s-apply         - apply k8s/deployment.yaml + service.yaml
  k8s-url           - print the public minikube URL
  k8s-scale -N 4    - scale deployment to N replicas
  k8s-rollout       - rolling restart of the deployment
  k8s-clean         - delete k8s resources
  fly-launch        - first-time Fly.io app creation
  fly-deploy        - deploy to Fly.io
  fly-status        - show Fly.io app status / public URL
  fly-logs          - stream Fly.io logs
  fly-secrets -NrKey <key>  - push NEW_RELIC_LICENSE_KEY to Fly.io
  clean             - remove caches and local MLflow runs
"@ | Write-Host
}

function Invoke-Target($name) {
    switch ($name) {

        "help"    { Show-Help }

        "venv" {
            $py = Get-Python
            & $py[0] $py[1..($py.Count-1)] -m venv .venv
            Write-Host "Activate with:  .\.venv\Scripts\Activate.ps1"
        }

        "install" {
            $py = Get-Python
            & $py[0] $py[1..($py.Count-1)] -m pip install --upgrade pip
            & $py[0] $py[1..($py.Count-1)] -m pip install -r requirements.txt
        }

        "train" {
            $py = Get-Python
            & $py[0] $py[1..($py.Count-1)] -m src.train
        }

        "register" {
            $py = Get-Python
            & $py[0] $py[1..($py.Count-1)] -m src.register
        }

        "dvc-repro" {
            Assert-Command dvc
            dvc repro
        }

        "mlflow-ui" {
            Assert-Command mlflow
            mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
        }

        "test" {
            $py = Get-Python
            & $py[0] $py[1..($py.Count-1)] -m pytest -q
        }

        "app" {
            Assert-Command streamlit
            streamlit run $APP --server.port $PORT --server.address 0.0.0.0
        }

        "api" {
            Assert-Command uvicorn
            uvicorn src.api:app --host 0.0.0.0 --port $API_PORT
        }

        "docker-build" {
            Assert-Command docker
            docker build -t $IMAGE .
        }

        "docker-run" {
            Assert-Command docker
            docker run --rm -p "$($API_PORT):8000" --name insurance-app $IMAGE
        }

        "docker-stop" {
            Assert-Command docker
            docker stop insurance-app 2>$null | Out-Null
        }

        "k8s-start" {
            Assert-Command minikube
            minikube start
        }

        "k8s-load" {
            Assert-Command minikube
            minikube image build -t $IMAGE .
        }

        "k8s-apply" {
            Assert-Command kubectl
            kubectl apply -f k8s/deployment.yaml
            kubectl apply -f k8s/service.yaml
            kubectl rollout status deployment/insurance-app-deployment
        }

        "k8s-url" {
            Assert-Command minikube
            minikube service insurance-app-service --url
        }

        "k8s-scale" {
            if ($N -le 0) { Write-Error "Usage: .\scripts\make.ps1 k8s-scale -N 4" }
            Assert-Command kubectl
            kubectl scale deployment insurance-app-deployment --replicas=$N
            kubectl get pods
        }

        "k8s-rollout" {
            Assert-Command kubectl
            kubectl rollout restart deployment/insurance-app-deployment
            kubectl rollout status deployment/insurance-app-deployment
        }

        "k8s-clean" {
            Assert-Command kubectl
            kubectl delete -f k8s/service.yaml -ErrorAction SilentlyContinue
            kubectl delete -f k8s/deployment.yaml -ErrorAction SilentlyContinue
        }

        "fly-launch" {
            Assert-Command flyctl
            flyctl launch --no-deploy --copy-config --name insurance-charge-predictor
        }

        "fly-deploy" {
            Assert-Command flyctl
            flyctl deploy
        }

        "fly-status" {
            Assert-Command flyctl
            flyctl status
        }

        "fly-logs" {
            Assert-Command flyctl
            flyctl logs
        }

        "fly-secrets" {
            if ([string]::IsNullOrWhiteSpace($NrKey)) {
                Write-Error "Usage: .\scripts\make.ps1 fly-secrets -NrKey <key>"
            }
            Assert-Command flyctl
            flyctl secrets set NEW_RELIC_LICENSE_KEY=$NrKey NEW_RELIC_APP_NAME=insurance-charge-predictor
        }

        "clean" {
            foreach ($p in @("__pycache__", ".pytest_cache", ".mypy_cache", "mlruns")) {
                if (Test-Path $p) { Remove-Item -Recurse -Force $p }
            }
            if (Test-Path "models/metrics.json") { Remove-Item -Force "models/metrics.json" }
        }

        default {
            Write-Warning "Unknown target '$name'."
            Show-Help
            exit 1
        }
    }
}

Invoke-Target $Target
