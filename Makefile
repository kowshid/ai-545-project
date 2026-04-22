# ============================================================
# Makefile — Insurance Charge Predictor
# Works on macOS (Apple Silicon) and Linux.
# Windows users: run `.\scripts\make.ps1 <target>` in PowerShell
# ============================================================

PYTHON ?= python3
PIP    ?= pip
IMAGE  ?= insurance-app:v1
APP    ?= src/app.py
PORT   ?= 8501

.PHONY: help venv install train register dvc-repro test app api mlflow-ui \
        docker-build docker-run docker-stop \
        k8s-start k8s-load k8s-apply k8s-url k8s-scale k8s-rollout k8s-clean \
        fly-launch fly-deploy fly-status fly-logs fly-secrets clean

help:
	@echo "Common targets:"
	@echo "  make venv            - create a Python 3.11 virtualenv in .venv"
	@echo "  make install         - install dependencies into the venv"
	@echo "  make train           - train the model (logs to MLflow)"
	@echo "  make register        - promote best model to champion registry"
	@echo "  make dvc-repro       - run full DVC pipeline (prepare/train/register)"
	@echo "  make mlflow-ui       - open the MLflow UI at http://127.0.0.1:5000"
	@echo "  make test            - run pytest"
	@echo "  make app             - run the Streamlit app locally"
	@echo "  make api             - run the FastAPI inference service locally"
	@echo "  make docker-build    - build the Docker image"
	@echo "  make docker-run      - run the container on port 8501"
	@echo "  make k8s-start       - start minikube"
	@echo "  make k8s-load        - build the image inside minikube"
	@echo "  make k8s-apply       - apply deployment + service"
	@echo "  make k8s-url         - print the public minikube URL"
	@echo "  make k8s-scale N=4   - scale to N replicas"
	@echo "  make k8s-rollout     - rolling restart of the deployment"
	@echo "  make k8s-clean       - delete the k8s resources"
	@echo "  make fly-launch      - first-time Fly.io app creation"
	@echo "  make fly-deploy      - deploy current code to Fly.io"
	@echo "  make fly-status      - show app status / public URL"

# ---------------- Local Python ----------------

venv:
	$(PYTHON) -m venv .venv
	@echo "Activate with:  source .venv/bin/activate"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) -m src.train

register:
	$(PYTHON) -m src.register

dvc-repro:
	dvc repro

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

test:
	pytest -q

app:
	streamlit run $(APP) --server.port $(PORT) --server.address 0.0.0.0

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

# ---------------- Docker ----------------

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 --name insurance-app $(IMAGE)

docker-stop:
	-docker stop insurance-app

# ---------------- Kubernetes (minikube) ----------------

k8s-start:
	minikube start

k8s-load:
	minikube image build -t $(IMAGE) .

k8s-apply:
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl rollout status deployment/insurance-app-deployment

k8s-url:
	minikube service insurance-app-service --url

k8s-scale:
	@test -n "$(N)" || (echo "Usage: make k8s-scale N=4" && exit 1)
	kubectl scale deployment insurance-app-deployment --replicas=$(N)
	kubectl get pods

k8s-rollout:
	kubectl rollout restart deployment/insurance-app-deployment
	kubectl rollout status deployment/insurance-app-deployment

k8s-clean:
	-kubectl delete -f k8s/service.yaml
	-kubectl delete -f k8s/deployment.yaml

# ---------------- Fly.io (public free hosting) ----------------

fly-launch:
	flyctl launch --no-deploy --copy-config --name insurance-charge-predictor

fly-deploy:
	flyctl deploy

fly-status:
	flyctl status

fly-logs:
	flyctl logs

fly-secrets:
	@test -n "$(NR_KEY)" || (echo "Usage: make fly-secrets NR_KEY=xxxx" && exit 1)
	flyctl secrets set NEW_RELIC_LICENSE_KEY=$(NR_KEY) NEW_RELIC_APP_NAME=insurance-charge-predictor

# ---------------- Cleanup ----------------

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf mlruns models/metrics.json
