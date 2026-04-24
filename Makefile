# ============================================================
# Makefile — Insurance Charge Predictor
# Works on macOS and Linux.
# Windows users: run `.\scripts\make.ps1 <target>` in PowerShell
# ============================================================

PYTHON ?= python3.11
PIP    ?= pip
IMAGE  ?= insurance-app:v1
APP    ?= src/app.py
PORT   ?= 8501

.PHONY: help venv install train test app mlflow-ui \
        docker-build docker-run docker-stop \
        k8s-start k8s-load k8s-apply k8s-url k8s-scale k8s-rollout k8s-clean \
        fly-launch fly-deploy fly-status fly-logs fly-secrets clean

help:
	@echo "Common targets:"
	@echo "  make venv            - create a Python 3.11 virtualenv in .venv"
	@echo "  make install         - install dependencies into the venv"
	@echo "  make train           - train the model (logs to MLflow)"
	@echo "  make mlflow-ui       - open the MLflow UI at http://127.0.0.1:5000"
	@echo "  make test            - run pytest"
	@echo "  make app             - run the Streamlit app locally"
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

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

test:
	pytest -q

app:
	streamlit run $(APP) --server.port $(PORT) --server.address 0.0.0.0

# ---------------- Docker ----------------

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8501 --name insurance-app $(IMAGE)

docker-stop:
	-docker stop insurance-app

# ---------------- Kubernetes (minikube) ----------------

k8s-start:
	docker system prune -a --volumes
	minikube delete --all --purge
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

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf mlruns models/metrics.json
