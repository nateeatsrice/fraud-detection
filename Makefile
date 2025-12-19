# Makefile for common development tasks

# This tells Make these are command names, not actual files. Without this, Make might get confused if you have a file named "test" or "install".
.PHONY: help setup install test lint docker-build docker-run terraform-init terraform-apply prefect-flow clean

# The @ suppresses printing the command itself. 
help:
	@echo "Available targets:"
	@echo "  setup           Create virtual environment with uv"
	@echo "  install         Install Python dependencies with uv"
	@echo "  test            Run unit and integration tests using pytest"
	@echo "  docker-build    Build the Docker image for the API service"
	@echo "  docker-run      Run the Docker image locally on port 8000"
	@echo "  terraform-init  Initialise Terraform in the terraform/ directory"
	@echo "  terraform-apply Apply the Terraform configuration"
	@echo "  prefect-flow    Run the Prefect orchestration flow locally"
	@echo "  clean           Remove virtual environment and cache files"

setup:
	uv venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

install:
	uv pip install -r requirements.txt

test:
	uv run pytest -q

docker-build:
	docker build -t fraud-detection-app .

docker-run:
	docker run -p 8000:8000 fraud-detection-app

terraform-init:
	cd terraform && terraform init

terraform-apply:
	cd terraform && terraform apply -auto-approve

prefect-flow:
	uv run python orchestration/flow.py

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete