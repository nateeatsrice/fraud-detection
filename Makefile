# Makefile for common development tasks

.PHONY: help install test lint docker-build docker-run terraform-init terraform-apply prefect-flow

help:
	@echo "Available targets:"
	@echo "  install         Install Python dependencies"
	@echo "  test            Run unit and integration tests using pytest"
	@echo "  docker-build    Build the Docker image for the API service"
	@echo "  docker-run      Run the Docker image locally on port 8000"
	@echo "  terraform-init  Initialise Terraform in the terraform/ directory"
	@echo "  terraform-apply Apply the Terraform configuration"
	@echo "  prefect-flow    Run the Prefect orchestration flow locally"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

docker-build:
	docker build -t fraud-detection-app .

docker-run:
	docker run -p 8000:8000 fraud-detection-app

terraform-init:
	cd terraform && terraform init

terraform-apply:
	cd terraform && terraform apply -auto-approve

prefect-flow:
	python orchestration/flow.py