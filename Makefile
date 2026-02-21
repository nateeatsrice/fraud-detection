# Makefile for common development tasks

# This tells Make these are command names, not actual files. Without this, Make might get confused if you have a file named "test" or "install".
.PHONY: help setup install test mlflow-push mlflow-pull mlflow-status lint docker-build docker-run docker-push ecr-login terraform-init terraform-apply terraform-destroy prefect-flow clean lambda-package lambda-clean deploy

# Variables
AWS_REGION := us-east-2
ECR_REPO := fraud-detection-app
IMAGE_NAME := fraud-detection-app
ECS_CLUSTER := fraud-detection-cluster
ECS_SERVICE := fraud-detection-service

# The @ suppresses printing the command itself. 

help:
	@echo "Available targets:"
	@echo ""
	@echo "  Setup & Install:"
	@echo "    setup           Create virtual environment with uv"
	@echo "    install         Install Python dependencies with uv"
	@echo ""
	@echo "  Testing:"
	@echo "    test            Run unit and integration tests"
	@echo ""
	@echo "  MLflow:"
	@echo "    push            Upload local mlflow.db to S3"
	@echo "    pull            Download mlflow.db from S3 to local"
	@echo "    status          status update on local and S3 mlflow.db files"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build    Build the Docker image locally"
	@echo "    docker-run      Run the Docker image locally on port 8000"
	@echo "    ecr-login       Login to AWS ECR"
	@echo "    docker-push     Build, tag, and push image to ECR"
	@echo ""
	@echo "  Infrastructure:"
	@echo "    terraform-init  Initialize Terraform"
	@echo "    terraform-apply Apply Terraform configuration"
	@echo "    terraform-destroy Destroy all Terraform resources"
	@echo ""
	@echo "  Deployment:"
	@echo "    deploy          Full deploy: build, push, and update ECS"
	@echo "    ecs-redeploy    Force ECS to pull latest image"
	@echo ""
	@echo "  Lambda:"
	@echo "    lambda-package  Package the Lambda function"
	@echo "    lambda-clean    Remove Lambda package"
	@echo ""
	@echo "  Other:"
	@echo "    prefect-flow    Run Prefect orchestration flow locally"
	@echo "    clean           Remove virtual environment and cache files"

setup:
	uv venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

install:
	uv sync

test:
	uv run pytest -q

mlflow-push:
	./scripts/sync_mlflow.sh push

mlflow-pull:
	./scripts/sync_mlflow.sh pull

mlflow-status:
	./scripts/sync_mlflow.sh status

docker-build:
	docker build --progress=plain -t $(IMAGE_NAME):latest .

docker-run:
	docker run -p 8000:8000 $(IMAGE_NAME):latest

ecr-login:
	@echo "Logging in to ECR..."
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_URL)

docker-push: ecr-login docker-build
	@echo "Tagging image..."
	docker tag $(IMAGE_NAME):latest $(ECR_URL)/$(ECR_REPO):latest
	@echo "Pushing to ECR..."
	docker push $(ECR_URL)/$(ECR_REPO):latest
	@echo "Image pushed to $(ECR_URL)/$(ECR_REPO):latest"

terraform-init:
	cd terraform && terraform init

terraform-apply: lambda-package
	cd terraform && terraform apply

terraform-destroy:
	cd terraform && terraform destroy

# Deployment
# =============================================================================

# Force ECS to pull the latest image and redeploy
ecs-redeploy:
	@echo "Forcing ECS service to redeploy..."
	aws ecs update-service \
		--cluster $(ECS_CLUSTER) \
		--service $(ECS_SERVICE) \
		--force-new-deployment \
		--region $(AWS_REGION)
	@echo "Deployment triggered. Monitor with: aws ecs describe-services --cluster $(ECS_CLUSTER) --services $(ECS_SERVICE)"

# Full deployment: build, push, and redeploy
deploy: docker-push ecs-redeploy
	@echo "Deployment complete!"

lambda-package:
	@echo "Packaging Lambda function..."
	@cd lambda && zip -r ../lambda_function_payload.zip . -x "*.pyc" -x "__pycache__/*" -x "*.git*"
	@echo "Lambda package created: lambda_function_payload.zip"

lambda-clean:
	@echo "Cleaning Lambda package..."
	@rm -f lambda_function_payload.zip
	@echo "Lambda package removed"

prefect-flow:
	uv run python orchestration/flow.py

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
