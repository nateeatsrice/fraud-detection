# =============================================================================
# VARIABLES
# =============================================================================
# All configurable inputs for the infrastructure.
# Override via terraform.tfvars, environment variables (TF_VAR_*), or CLI.
# =============================================================================

variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "us-east-2"
}

variable "bucket_name" {
  description = "Name of the S3 bucket used to store artifacts"
  type        = string
  default     = "fraud-detection-artifacts-nateeatsrice-2026"
}

variable "ecr_repo_name" {
  description = "Name of the ECR repository for the API image"
  type        = string
  default     = "fraud-detection-app"
}

variable "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  type        = string
  default     = "fraud-detection-cluster"
}

variable "sqs_queue_name" {
  description = "Name of the SQS queue"
  type        = string
  default     = "fraud-detection-events"
}

variable "lambda_role_name" {
  description = "Name of the IAM role used by the Lambda function"
  type        = string
  default     = "fraud-detection-lambda-role"
}

variable "lambda_function_name" {
  description = "Name of the Lambda function"
  type        = string
  default     = "fraud-detection-worker"
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest" # Fallback for local testing
}