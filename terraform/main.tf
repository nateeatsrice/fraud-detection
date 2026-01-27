terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.4.0"
}

provider "aws" {
  region = var.aws_region
}

# S3 bucket for storing datasets, models and monitoring reports
resource "aws_s3_bucket" "artifacts" {
  bucket        = var.bucket_name
  force_destroy = true
  tags = {
    Project     = "fraud-detection"
    Environment = "dev"
  }
}

# ECR repository to hold the Docker image for the FastAPI application
resource "aws_ecr_repository" "app_repo" {
  name = var.ecr_repo_name
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = {
    Project = "fraud-detection"
  }
}

# ECS cluster for running the API service
resource "aws_ecs_cluster" "cluster" {
  name = var.ecs_cluster_name
  tags = {
    Project = "fraud-detection"
  }
}

# SQS queue for asynchronous event handling (e.g. background training jobs)
resource "aws_sqs_queue" "events" {
  name                       = var.sqs_queue_name
  visibility_timeout_seconds = 60
}

# IAM role for Lambda execution
data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = var.lambda_role_name
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

# Basic inline policy allowing the Lambda to write logs and access SQS and S3
resource "aws_iam_role_policy" "lambda_policy" {
  name = "lambda_policy"
  role = aws_iam_role.lambda_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect   = "Allow"
        Action   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"]
        Resource = aws_sqs_queue.events.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
      }
    ]
  })
}

# Placeholder Lambda function for background processing
resource "aws_lambda_function" "worker" {
  function_name    = var.lambda_function_name
  role             = aws_iam_role.lambda_exec.arn
  runtime          = "python3.12"
  handler          = "handler.lambda_handler"
  filename         = "../lambda_function_payload.zip"
  source_code_hash = filebase64sha256("../lambda_function_payload.zip")
  timeout          = 30
}

# CloudWatch log group for the Lambda function
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${var.lambda_function_name}"
  retention_in_days = 14
}
