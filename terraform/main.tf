# =============================================================================
# FRAUD DETECTION PROJECT - MAIN INFRASTRUCTURE
# =============================================================================
# Cost-optimized setup for portfolio/learning (~$25/month)
#
# What this creates:
#   - VPC with public subnets (no NAT Gateway to save costs)
#   - Application Load Balancer
#   - ECS Fargate service running your FastAPI app
#   - ECR repository for Docker images
#   - S3 bucket for artifacts
#   - SQS queue for async processing
#   - Lambda function for background work
# =============================================================================

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

# -----------------------------------------------------------------------------
# DATA SOURCES
# -----------------------------------------------------------------------------

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# =============================================================================
# NETWORKING
# =============================================================================
# VPC with public subnets only (no NAT Gateway = saves ~$32/month)
# Containers get public IPs but security groups restrict access to ALB only

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name    = "fraud-detection-vpc"
    Project = "fraud-detection"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name    = "fraud-detection-igw"
    Project = "fraud-detection"
  }
}

# Two public subnets in different AZs (ALB requires at least 2)
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name    = "fraud-detection-public-${count.index + 1}"
    Project = "fraud-detection"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name    = "fraud-detection-public-rt"
    Project = "fraud-detection"
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# =============================================================================
# SECURITY GROUPS
# =============================================================================

# ALB security group - allows HTTP from internet
resource "aws_security_group" "alb" {
  name        = "fraud-detection-alb-sg"
  description = "Allow HTTP traffic to ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP from internet"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "fraud-detection-alb-sg"
    Project = "fraud-detection"
  }
}

# ECS tasks security group - only allows traffic from ALB
resource "aws_security_group" "ecs_tasks" {
  name        = "fraud-detection-ecs-sg"
  description = "Allow traffic from ALB only"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "HTTP from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "fraud-detection-ecs-sg"
    Project = "fraud-detection"
  }
}

# =============================================================================
# S3 BUCKET
# =============================================================================

resource "aws_s3_bucket" "artifacts" {
  bucket        = var.bucket_name
  force_destroy = true

  tags = {
    Project     = "fraud-detection"
    Environment = "dev"
  }
}

# =============================================================================
# ECR REPOSITORY
# =============================================================================

resource "aws_ecr_repository" "app_repo" {
  name = var.ecr_repo_name

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project = "fraud-detection"
  }
}

# =============================================================================
# SQS QUEUE
# =============================================================================

resource "aws_sqs_queue" "events" {
  name                       = var.sqs_queue_name
  visibility_timeout_seconds = 60

  tags = {
    Project = "fraud-detection"
  }
}

# =============================================================================
# IAM ROLES FOR ECS
# =============================================================================

data "aws_iam_policy_document" "ecs_task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Execution role - ECS uses this to pull images and write logs
resource "aws_iam_role" "ecs_task_execution" {
  name               = "fraud-detection-ecs-execution-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Task role - your application uses this to access S3 and SQS
resource "aws_iam_role" "ecs_task" {
  name               = "fraud-detection-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

resource "aws_iam_role_policy" "ecs_task" {
  name = "fraud-detection-ecs-task-policy"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["sqs:SendMessage", "sqs:ReceiveMessage", "sqs:DeleteMessage"]
        Resource = aws_sqs_queue.events.arn
      }
    ]
  })
}

# =============================================================================
# APPLICATION LOAD BALANCER
# =============================================================================

resource "aws_lb" "main" {
  name               = "fraud-detection-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  tags = {
    Name    = "fraud-detection-alb"
    Project = "fraud-detection"
  }
}

resource "aws_lb_target_group" "app" {
  name        = "fraud-detection-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/"
    protocol            = "HTTP"
    matcher             = "200"
  }

  tags = {
    Name    = "fraud-detection-tg"
    Project = "fraud-detection"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# =============================================================================
# ECS CLUSTER, TASK DEFINITION, AND SERVICE
# =============================================================================

resource "aws_ecs_cluster" "cluster" {
  name = var.ecs_cluster_name

  tags = {
    Project = "fraud-detection"
  }
}

# CloudWatch log group for ECS tasks
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/fraud-detection-app"
  retention_in_days = 7

  tags = {
    Project = "fraud-detection"
  }
}

# Task definition - the "recipe" for your container
resource "aws_ecs_task_definition" "app" {
  family                   = "fraud-detection-app"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 256
  memory                   = 512
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "fraud-detection-app"
      image = "${aws_ecr_repository.app_repo.repository_url}:latest"

      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        { name = "ENVIRONMENT", value = "production" },
        { name = "S3_BUCKET", value = var.bucket_name },
        { name = "SQS_QUEUE_URL", value = aws_sqs_queue.events.url }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "ecs"
        }
      }

      essential = true
    }
  ])

  tags = {
    Project = "fraud-detection"
  }
}

# ECS Service - keeps your desired number of tasks running
resource "aws_ecs_service" "app" {
  name            = "fraud-detection-service"
  cluster         = aws_ecs_cluster.cluster.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "fraud-detection-app"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]

  tags = {
    Project = "fraud-detection"
  }
}

# =============================================================================
# LAMBDA FUNCTION
# =============================================================================

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

resource "aws_lambda_function" "worker" {
  function_name    = var.lambda_function_name
  role             = aws_iam_role.lambda_exec.arn
  runtime          = "python3.12"
  handler          = "handler.lambda_handler"
  filename         = "../lambda_function_payload.zip"
  source_code_hash = filebase64sha256("../lambda_function_payload.zip")
  timeout          = 30

  tags = {
    Project = "fraud-detection"
  }
}

resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${var.lambda_function_name}"
  retention_in_days = 7

  tags = {
    Project = "fraud-detection"
  }
}
