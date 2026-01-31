# syntax=docker/dockerfile:1
# This syntax directive enables BuildKit features (better caching, parallel builds)
# Always use this for modern Docker builds

# ============================================================================
# STAGE 1: BUILDER - Build dependencies in isolation
# ============================================================================
FROM python:3.12-slim AS builder

# ----------------------------------------------------------------------------
# Install uv package manager
# ----------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ----------------------------------------------------------------------------
# Install system-level build dependencies
# ----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        # Add runtime dependencies needed for xgboost and lightgbm
        libgomp1 \
        && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Set working directory
# ----------------------------------------------------------------------------
WORKDIR /app

# ----------------------------------------------------------------------------
# Copy dependency specification files
# ----------------------------------------------------------------------------
COPY pyproject.toml uv.lock README.md ./

# ----------------------------------------------------------------------------
# Install Python dependencies with uv sync
# ----------------------------------------------------------------------------
RUN uv sync \
    --frozen \
    --no-dev

# ============================================================================
# STAGE 2: RUNTIME - Final lightweight production image
# ============================================================================
FROM python:3.12-slim

# ----------------------------------------------------------------------------
# Install RUNTIME system dependencies
# ----------------------------------------------------------------------------
# These are needed by compiled Python packages at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # OpenMP library - required by LightGBM and XGBoost
        libgomp1 \
        # Optional: curl for healthchecks
        curl \
        && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Copy the virtual environment from builder stage
# ----------------------------------------------------------------------------
COPY --from=builder /app/.venv /app/.venv

# ----------------------------------------------------------------------------
# Set working directory for runtime
# ----------------------------------------------------------------------------
WORKDIR /app

# ----------------------------------------------------------------------------
# Copy application source code
# ----------------------------------------------------------------------------
COPY . .

# ----------------------------------------------------------------------------
# Expose port for the application
# ----------------------------------------------------------------------------
EXPOSE 8000

# ----------------------------------------------------------------------------
# Set environment variables
# ----------------------------------------------------------------------------
# PYTHONUNBUFFERED=1: Force Python to run in unbuffered mode
# - Print statements appear immediately in logs (no buffering delay)
# - Critical for seeing real-time logs in Docker and cloud environments
# - Ensures stdout/stderr aren't buffered, important for debugging
ENV PYTHONUNBUFFERED=1

# ----------------------------------------------------------------------------
# Start the FastAPI application
# ----------------------------------------------------------------------------
# Run uvicorn server on all interfaces (0.0.0.0) to accept external connections
# Port 8000 matches the EXPOSE directive above
# Using exec form (JSON array) ensures proper signal handling for graceful shutdown
# This means SIGTERM signals are properly forwarded to the uvicorn process
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]