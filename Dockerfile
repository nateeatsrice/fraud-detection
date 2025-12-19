# syntax=docker/dockerfile:1
# This syntax directive enables BuildKit features (better caching, parallel builds)
# Always use this for modern Docker builds

# ============================================================================
# STAGE 1: BUILDER - Build dependencies in isolation
# ============================================================================
# We use a multi-stage build to keep the final image small and secure
# This stage installs all build tools and compiles dependencies
FROM python:3.12-slim AS builder

# ----------------------------------------------------------------------------
# Install uv package manager
# ----------------------------------------------------------------------------
# Copy the uv binary from the official uv Docker image
# This is faster and more reliable than downloading/installing uv manually
# The binary is copied to /usr/local/bin so it's available in PATH
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ----------------------------------------------------------------------------
# Install system-level build dependencies
# ----------------------------------------------------------------------------
# Many Python packages (like numpy, pandas, cryptography) need C compilers
# to build native extensions from source
RUN apt-get update && \
    # build-essential: Includes gcc, g++, make - needed for compiling C/C++ extensions
    apt-get install -y --no-install-recommends build-essential && \
    # Clean up apt cache to reduce image size (saves ~100MB)
    # /var/lib/apt/lists/* contains package metadata that's no longer needed
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Set working directory
# ----------------------------------------------------------------------------
# All subsequent commands will run from /app
# This also creates the /app directory if it doesn't exist
WORKDIR /app

# ----------------------------------------------------------------------------
# Copy dependency specification files
# ----------------------------------------------------------------------------
# Copy ONLY the files needed for dependency installation
# This leverages Docker's layer caching:
# - If these files don't change, Docker reuses the cached layer
# - Dependencies won't be reinstalled unless pyproject.toml or uv.lock changes
# - This dramatically speeds up rebuilds during development
COPY pyproject.toml uv.lock ./

# ----------------------------------------------------------------------------
# Install Python dependencies with uv sync
# ----------------------------------------------------------------------------
# uv sync reads pyproject.toml and uv.lock to install exact dependency versions
RUN uv sync \
    # --frozen: Don't update uv.lock, use exact versions specified
    #           Critical for reproducible builds across environments
    #           Prevents "works on my machine" issues
    --frozen \
    # --no-dev: Skip development dependencies (pytest, black, ruff, etc.)
    #           Reduces image size and attack surface in production
    #           Dev dependencies are only needed for testing/linting
    --no-dev

# uv sync automatically:
# - Creates a virtual environment at /app/.venv
# - Installs all production dependencies
# - Resolves and locks all transitive dependencies
# - Compiles any native extensions using build-essential

# ============================================================================
# STAGE 2: RUNTIME - Final lightweight production image
# ============================================================================
# Start fresh from a clean Python image (no build tools, smaller size)
# Only copy over the compiled dependencies, not the build environment
FROM python:3.12-slim

# ----------------------------------------------------------------------------
# Copy the virtual environment from builder stage
# ----------------------------------------------------------------------------
# Copy ONLY the compiled virtual environment, not source files or build tools
# --from=builder: Copy from the previous build stage (not from host machine)
# This .venv contains all installed packages ready to use
COPY --from=builder /app/.venv /app/.venv

# ----------------------------------------------------------------------------
# Set working directory for runtime
# ----------------------------------------------------------------------------
WORKDIR /app

# ----------------------------------------------------------------------------
# Copy application source code
# ----------------------------------------------------------------------------
# Copy all application code into the container
# This happens AFTER dependency installation to maximize cache efficiency
# If you change application code, only this layer and beyond need to rebuild
COPY . .

# NOTE: Make sure you have a .dockerignore file to exclude:
# - .venv (local virtual environment)
# - __pycache__ (Python bytecode cache)
# - .git (git history)
# - *.pyc, *.pyo (compiled Python files)
# This keeps the image clean and small

# ----------------------------------------------------------------------------
# Expose port for the application
# ----------------------------------------------------------------------------
# Documents that the container listens on port 8000
# This is informational - doesn't actually publish the port
# You still need -p 8000:8000 when running: docker run -p 8000:8000 image_name
EXPOSE 8000

# ----------------------------------------------------------------------------
# Set environment variables
# ----------------------------------------------------------------------------
# PYTHONUNBUFFERED=1: Force Python to run in unbuffered mode
# - Print statements appear immediately in logs (no buffering delay)
# - Critical for seeing real-time logs in