# Production Dockerfile for Plant Leaf Disease Detection System
# Multi-stage build for optimized production deployment

# Base stage - System dependencies
FROM python:3.11-slim-bullseye AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Dependencies stage - Python packages
FROM base AS dependencies

WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-prod.txt ./

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-prod.txt

# Application stage - Production application
FROM base AS production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install runtime-only dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from dependencies stage
COPY --from=dependencies /opt/venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/temp /app/data && \
    chown -R appuser:appuser /app

# Set permissions
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# Security configurations
RUN python -m pip uninstall --yes setuptools wheel && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "*.pyo" -delete && \
    find /opt/venv -type d -name "__pycache__" -delete

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Labels for metadata
LABEL maintainer="Plant Disease Detection Team" \
      version="1.0.0" \
      description="Production Plant Leaf Disease Detection System" \
      org.opencontainers.image.title="Plant Disease Detection" \
      org.opencontainers.image.description="AI-powered plant disease detection system" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Agriculture AI Solutions" \
      org.opencontainers.image.licenses="MIT"