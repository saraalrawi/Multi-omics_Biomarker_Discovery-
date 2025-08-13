# Multi-stage Docker build for Multi-omics Biomarker Discovery
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    libhdf5-dev \
    libnetcdf-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    jupyter-dash

# Copy source code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/results logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8888 8050 8080

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser requirements.txt .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/results logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Default command for production
CMD ["python", "-m", "src.main"]

# Testing stage
FROM development as testing

# Copy test files
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser pytest.ini .
COPY --chown=appuser:appuser .coveragerc .

# Run tests
RUN python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Documentation stage
FROM base as docs

# Install documentation dependencies
RUN pip install --no-cache-dir \
    sphinx \
    sphinx-rtd-theme \
    nbsphinx \
    myst-parser

# Copy documentation files
COPY --chown=appuser:appuser docs/ ./docs/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser notebooks/ ./notebooks/

# Switch to non-root user
USER appuser

# Build documentation
RUN cd docs && make html

# Expose documentation port
EXPOSE 8000

# Serve documentation
CMD ["python", "-m", "http.server", "8000", "--directory", "docs/_build/html"]