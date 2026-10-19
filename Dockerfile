# syntax=docker/dockerfile:1
# =============================================================================
# Reproducible Container for Adversarial Guardrail Co-Evolution
# Target Hardware: Fedora 41 Linux Desktop + NVIDIA TITAN RTX 24GB (CUDA 12.8)
# =============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Copy the official standalone pixi binary
COPY --from=ghcr.io/prefix-dev/pixi:latest /usr/local/bin/pixi /usr/local/bin/pixi

# Prevent interactive prompts and configure rich terminal + unbuffered Python output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TERM=xterm-256color \
    COLORTERM=truecolor \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    RESULTS_DIR=/app/experimental_results \
    HF_HOME=/app/experimental_results/.hf_cache \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install locked Pixi environment first for optimal Docker layer caching
COPY pixi.toml pixi.lock ./
RUN pixi install --locked

# Copy project source code and entrypoint
COPY src/ ./src/
COPY docker-entrypoint.sh ./docker-entrypoint.sh
COPY README.md ./README.md

# Prepare the mount point directory with full read/write permissions
RUN mkdir -p /app/experimental_results \
    && chmod -R 0777 /app/experimental_results \
    && chmod +x /app/docker-entrypoint.sh

VOLUME ["/app/experimental_results"]

# Port 6006 exposes the live TensorBoard server running inside the container
EXPOSE 6006

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
