# Qwen Sovereign Core Dockerfile
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install ML stack
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    fastapi \
    uvicorn \
    sentencepiece

# Copy core modules and server
COPY core/ ./core/
COPY api/models/qwen_server.py ./server.py

# Execution
EXPOSE 5555
CMD ["python3", "server.py"]
