FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy environment and source
COPY environment.yml /app/environment.yml
RUN python3 -m venv /opt/venv && . /opt/venv/bin/activate && pip install --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir tensorflow==2.12 && \
    pip install --no-cache-dir numpy scipy matplotlib opencv-python Pillow pyyaml tqdm rich pytest pytest-mock einops albumentations tensorboard

ENV PATH="/opt/venv/bin:${PATH}"

COPY . /app

CMD ["python", "src/train_pytorch.py", "--config", "config/config_pytorch.yaml"]
