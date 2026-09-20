# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ARG VOICE_MODEL=mo-thecreator/Deepfake-audio-detection

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    VOICE_MODEL=${VOICE_MODEL}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Railway runs this service on CPU. Installing the CPU-only wheel avoids
# shipping unused CUDA libraries in the container image.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt

# Cache the public Hugging Face model in the image. This makes container
# startup deterministic and avoids a 378 MB download on every cold boot.
RUN python -c "import os; from transformers import AutoFeatureExtractor, AutoModelForAudioClassification; model_id = os.environ['VOICE_MODEL']; AutoFeatureExtractor.from_pretrained(model_id); AutoModelForAudioClassification.from_pretrained(model_id, use_safetensors=True)"

COPY main.py index.html ./

RUN useradd --create-home --uid 10001 voxguard \
    && chown -R voxguard:voxguard /app /opt/huggingface

USER voxguard

EXPOSE 8000

# Railway injects PORT. One worker is intentional because each worker would
# load a separate copy of the large audio model into memory.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1"]
