FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install OmniVoice + API deps
# omnivoice pulls torch/torchvision/torchaudio with CUDA automatically
RUN pip install --no-cache-dir \
    omnivoice \
    fastapi \
    uvicorn[standard] \
    python-multipart

# Copy API source
COPY main.py /app/main.py
COPY index.html /app/web/index.html

# HuggingFace cache dir
ENV HF_HOME=/app/models
ENV DEVICE=cuda:0
ENV MODEL_ID=k2-fsa/OmniVoice

EXPOSE 8880

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8880", "--workers", "1"]
