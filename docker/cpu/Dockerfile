FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.3.0+cpu torchaudio==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    omnivoice \
    fastapi \
    uvicorn[standard] \
    python-multipart

# Copy API source
COPY main.py /app/main.py
COPY index.html /app/web/index.html

ENV HF_HOME=/app/models
ENV DEVICE=cpu
ENV MODEL_ID=k2-fsa/OmniVoice

EXPOSE 8880

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8880", "--workers", "1"]
