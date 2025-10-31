FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git build-essential cmake ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/ggerganov/whisper.cpp
WORKDIR /opt/whisper.cpp
RUN cmake -B build && cmake --build build --config Release

WORKDIR /app
COPY backend/ .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.24.0 \
    requests==2.31.0 \
    python-multipart==0.0.6 \
    boto3==1.34.0

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchaudio==2.0.2 \
    transformers==4.37.2 \
    huggingface-hub==0.20.3 \
    pyannote.audio==3.1.1 \
    pytorch-lightning==2.1.3 \
    torchmetrics==1.3.1 \
    matplotlib==3.8.4 \
    scipy==1.11.4 \
    loguru \
    openai-whisper \
    whisper \
    librosa

RUN pip install --force-reinstall --no-cache-dir "numpy==1.26.4"

# 🗂️ Настраиваем кэш Hugging Face
ENV HF_HOME=/models/hf_cache

RUN test -f /opt/whisper.cpp/build/bin/whisper-cli

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]