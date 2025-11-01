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
    fastapi\
    uvicorn\
    requests\
    python-multipart\
    boto3==1.34.0

RUN pip install --no-cache-dir \
    torch \
    torchaudio\
    transformers\
    huggingface-hub\
    pyannote.audio\
    pytorch-lightning\
    torchmetrics\
    matplotlib\
    scipy\
    loguru \
    openai-whisper \
    whisper \
    librosa

RUN pip install --force-reinstall --no-cache-dir "numpy"

# 🗂️ Настраиваем кэш Hugging Face
ENV HF_HOME=/models/hf_cache

RUN test -f /opt/whisper.cpp/build/bin/whisper-cli

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]