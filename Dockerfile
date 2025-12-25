# Use a smaller base image for CPU
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install GigaAM from GitHub
RUN pip install --no-cache-dir git+https://github.com/salute-developers/GigaAM.git@main

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/tmp/audiot /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HOME=/app/models/hf_cache
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV TORCH_HOME=/app/models/torch
ENV PYANNOTE_HOME=/app/models/pyannote

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]