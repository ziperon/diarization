# Use CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set architecture explicitly
ENV DEBIAN_FRONTEND=noninteractive
ENV ARCH=amd64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ffmpeg \
    curl \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Set up Python environment
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# # Verify Python architecture and CUDA
# RUN python -c "import platform; print(f'Python architecture: {platform.architecture()}')" && \
#     python -c "import struct; print(f'Pointer size: {8 * struct.calcsize('P')} bits')" && \
#     python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
#                print(f'Number of GPUs: {torch.cuda.device_count()}'); \
#                [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]