FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# CPU-only torch first
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# Only the non-ambiguous base deps go here
RUN pip install --no-cache-dir -r requirements.txt

# Grab Real-ESRGAN source
RUN git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git /tmp/realesrgan

# Install the GitHub-fixed BasicSR commit directly
RUN PIP_USE_PEP517=1 pip install --no-cache-dir \
    "basicsr @ git+https://github.com/XPixelGroup/BasicSR@8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a"

# Install Real-ESRGAN package code without letting pip resolve/reinstall deps
RUN pip install --no-cache-dir --no-deps /tmp/realesrgan

# Copy the inference script we call from main.py
RUN cp /tmp/realesrgan/inference_realesrgan.py /app/inference_realesrgan.py

# Model weights
RUN mkdir -p /app/weights \
    && wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/weights/RealESRGAN_x4plus.pth \
    && wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O /app/weights/RealESRGAN_x2plus.pth

# Fail the build if the exact import chain is still broken
RUN python -c "import cv2; print('cv2 OK')" \
    && python -c "from basicsr.data import realesrgan_dataset; from realesrgan import RealESRGANer; print('imports OK')"

COPY main.py .

EXPOSE 8000

CMD sh -c 'uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --timeout-keep-alive 300 --workers 1'
