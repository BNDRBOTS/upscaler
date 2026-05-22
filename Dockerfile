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

RUN pip install --no-cache-dir \
    torch==2.3.1+cpu torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN PIP_USE_PEP517=1 pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    python-multipart==0.0.9 \
    "Pillow>=10.0.0" \
    "numpy>=1.24.0" \
    "opencv-python>=4.8.0" \
    "facexlib>=0.3.0" \
    "gfpgan>=1.3.8" \
    "basicsr-fixed>=1.4.2.4" \
    "realesrgan>=0.3.0"

# Fix the broken torchvision import across all basicsr files
RUN find /usr/local/lib/python3.11/site-packages/basicsr -type f -name "*.py" -exec sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' {} +

RUN git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git /tmp/realesrgan \
    && cp /tmp/realesrgan/inference_realesrgan.py /app/ \
    && rm -rf /tmp/realesrgan

RUN mkdir -p /app/weights \
    && wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/weights/RealESRGAN_x4plus.pth \
    && wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O /app/weights/RealESRGAN_x2plus.pth

# Fail the build immediately if the imports are still broken
RUN python -c "import basicsr.data.realesrgan_dataset; print('basicsr OK')"
RUN python -c "import cv2; print('cv2 OK')"

COPY main.py .

EXPOSE 8000

CMD sh -c 'uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --timeout-keep-alive 300 --workers 1'
