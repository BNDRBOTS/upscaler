FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 git wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pull Real-ESRGAN inference script into /app
RUN git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git /tmp/realesrgan \
    && cp /tmp/realesrgan/inference_realesrgan.py /app/ \
    && rm -rf /tmp/realesrgan

# Download BOTH model weights (x4 for quality, x2 for speed option)
RUN mkdir -p /app/weights \
    && wget -q \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
       -O /app/weights/RealESRGAN_x4plus.pth \
    && wget -q \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth \
       -O /app/weights/RealESRGAN_x2plus.pth

COPY main.py .

EXPOSE 8000

# Increase uvicorn timeout to handle large image inference
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "300", "--workers", "1"]
