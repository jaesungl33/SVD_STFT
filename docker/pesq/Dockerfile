# syntax=docker/dockerfile:1
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Pin numpy compatible with pesq build
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.11.4 soundfile==0.12.1
# Build PESQ wheel in linux (works better than macOS ARM host)
RUN pip install --no-cache-dir pesq==0.0.4

WORKDIR /app

# Runner script
COPY run_pesq.py /app/run_pesq.py

ENTRYPOINT ["python", "/app/run_pesq.py"]
