FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install "setuptools<82" && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    python -m pip install -r /app/requirements.txt

COPY . /app

CMD sh -c 'gunicorn wsgi:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 300'
