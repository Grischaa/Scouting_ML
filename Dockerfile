FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=src \
    MODEL_ARTIFACTS_DIR=/app/data/model

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    fonts-dejavu-core \
    libcairo2 \
    libffi-dev \
    libgdk-pixbuf-2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/model/ ./data/model/
COPY frontend/ ./frontend/

EXPOSE 8000

CMD ["uvicorn", "scouting_ml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
