FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY models ./models
COPY reports ./reports

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e .

RUN python scripts/download_data.py \
    && python -m heart_disease_mlops.train --fast --tracking-uri /app/mlruns

EXPOSE 8000

CMD ["uvicorn", "heart_disease_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
