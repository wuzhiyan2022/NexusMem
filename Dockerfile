FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LINEARRAG_HOST=0.0.0.0 \
    LINEARRAG_PORT=8000 \
    LINEARRAG_STORAGE_DIR=/data/import_api \
    LINEARRAG_DEFAULT_TOP_K=100 \
    LINEARRAG_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2 \
    LINEARRAG_SPACY_MODEL=en_core_web_trf \
    LINEARRAG_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements_api.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements_api.txt \
    && python -m spacy download en_core_web_trf \
    && python - <<'PY'
from sentence_transformers import SentenceTransformer

SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
PY

COPY . .

RUN chmod +x scripts/run_api.sh

EXPOSE 8000

CMD ["scripts/run_api.sh"]
