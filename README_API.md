# LinearRAG Memory API

This is a competition-oriented API wrapper around LinearRAG retrieval.
It implements the Agent Memory Challenge Add/Search contract without generating final answers.

## Endpoints

- `GET /health`
- `POST /add`
- `POST /search`

`/add` persists and indexes memories synchronously before returning `success=true`.
`/search` only returns ranked memory evidence under `data`; it does not answer the question.

## Setup

Use Python 3.9 as recommended by the original LinearRAG project:

```bash
cd /home/my5090/LinearRAG
conda create -n linearrag-api python=3.9 -y
conda activate linearrag-api
pip install -r requirements_api.txt
pip install model/spacy/en_core_web_trf-3.6.1-py3-none-any.whl
```

## Run

```bash
cd /home/my5090/LinearRAG
conda activate linearrag-api
scripts/run_api.sh
```

Useful environment variables:

```bash
export LINEARRAG_API_TOKEN="smoke-test-key"
export LINEARRAG_HOST=0.0.0.0
export LINEARRAG_PORT=8000
export LINEARRAG_STORAGE_DIR=import_api
export LINEARRAG_EMBEDDING_MODEL=model/all-mpnet-base-v2
export LINEARRAG_SPACY_MODEL=en_core_web_trf
export LINEARRAG_DEVICE=cpu     # or cuda
```

If `LINEARRAG_API_TOKEN` is set, `/add` and `/search` require an API token.
`MEMORY_API_KEY` is still supported as a backwards-compatible alias.
The challenge platform should use:

- authentication mode: `Authorization: Token`
- memory system key: the same value as `LINEARRAG_API_TOKEN`

Accepted request headers are:

- `Authorization: Bearer <key>`
- `Authorization: Token <key>`
- `X-Api-Key: <key>`

`GET /health` stays unauthenticated and returns `auth_required=true` when a token is configured.

## Smoke Test

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/add \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Token smoke-test-key' \
  -d '{
    "request_id": "smoke:add:1",
    "user_id": "smoke:user:1",
    "session_id": "smoke:session:1",
    "messages": [
      {"role": "user", "timestamp": 1704067200000, "content": "Caroline went to an LGBTQ support group yesterday."}
    ]
  }'

curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Token smoke-test-key' \
  -d '{
    "user_id": "smoke:user:1",
    "query": "When did Caroline go to the LGBTQ support group?",
    "top_k": 10
  }'
```

## Storage Model

`user_id` is mapped to a SHA-256 namespace under `LINEARRAG_STORAGE_DIR/users/`.
Each namespace keeps its own embeddings, NER cache, graph output, and request metadata.
Search never reads memories from another `user_id`.
