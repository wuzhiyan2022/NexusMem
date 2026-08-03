# NexusMem Submission Guide

## Repository

This repository is based on the original LinearRAG implementation:

- Original method: https://github.com/DEEP-PolyU/LinearRAG
- Adapted system name: NexusMem
- Version: v0.1-smoke

## Service Endpoints

The Agent Memory Challenge platform should call:

- `POST /add`
- `POST /search`
- `GET /health` for unauthenticated liveness checks

For the current smoke deployment, the externally exposed URLs are produced by a temporary Cloudflare Tunnel. Replace the host with the active tunnel/domain:

```text
Add API:    https://<public-host>/add
Search API: https://<public-host>/search
```

Authentication:

```text
Authorization: Token <your-token>
```

The same token must be configured in the service through `LINEARRAG_API_TOKEN`.

## Conda Run Command

The tested server environment is:

```bash
cd /home/my5090/LinearRAG
source /home/my5090/miniconda3/etc/profile.d/conda.sh
conda activate linearrag_pf

export LINEARRAG_API_TOKEN="<your-token>"
export LINEARRAG_HOST=127.0.0.1
export LINEARRAG_PORT=8000
export LINEARRAG_DEVICE=cpu
export LINEARRAG_STORAGE_DIR=import_api
export LINEARRAG_EMBEDDING_MODEL=model/all-mpnet-base-v2
export LINEARRAG_SPACY_MODEL=en_core_web_trf

scripts/run_api.sh
```

For direct deployment without a tunnel, set `LINEARRAG_HOST=0.0.0.0`. For the smoke test, `127.0.0.1` is safer because only the tunnel can reach the API.

## Docker Build And Run

Build:

```bash
docker build -t nexusmem-linearrag .
```

CPU run:

```bash
docker run --rm \
  -p 8000:8000 \
  -e LINEARRAG_API_TOKEN="<your-token>" \
  -e LINEARRAG_DEVICE=cpu \
  -v "$PWD/import_api:/data/import_api" \
  nexusmem-linearrag
```

GPU run, if the host has NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e LINEARRAG_API_TOKEN="<your-token>" \
  -e LINEARRAG_DEVICE=cuda \
  -v "$PWD/import_api:/data/import_api" \
  nexusmem-linearrag
```

## Add/Search Wrapper Location

The competition wrapper is implemented in:

```text
api_server.py
scripts/run_api.sh
requirements_api.txt
```

`api_server.py` exposes:

- `POST /add`: receives platform memories, persists them by `user_id`, chunks long messages, builds LinearRAG indexes, and returns success only after indexing finishes.
- `POST /search`: receives a query and `user_id`, searches only that user's indexed memory, and returns top-k evidence under `data`.
- `GET /health`: returns service status and whether authentication is enabled.

## Storage Layout

Each `user_id` is hashed into an isolated namespace under:

```text
LINEARRAG_STORAGE_DIR/users/<user_hash>/
```

Each namespace stores LinearRAG artifacts independently:

```text
metadata.json
passage_embedding.parquet
entity_embedding.parquet
sentence_embedding.parquet
ner_results.json
LinearRAG.graphml
```

Search for one `user_id` never reads another user's memory directory.

## Corpus Processing

The challenge platform sends memory through `/add`. The wrapper treats each message as one memory passage by default. If a message is longer than `LINEARRAG_MAX_CHUNK_CHARS`, it is split by sentence-like boundaries and then by character length as a fallback.

Each stored passage is prefixed with metadata:

```text
[session_id=...] [request_id=...] [message_index=...] [part_index=...] [role=...] [timestamp=...]
role: content
```

This preserves the original session/message context while keeping LinearRAG's retrieval input text-only.

## Main Method Changes

Compared with the original LinearRAG repository, this submission adds:

1. A FastAPI memory service for the Agent Memory Challenge Add/Search contract.
2. Per-user storage isolation based on hashed `user_id`.
3. Synchronous indexing during `/add`, so the service returns success only after the memory is searchable.
4. Token authentication for `/add` and `/search` through `Authorization: Token <key>`.
5. Text-only evidence return from `/search`; the system does not generate final answers.
6. LoCoMo retrieval evaluation script at `scripts/evaluate_locomo_retrieval.py`, which compares retrieved `dia_id` values with LoCoMo gold evidence and reports hit/recall/precision/full-recall/MRR.
7. Optional Docker packaging for reproducible API deployment.

## LoCoMo Retrieval Evaluation

The local retrieval evaluator can be run without the HTTP API:

```bash
cd /home/my5090/LinearRAG
source /home/my5090/miniconda3/etc/profile.d/conda.sh
conda activate linearrag_pf
export LINEARRAG_DEVICE=cpu

python scripts/evaluate_locomo_retrieval.py \
  --locomo-dir /home/my5090/LoCoMo_refined \
  --storage-dir import_api_locomo_eval \
  --output-dir outputs/locomo_retrieval_eval \
  --top-k 100 \
  --ks 1,5,10,20,50,100 \
  --omit-image-urls
```

Outputs:

```text
outputs/locomo_retrieval_eval/retrieval_summary.json
outputs/locomo_retrieval_eval/retrieval_details.jsonl
```

This evaluator is only for offline analysis. The formal `/search` API does not use gold evidence.

## Smoke Test

```bash
curl https://<public-host>/health

curl -X POST https://<public-host>/add \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Token <your-token>' \
  -d '{
    "request_id": "smoke:add:1",
    "user_id": "smoke:user:1",
    "session_id": "smoke:session:1",
    "messages": [
      {
        "role": "user",
        "timestamp": 1704067200000,
        "content": "Caroline went to an LGBTQ support group yesterday."
      }
    ]
  }'

curl -X POST https://<public-host>/search \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Token <your-token>' \
  -d '{
    "user_id": "smoke:user:1",
    "query": "When did Caroline go to the LGBTQ support group?",
    "top_k": 10
  }'
```
