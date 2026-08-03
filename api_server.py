from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.LinearRAG import LinearRAG
from src.config import LinearRAGConfig
from src.ner import SpacyNER


class Message(BaseModel):
    role: str
    content: str
    timestamp: int | float | str | None = None

    class Config:
        extra = "ignore"


class AddRequest(BaseModel):
    request_id: str
    messages: list[Message]
    user_id: str
    session_id: str

    class Config:
        extra = "ignore"


class SearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = Field(default=100, ge=0)
    options: list[str] | None = None

    class Config:
        extra = "ignore"


app = FastAPI(title="LinearRAG Memory API", version="0.1.0")


def configured_api_token() -> str:
    return (os.getenv("LINEARRAG_API_TOKEN") or os.getenv("MEMORY_API_KEY") or "").strip()


def verify_api_key(request: Request) -> None:
    expected = configured_api_token()
    if not expected:
        return

    x_api_key = request.headers.get("x-api-key", "").strip()
    authorization = request.headers.get("authorization", "").strip()
    accepted_auth = {
        expected,
        f"Bearer {expected}",
        f"Token {expected}",
    }
    if hmac.compare_digest(x_api_key, expected):
        return
    if any(hmac.compare_digest(authorization, accepted) for accepted in accepted_auth):
        return
    raise HTTPException(status_code=401, detail={"reason": "invalid api token"})


class LinearRAGMemoryService:
    def __init__(self) -> None:
        self.root = Path(os.getenv("LINEARRAG_STORAGE_DIR", "import_api")).resolve()
        self.users_root = self.root / "users"
        self.embedding_model_path = os.getenv("LINEARRAG_EMBEDDING_MODEL", "model/all-mpnet-base-v2")
        self.spacy_model_name = os.getenv("LINEARRAG_SPACY_MODEL", "en_core_web_trf")
        self.batch_size = int(os.getenv("LINEARRAG_BATCH_SIZE", "64"))
        self.max_workers = int(os.getenv("LINEARRAG_MAX_WORKERS", "4"))
        self.max_chunk_chars = int(os.getenv("LINEARRAG_MAX_CHUNK_CHARS", "3500"))
        self.max_cached_users = int(os.getenv("LINEARRAG_MAX_CACHED_USERS", "4"))
        self.default_top_k = int(os.getenv("LINEARRAG_DEFAULT_TOP_K", "100"))
        self.min_score = self._optional_float(os.getenv("LINEARRAG_MIN_SCORE"))
        self.use_vectorized_retrieval = self._env_bool("LINEARRAG_USE_VECTORIZED_RETRIEVAL", False)
        self.enable_attribute_fallback = self._env_bool("LINEARRAG_ENABLE_ATTRIBUTE_FALLBACK", True)

        self.users_root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.rag_cache: OrderedDict[str, LinearRAG] = OrderedDict()

        device = os.getenv("LINEARRAG_DEVICE") or self._default_device()
        self.embedding_model = SentenceTransformer(self.embedding_model_path, device=device)
        self.spacy_ner = SpacyNER(self.spacy_model_name)

    @staticmethod
    def _optional_float(value: str | None) -> float | None:
        if not value:
            return None
        return float(value)

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def add(self, request: AddRequest) -> dict[str, Any]:
        with self.lock:
            user_key = self._user_key(request.user_id)
            meta = self._load_meta(user_key, request.user_id)
            processed = meta.setdefault("processed_request_ids", {})
            if request.request_id in processed:
                return self._add_response(request)

            passages = self._build_passages(request, int(meta.get("next_seq", 0)))
            if passages:
                rag = self._new_rag(user_key)
                rag.index(passages)
                self._put_cached_rag(user_key, rag)
                meta["next_seq"] = int(meta.get("next_seq", 0)) + len(passages)

            processed[request.request_id] = {
                "session_id": request.session_id,
                "passage_count": len(passages),
            }
            self._save_meta(user_key, meta)
            return self._add_response(request)

    def search(self, request: SearchRequest) -> dict[str, Any]:
        with self.lock:
            user_key = self._user_key(request.user_id)
            rag = self._get_rag(user_key)
            if rag is None or not rag.passage_embedding_store.texts:
                return {"data": []}

            limit = min(max(int(request.top_k), 0), 100)
            if limit == 0:
                return {"data": []}

            query = self._query_text(request)
            rag.config.retrieval_top_k = limit
            results = rag.retrieve([{"question": query, "answer": ""}])[0]
            passages = results.get("sorted_passage", [])
            scores = results.get("sorted_passage_scores", [])

            if self.min_score is not None and scores and max(scores) < self.min_score:
                return {"data": []}

            data = []
            seen_ids: set[str] = set()
            for passage, score in zip(passages, scores):
                memory_id = rag.passage_embedding_store.text_to_hash_id.get(passage)
                if memory_id is None:
                    memory_id = "passage-" + hashlib.sha256(passage.encode("utf-8")).hexdigest()
                if memory_id in seen_ids:
                    continue
                seen_ids.add(memory_id)
                data.append(
                    {
                        "id": memory_id,
                        "content": self._strip_internal_prefix(passage),
                        "score": float(score),
                    }
                )
                if len(data) >= limit:
                    break
            return {"data": data}

    @staticmethod
    def _add_response(request: AddRequest) -> dict[str, Any]:
        return {
            "success": True,
            "request_id": request.request_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
        }

    @staticmethod
    def _user_key(user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]

    def _user_dir(self, user_key: str) -> Path:
        return self.users_root / user_key

    def _meta_path(self, user_key: str) -> Path:
        return self._user_dir(user_key) / "metadata.json"

    def _load_meta(self, user_key: str, user_id: str) -> dict[str, Any]:
        user_dir = self._user_dir(user_key)
        user_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._meta_path(user_key)
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {
            "user_id": user_id,
            "next_seq": 0,
            "processed_request_ids": {},
        }

    def _save_meta(self, user_key: str, meta: dict[str, Any]) -> None:
        meta_path = self._meta_path(user_key)
        tmp_path = meta_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(meta_path)

    def _build_passages(self, request: AddRequest, start_seq: int) -> list[str]:
        passages: list[str] = []
        seq = start_seq
        for message_index, message in enumerate(request.messages):
            role = message.role.strip() or "unknown"
            content = message.content.strip()
            if not content:
                continue
            for part_index, part in enumerate(self._split_content(content)):
                timestamp = "" if message.timestamp is None else str(message.timestamp)
                header = (
                    f"[session_id={request.session_id}] "
                    f"[request_id={request.request_id}] "
                    f"[message_index={message_index}] "
                    f"[part_index={part_index}] "
                    f"[role={role}]"
                )
                if timestamp:
                    header += f" [timestamp={timestamp}]"
                passages.append(f"{seq}:{header}\n{role}: {part}")
                seq += 1
        return passages

    def _split_content(self, content: str) -> list[str]:
        if len(content) <= self.max_chunk_chars:
            return [content]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for piece in re.split(r"(\n+|(?<=[.!?。！？])\s+)", content):
            if not piece:
                continue
            if current and current_len + len(piece) > self.max_chunk_chars:
                chunks.append("".join(current).strip())
                current = []
                current_len = 0
            if len(piece) > self.max_chunk_chars:
                for start in range(0, len(piece), self.max_chunk_chars):
                    sub = piece[start : start + self.max_chunk_chars].strip()
                    if sub:
                        chunks.append(sub)
                continue
            current.append(piece)
            current_len += len(piece)
        if current:
            chunks.append("".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _new_rag(self, user_key: str) -> LinearRAG:
        config = LinearRAGConfig(
            dataset_name=user_key,
            embedding_model=self.embedding_model,
            spacy_model=self.spacy_model_name,
            working_dir=str(self.users_root),
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            retrieval_top_k=self.default_top_k,
            use_vectorized_retrieval=self.use_vectorized_retrieval,
            enable_hybrid_attribute_fallback=self.enable_attribute_fallback,
        )
        config.spacy_ner = self.spacy_ner
        return LinearRAG(config)

    def _get_rag(self, user_key: str) -> LinearRAG | None:
        cached = self.rag_cache.get(user_key)
        if cached is not None:
            self.rag_cache.move_to_end(user_key)
            return cached

        user_dir = self._user_dir(user_key)
        if not user_dir.exists():
            return None
        rag = self._new_rag(user_key)
        if rag.passage_embedding_store.texts:
            rag.index(rag.passage_embedding_store.texts)
        self._put_cached_rag(user_key, rag)
        return rag

    def _put_cached_rag(self, user_key: str, rag: LinearRAG) -> None:
        self.rag_cache[user_key] = rag
        self.rag_cache.move_to_end(user_key)
        while len(self.rag_cache) > self.max_cached_users:
            self.rag_cache.popitem(last=False)

    @staticmethod
    def _query_text(request: SearchRequest) -> str:
        query = request.query.strip()
        if request.options:
            options = "\n".join(str(option) for option in request.options if str(option).strip())
            if options:
                query = f"{query}\nOptions:\n{options}"
        return query

    @staticmethod
    def _strip_internal_prefix(passage: str) -> str:
        return re.sub(r"^\d+:", "", passage, count=1).lstrip()


service: LinearRAGMemoryService | None = None


def get_service() -> LinearRAGMemoryService:
    global service
    if service is None:
        service = LinearRAGMemoryService()
    return service


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "success": True,
        "service": "linearrag-memory-api",
        "auth_required": bool(configured_api_token()),
    }


@app.post("/add", dependencies=[Depends(verify_api_key)])
def add_memory(
    request: AddRequest,
    memory: LinearRAGMemoryService = Depends(get_service),
) -> dict[str, Any]:
    return memory.add(request)


@app.post("/search", dependencies=[Depends(verify_api_key)])
def search_memory(
    request: SearchRequest,
    memory: LinearRAGMemoryService = Depends(get_service),
) -> dict[str, Any]:
    return memory.search(request)
