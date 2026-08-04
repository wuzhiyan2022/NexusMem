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
from src.memory_layer import MemoryLayer
from src.ner import SpacyNER


class Message(BaseModel):
    role: str
    content: str
    timestamp: int | float | str | None = None
    speaker: str | None = None

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
        self.enable_memory_layer = self._env_bool("LINEARRAG_ENABLE_MEMORY_LAYER", True)
        self.enable_status_rerank = self._env_bool("LINEARRAG_ENABLE_STATUS_RERANK", True)
        self.enable_structured_memory_graph = self._env_bool("LINEARRAG_ENABLE_STRUCTURED_MEMORY_GRAPH", True)
        self.structured_node_top_k = int(os.getenv("LINEARRAG_STRUCTURED_NODE_TOP_K", "24"))
        self.structured_node_weight = float(os.getenv("LINEARRAG_STRUCTURED_NODE_WEIGHT", "0.35"))
        self.structured_event_weight = float(os.getenv("LINEARRAG_STRUCTURED_EVENT_WEIGHT", "1.0"))
        self.structured_memory_weight = float(os.getenv("LINEARRAG_STRUCTURED_MEMORY_WEIGHT", "0.9"))
        self.structured_time_weight = float(os.getenv("LINEARRAG_STRUCTURED_TIME_WEIGHT", "0.8"))
        self.structured_score_threshold = float(os.getenv("LINEARRAG_STRUCTURED_SCORE_THRESHOLD", "0.2"))
        self.rerank_candidate_multiplier = int(os.getenv("LINEARRAG_RERANK_CANDIDATE_MULTIPLIER", "3"))
        self.rerank_max_candidates = int(os.getenv("LINEARRAG_RERANK_MAX_CANDIDATES", "300"))

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

            passages = self._build_passages_for_index(user_key, request, int(meta.get("next_seq", 0)))
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
            retrieve_limit = self._retrieve_limit(limit)
            rag.config.retrieval_top_k = retrieve_limit
            results = rag.retrieve([{"question": query, "answer": ""}])[0]
            passages = results.get("sorted_passage", [])
            scores = results.get("sorted_passage_scores", [])

            if self.min_score is not None and scores and max(scores) < self.min_score:
                return {"data": []}

            data = []
            seen_ids: set[str] = set()
            ranked_results = self._rank_search_results(user_key, query, passages, scores)
            for passage, score in ranked_results:
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

    def _build_passages_for_index(self, user_key: str, request: AddRequest, start_seq: int) -> list[str]:
        if not self.enable_memory_layer:
            return self._build_passages(request, start_seq)

        layer = MemoryLayer(self._user_dir(user_key), max_chunk_chars=self.max_chunk_chars)
        return layer.build_passages(
            request,
            self._split_content,
            start_seq,
            self._split_sentences,
            self._extract_event_candidates,
            self._extract_generic_memory_candidates,
        )

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

    def _split_sentences(self, content: str) -> list[str]:
        try:
            doc = self.spacy_ner.spacy_model(content)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            return sentences or [content]
        except Exception:
            return MemoryLayer._split_sentences(content)

    def _extract_event_candidates(self, sentence: str) -> list[dict[str, Any]]:
        try:
            doc = self.spacy_ner.spacy_model(sentence)
            candidates = MemoryLayer.event_candidates_from_spacy(doc)
            return candidates or MemoryLayer._event_candidates(sentence)
        except Exception:
            return MemoryLayer._event_candidates(sentence)

    def _extract_generic_memory_candidates(self, sentence: str, speaker: str) -> list[dict[str, Any]]:
        try:
            doc = self.spacy_ner.spacy_model(sentence)
            return MemoryLayer.generic_candidates_from_spacy(doc, speaker=speaker)
        except Exception:
            return []

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
            enable_structured_memory_graph=self.enable_structured_memory_graph,
            structured_node_top_k=self.structured_node_top_k,
            structured_node_weight=self.structured_node_weight,
            structured_event_weight=self.structured_event_weight,
            structured_memory_weight=self.structured_memory_weight,
            structured_time_weight=self.structured_time_weight,
            structured_score_threshold=self.structured_score_threshold,
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

    def _retrieve_limit(self, limit: int) -> int:
        if not (self.enable_memory_layer and self.enable_status_rerank):
            return limit
        multiplier = max(1, self.rerank_candidate_multiplier)
        max_candidates = max(limit, self.rerank_max_candidates)
        return min(max_candidates, max(limit, limit * multiplier))

    def _rank_search_results(
        self,
        user_key: str,
        query: str,
        passages: list[str],
        scores: list[float],
    ) -> list[tuple[str, float]]:
        ranked = [(passage, float(score)) for passage, score in zip(passages, scores)]
        if not (self.enable_memory_layer and self.enable_status_rerank):
            return ranked

        try:
            layer = MemoryLayer(self._user_dir(user_key), max_chunk_chars=self.max_chunk_chars)
            raw_status = layer.load_raw_memory_status()
            intent = MemoryLayer.query_intent(query)
            adjusted = []
            for passage, score in ranked:
                raw_id = MemoryLayer.parse_raw_id(passage)
                status_info = raw_status.get(raw_id) if raw_id else None
                adjusted.append((passage, MemoryLayer.adjust_score(score, status_info, intent)))
            adjusted.sort(key=lambda item: item[1], reverse=True)
            return adjusted
        except Exception:
            return ranked

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
