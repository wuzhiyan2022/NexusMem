from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import urllib.error
import urllib.request
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
        self.enable_evidence_rerank = self._env_bool("LINEARRAG_ENABLE_EVIDENCE_RERANK", True)
        self.enable_llm_query_intent = self._env_bool("LINEARRAG_ENABLE_LLM_QUERY_INTENT", True)
        self.query_llm_model = os.getenv("LINEARRAG_QUERY_LLM_MODEL", "gpt-4o-mini").strip()
        self.query_llm_base_url = (
            os.getenv("LINEARRAG_QUERY_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
        ).strip().rstrip("/")
        self.query_llm_api_key = (
            os.getenv("LINEARRAG_QUERY_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        ).strip()
        self.query_llm_timeout = float(os.getenv("LINEARRAG_QUERY_LLM_TIMEOUT", "4"))
        self.enable_llm_memory_extraction = self._env_bool("LINEARRAG_ENABLE_LLM_MEMORY_EXTRACTION", True)
        self.extract_llm_model = os.getenv("LINEARRAG_EXTRACT_LLM_MODEL", self.query_llm_model).strip()
        self.extract_llm_base_url = (
            os.getenv("LINEARRAG_EXTRACT_LLM_BASE_URL") or self.query_llm_base_url
        ).strip().rstrip("/")
        self.extract_llm_api_key = (
            os.getenv("LINEARRAG_EXTRACT_LLM_API_KEY") or self.query_llm_api_key
        ).strip()
        self.extract_llm_timeout = float(os.getenv("LINEARRAG_EXTRACT_LLM_TIMEOUT", "8"))
        self.extract_llm_batch_size = int(os.getenv("LINEARRAG_EXTRACT_LLM_BATCH_SIZE", "8"))
        self.extract_llm_max_tokens = int(os.getenv("LINEARRAG_EXTRACT_LLM_MAX_TOKENS", "1800"))
        self.enable_structured_memory_graph = self._env_bool("LINEARRAG_ENABLE_STRUCTURED_MEMORY_GRAPH", True)
        self.structured_node_top_k = int(os.getenv("LINEARRAG_STRUCTURED_NODE_TOP_K", "24"))
        self.structured_node_weight = float(os.getenv("LINEARRAG_STRUCTURED_NODE_WEIGHT", "0.35"))
        self.structured_event_weight = float(os.getenv("LINEARRAG_STRUCTURED_EVENT_WEIGHT", "1.0"))
        self.structured_memory_weight = float(os.getenv("LINEARRAG_STRUCTURED_MEMORY_WEIGHT", "0.9"))
        self.structured_time_weight = float(os.getenv("LINEARRAG_STRUCTURED_TIME_WEIGHT", "0.8"))
        self.structured_score_threshold = float(os.getenv("LINEARRAG_STRUCTURED_SCORE_THRESHOLD", "0.2"))
        self.rerank_candidate_multiplier = int(os.getenv("LINEARRAG_RERANK_CANDIDATE_MULTIPLIER", "3"))
        self.rerank_max_candidates = int(os.getenv("LINEARRAG_RERANK_MAX_CANDIDATES", "300"))
        self.enable_search_debug = self._env_bool("LINEARRAG_ENABLE_SEARCH_DEBUG", False)
        self.search_debug_path = Path(
            os.getenv("LINEARRAG_SEARCH_DEBUG_PATH", str(self.root / "search_debug.jsonl"))
        ).resolve()

        self.users_root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.rag_cache: OrderedDict[str, LinearRAG] = OrderedDict()
        self.query_intent_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

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

            query = request.query.strip()
            retrieve_limit = self._retrieve_limit(limit)
            rag.config.retrieval_top_k = retrieve_limit
            results = rag.retrieve([{"question": query, "answer": ""}])[0]
            passages = results.get("sorted_passage", [])
            scores = results.get("sorted_passage_scores", [])

            if self.min_score is not None and scores and max(scores) < self.min_score:
                return {"data": []}

            data = []
            seen_ids: set[str] = set()
            ranked_results = self._rank_search_results(user_key, query, request.options or [], passages, scores)
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
        llm_extractor = self._extract_llm_structured_memory if self._llm_memory_extraction_ready() else None
        return layer.build_passages(
            request,
            self._split_content,
            start_seq,
            self._split_sentences,
            self._extract_event_candidates,
            self._extract_generic_memory_candidates,
            llm_extractor,
        )

    def _llm_memory_extraction_ready(self) -> bool:
        return bool(
            self.enable_llm_memory_extraction
            and self.extract_llm_model
            and self.extract_llm_base_url
            and self.extract_llm_api_key
        )

    def _extract_llm_structured_memory(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        sentences = payload.get("sentences") or []
        if not isinstance(sentences, list) or not sentences:
            return None

        batch_size = max(1, self.extract_llm_batch_size)
        if len(sentences) > batch_size:
            merged: dict[str, list[Any]] = {"events": [], "memories": [], "times": [], "relations": []}
            for start in range(0, len(sentences), batch_size):
                batch_payload = dict(payload)
                batch_payload["sentences"] = sentences[start : start + batch_size]
                result = self._extract_llm_structured_memory_single(batch_payload)
                if not isinstance(result, dict):
                    continue
                for key in merged:
                    values = result.get(key) or []
                    if isinstance(values, list):
                        merged[key].extend(values)
            return merged

        return self._extract_llm_structured_memory_single(payload)

    def _extract_llm_structured_memory_single(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        url = self._chat_completions_url(self.extract_llm_base_url)
        system_prompt = (
            "You are an information extraction module for a long-term memory retrieval system. "
            "Extract only information explicitly supported by the input text. Return valid JSON only.\n\n"
            "Definitions:\n"
            "1. Event nodes: concrete actions, changes, decisions, experiences, or state transitions involving "
            "the user or a named participant. Do not extract events for generic facts, greetings, questions, "
            "instructions, vague future wishes, or simple preferences without action/change.\n"
            "2. Memory nodes: durable information useful for future user-related questions, including preferences, "
            "relationships, profile, habits, constraints, goals, stable states, important plans, and important "
            "past experiences. Do not extract trivial chat, one-time commands, generic world knowledge, vague "
            "speculation, or low-value duplicates.\n"
            "3. Time nodes: time expressions present in text, including absolute dates, relative time, sequence "
            "markers, duration, current-state cues, and history cues. Do not convert relative time to absolute "
            "dates. message_timestamp is only ordering metadata, not semantic event time.\n\n"
            "Every extracted item must include sentence_index, evidence_text, and confidence. evidence_text must "
            "be an exact substring of the corresponding sentence. Use empty arrays if nothing qualifies.\n\n"
            "Return this JSON schema exactly:\n"
            "{"
            "\"events\":[{\"sentence_index\":0,\"is_event\":true,\"event_worthiness\":0.0,"
            "\"subject\":\"\",\"trigger\":\"\",\"object\":\"\",\"event_type\":\"\",\"participants\":[],"
            "\"time_expressions\":[],\"evidence_text\":\"\",\"confidence\":0.0}],"
            "\"memories\":[{\"sentence_index\":0,\"is_memory\":true,\"memory_worthiness\":0.0,"
            "\"memory_type\":\"preference|relation|profile|fact|habit|constraint|goal|plan|state|location|work|education|experience\","
            "\"subject\":\"\",\"predicate\":\"\",\"object\":\"\",\"polarity\":\"positive|negative|neutral\","
            "\"stability\":\"current|history|stable|planned|unknown\",\"update_signal\":false,"
            "\"time_expressions\":[],\"evidence_text\":\"\",\"confidence\":0.0}],"
            "\"times\":[{\"sentence_index\":0,\"text\":\"\",\"time_type\":\"absolute_date|year|relative|sequence|duration|current|history|unknown\","
            "\"temporal_role\":\"event_time|memory_time|sequence_marker|duration|current_marker|history_marker|unknown\","
            "\"sequence_direction\":\"before|after|\",\"evidence_text\":\"\",\"confidence\":0.0}],"
            "\"relations\":[]"
            "}"
        )
        user_payload = {
            "raw_id": payload.get("raw_id"),
            "role": payload.get("role"),
            "speaker": payload.get("speaker"),
            "actor": payload.get("actor"),
            "message_timestamp": payload.get("message_timestamp"),
            "sentences": payload.get("sentences") or [],
        }
        request_payload = {
            "model": self.extract_llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": self.extract_llm_max_tokens,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.extract_llm_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.extract_llm_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            data = self._parse_json_object(content)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return {
            "events": data.get("events") if isinstance(data.get("events"), list) else [],
            "memories": data.get("memories") if isinstance(data.get("memories"), list) else [],
            "times": data.get("times") if isinstance(data.get("times"), list) else [],
            "relations": data.get("relations") if isinstance(data.get("relations"), list) else [],
        }

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
        has_status_rerank = self.enable_memory_layer and self.enable_status_rerank
        if not (has_status_rerank or self.enable_evidence_rerank):
            return limit
        multiplier = max(1, self.rerank_candidate_multiplier)
        max_candidates = max(limit, self.rerank_max_candidates)
        return min(max_candidates, max(limit, limit * multiplier))

    def _rank_search_results(
        self,
        user_key: str,
        query: str,
        options: list[str],
        passages: list[str],
        scores: list[float],
    ) -> list[tuple[str, float]]:
        ranked = [(passage, float(score)) for passage, score in zip(passages, scores)]
        if not ranked:
            return ranked

        raw_status: dict[str, Any] = {}
        raw_temporal_profiles: dict[str, dict[str, Any]] = {}
        intent = MemoryLayer.query_intent(query)
        temporal_intent = self._query_temporal_intent(query)
        if self.enable_memory_layer and (self.enable_status_rerank or self.enable_evidence_rerank):
            try:
                layer = MemoryLayer(self._user_dir(user_key), max_chunk_chars=self.max_chunk_chars)
                if self.enable_status_rerank:
                    raw_status = layer.load_raw_memory_status()
                if self.enable_evidence_rerank:
                    raw_temporal_profiles = layer.load_raw_temporal_profiles()
            except Exception:
                raw_status = {}
                raw_temporal_profiles = {}

        passage_profiles: dict[str, dict[str, Any]] = {}
        for passage, _ in ranked:
            raw_id = MemoryLayer.parse_raw_id(passage)
            if raw_id and raw_id in raw_temporal_profiles:
                passage_profiles[passage] = raw_temporal_profiles[raw_id]
        temporal_context = self._temporal_context(passage_profiles.values())
        anchor_context = self._anchor_temporal_context(temporal_intent, ranked, passage_profiles)

        adjusted = []
        for passage, score in ranked:
            adjusted_score = score
            raw_id = MemoryLayer.parse_raw_id(passage)
            if raw_status:
                status_info = raw_status.get(raw_id) if raw_id else None
                adjusted_score = MemoryLayer.adjust_score(adjusted_score, status_info, intent)
            if self.enable_evidence_rerank:
                temporal_profile = passage_profiles.get(passage, {})
                adjusted_score *= self._evidence_rerank_factor(
                    query,
                    options,
                    passage,
                    temporal_intent,
                    temporal_profile,
                    temporal_context,
                    anchor_context,
                )
            adjusted.append((passage, adjusted_score))
        adjusted.sort(key=lambda item: item[1], reverse=True)
        self._write_search_debug(
            user_key,
            query,
            options,
            temporal_intent,
            ranked,
            adjusted,
            passage_profiles,
            temporal_context,
            anchor_context,
        )
        return adjusted

    def _evidence_rerank_factor(
        self,
        query: str,
        options: list[str],
        passage: str,
        temporal_intent: dict[str, Any] | None = None,
        temporal_profile: dict[str, Any] | None = None,
        temporal_context: dict[str, float] | None = None,
        anchor_context: dict[str, Any] | None = None,
    ) -> float:
        content = self._strip_internal_prefix(passage)
        content_lower = content.lower()
        query_tokens = self._rerank_tokens(query)
        passage_tokens = self._rerank_tokens(content)
        factor = 1.0

        if query_tokens and passage_tokens:
            overlap = len(query_tokens & passage_tokens)
            coverage = overlap / max(len(query_tokens), 1)
            factor += min(0.20, 0.025 * overlap)
            factor += min(0.15, 0.12 * coverage)

        phrase_hits = sum(1 for phrase in self._important_phrases(query) if phrase in content_lower)
        if phrase_hits:
            factor += min(0.12, 0.04 * phrase_hits)

        query_times = self._time_values(query)
        if query_times:
            exact_time_hits = sum(1 for value in query_times if value in content_lower)
            factor += min(0.18, 0.08 * exact_time_hits)
        if self._is_temporal_query(query) and self._has_time_expression(content):
            factor += 0.05

        factor += self._best_option_match(options, passage_tokens, content_lower)
        factor *= self._target_rerank_factor(temporal_intent or {}, passage_tokens, content_lower)
        factor *= self._temporal_rerank_factor(
            temporal_intent or {},
            temporal_profile or {},
            temporal_context or {},
            anchor_context or {},
        )
        return max(0.65, min(2.05, factor))

    def _query_temporal_intent(self, query: str) -> dict[str, Any]:
        cache_key = query.strip().lower()
        cached = self.query_intent_cache.get(cache_key)
        if cached is not None:
            self.query_intent_cache.move_to_end(cache_key)
            return cached

        intent = self._rule_query_temporal_intent(query)
        if self.enable_llm_query_intent and self.query_llm_api_key and self.query_llm_base_url:
            llm_intent = self._llm_query_temporal_intent(query)
            if llm_intent:
                intent.update(llm_intent)
                intent["source"] = "llm"
        self.query_intent_cache[cache_key] = intent
        self.query_intent_cache.move_to_end(cache_key)
        while len(self.query_intent_cache) > 256:
            self.query_intent_cache.popitem(last=False)
        return intent

    @staticmethod
    def _rule_query_temporal_intent(query: str) -> dict[str, Any]:
        lowered = query.lower()
        temporal_intent = "neutral"
        if re.search(r"\bhow long|since when|for how many|duration\b", lowered):
            temporal_intent = "duration"
        elif re.search(r"\bbefore|prior to|earlier than\b", lowered):
            temporal_intent = "sequence_before"
        elif re.search(r"\bafter|following|later than|subsequently|since\b", lowered):
            temporal_intent = "sequence_after"
        elif re.search(r"\bcurrently|current|now|latest|still|anymore|at the moment\b", lowered):
            temporal_intent = "current_state"
        elif re.search(r"\bpreviously|used to|formerly|past|once|before\b", lowered):
            temporal_intent = "history_state"
        elif re.search(r"\brecently|most recent|latest\b", lowered):
            temporal_intent = "recent"
        elif re.search(r"\bwhen|what date|which date|what day|which day|what month|what year|date|time\b", lowered):
            temporal_intent = "when_exact"
        return {
            "temporal_intent": temporal_intent,
            "prefer_current": temporal_intent in {"current_state", "recent", "sequence_after"},
            "prefer_history": temporal_intent in {"history_state", "sequence_before"},
            "prefer_recent": temporal_intent in {"recent", "current_state"},
            "needs_explicit_time": temporal_intent in {"when_exact", "duration"},
            "sequence_direction": "before" if temporal_intent == "sequence_before" else "after" if temporal_intent == "sequence_after" else "",
            "anchor_event": "",
            "target": "",
            "confidence": 0.45 if temporal_intent != "neutral" else 0.2,
            "source": "rule",
        }

    def _llm_query_temporal_intent(self, query: str) -> dict[str, Any] | None:
        url = self._chat_completions_url(self.query_llm_base_url)
        prompt = (
            "You parse the temporal intent of a memory search query. "
            "Return only a compact JSON object. Allowed temporal_intent values: "
            "when_exact, current_state, history_state, sequence_before, sequence_after, "
            "recent, duration, neutral. Include keys: temporal_intent, target, anchor_event, "
            "prefer_current, prefer_history, prefer_recent, needs_explicit_time, "
            "sequence_direction, confidence."
        )
        payload = {
            "model": self.query_llm_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "max_tokens": 180,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.query_llm_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.query_llm_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            data = self._parse_json_object(content)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        allowed = {
            "when_exact", "current_state", "history_state", "sequence_before", "sequence_after",
            "recent", "duration", "neutral",
        }
        temporal_intent = str(data.get("temporal_intent") or "neutral").strip()
        if temporal_intent not in allowed:
            temporal_intent = "neutral"
        return {
            "temporal_intent": temporal_intent,
            "target": str(data.get("target") or ""),
            "anchor_event": str(data.get("anchor_event") or ""),
            "prefer_current": bool(data.get("prefer_current")),
            "prefer_history": bool(data.get("prefer_history")),
            "prefer_recent": bool(data.get("prefer_recent")),
            "needs_explicit_time": bool(data.get("needs_explicit_time")),
            "sequence_direction": str(data.get("sequence_direction") or ""),
            "confidence": max(0.0, min(1.0, float(data.get("confidence") or 0.0))),
        }

    @staticmethod
    def _chat_completions_url(base_url: str) -> str:
        value = base_url.rstrip("/")
        if value.endswith("/chat/completions"):
            return value
        return value + "/chat/completions"

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)
        try:
            data = json.loads(cleaned)
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _temporal_context(profiles: Any) -> dict[str, float]:
        values = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            value = MemoryLayer._time_value(profile.get("message_timestamp"))
            if not value:
                value = MemoryLayer._time_value(profile.get("order_index"))
            if value:
                values.append(value)
        if not values:
            return {"min": 0.0, "max": 0.0}
        return {"min": min(values), "max": max(values)}

    @staticmethod
    def _temporal_rerank_factor(
        temporal_intent: dict[str, Any],
        profile: dict[str, Any],
        temporal_context: dict[str, float],
        anchor_context: dict[str, Any] | None = None,
    ) -> float:
        intent = str(temporal_intent.get("temporal_intent") or "neutral")
        if intent == "neutral":
            return 1.0
        cues = set(profile.get("temporal_cues") or [])
        kinds = set(profile.get("time_kinds") or [])
        has_time = bool(profile.get("has_explicit_time"))
        confidence = max(0.0, min(1.0, float(temporal_intent.get("confidence") or 0.0)))
        factor = 1.0

        if intent == "when_exact":
            if has_time:
                factor += 0.10
            if kinds & {"date", "year", "clock_time"}:
                factor += 0.08
            elif kinds & {"relative"}:
                factor += 0.04
        elif intent == "duration":
            if "duration_signal" in cues:
                factor += 0.14
            if has_time:
                factor += 0.05
        elif intent == "current_state":
            factor += 0.10 * LinearRAGMemoryService._candidate_recency(profile, temporal_context)
            if "current_signal" in cues:
                factor += 0.10
            if "history_signal" in cues:
                factor -= 0.08
        elif intent == "history_state":
            factor += 0.10 * (1.0 - LinearRAGMemoryService._candidate_recency(profile, temporal_context))
            if "history_signal" in cues:
                factor += 0.10
            if "current_signal" in cues:
                factor -= 0.06
        elif intent == "sequence_before":
            anchor_factor = LinearRAGMemoryService._anchor_sequence_factor(
                profile,
                temporal_context,
                anchor_context or {},
                direction="before",
                confidence=confidence,
            )
            factor *= anchor_factor
            if not anchor_context:
                factor += 0.08 * (1.0 - LinearRAGMemoryService._candidate_recency(profile, temporal_context))
            if cues & {"sequence_before", "history_signal"}:
                factor += 0.08
            if "current_signal" in cues:
                factor -= 0.05
        elif intent == "sequence_after":
            anchor_factor = LinearRAGMemoryService._anchor_sequence_factor(
                profile,
                temporal_context,
                anchor_context or {},
                direction="after",
                confidence=confidence,
            )
            factor *= anchor_factor
            if not anchor_context:
                factor += 0.08 * LinearRAGMemoryService._candidate_recency(profile, temporal_context)
            if cues & {"sequence_after", "current_signal"}:
                factor += 0.08
        elif intent == "recent":
            factor += 0.12 * LinearRAGMemoryService._candidate_recency(profile, temporal_context)
            if "current_signal" in cues:
                factor += 0.08
            if "history_signal" in cues:
                factor -= 0.05

        return max(0.78, min(1.42, factor))

    @classmethod
    def _target_rerank_factor(
        cls,
        temporal_intent: dict[str, Any],
        passage_tokens: set[str],
        content_lower: str,
    ) -> float:
        factor = 1.0
        target = str(temporal_intent.get("target") or "").strip()
        if target:
            target_tokens = cls._rerank_tokens(target)
            if target_tokens and passage_tokens:
                overlap = len(target_tokens & passage_tokens)
                coverage = overlap / max(len(target_tokens), 1)
                factor += min(0.20, 0.045 * overlap + 0.12 * coverage)
            target_phrase = target.lower()
            if len(target_phrase) > 4 and target_phrase in content_lower:
                factor += 0.08

        anchor_event = str(temporal_intent.get("anchor_event") or "").strip()
        intent = str(temporal_intent.get("temporal_intent") or "")
        if anchor_event and intent not in {"sequence_before", "sequence_after"}:
            anchor_tokens = cls._rerank_tokens(anchor_event)
            if anchor_tokens and passage_tokens:
                overlap = len(anchor_tokens & passage_tokens)
                coverage = overlap / max(len(anchor_tokens), 1)
                factor += min(0.12, 0.035 * overlap + 0.06 * coverage)

        return max(0.90, min(1.32, factor))

    @classmethod
    def _anchor_temporal_context(
        cls,
        temporal_intent: dict[str, Any],
        ranked: list[tuple[str, float]],
        passage_profiles: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        intent = str(temporal_intent.get("temporal_intent") or "")
        if intent not in {"sequence_before", "sequence_after"}:
            return {}
        anchor_event = str(temporal_intent.get("anchor_event") or "").strip()
        if not anchor_event:
            return {}

        anchor_tokens = cls._rerank_tokens(anchor_event)
        if not anchor_tokens:
            return {}

        best: dict[str, Any] = {}
        best_score = 0.0
        for rank, (passage, base_score) in enumerate(ranked):
            profile = passage_profiles.get(passage) or {}
            value = cls._profile_time_value(profile)
            if not value:
                continue
            content = cls._strip_internal_prefix(passage).lower()
            passage_tokens = cls._rerank_tokens(content)
            if not passage_tokens:
                continue
            overlap = len(anchor_tokens & passage_tokens)
            coverage = overlap / max(len(anchor_tokens), 1)
            score = 0.0
            if overlap:
                score += 0.45 * overlap + 0.80 * coverage
            anchor_phrase = anchor_event.lower()
            if len(anchor_phrase) > 4 and anchor_phrase in content:
                score += 1.0
            score += min(0.25, max(float(base_score), 0.0) * 0.05)
            score -= min(0.20, rank * 0.01)
            if score > best_score:
                best_score = score
                best = {
                    "anchor_value": value,
                    "anchor_raw_id": MemoryLayer.parse_raw_id(passage) or "",
                    "anchor_score": score,
                    "anchor_event": anchor_event,
                    "direction": "before" if intent == "sequence_before" else "after",
                }
        return best if best_score >= 0.55 else {}

    @staticmethod
    def _anchor_sequence_factor(
        profile: dict[str, Any],
        temporal_context: dict[str, float],
        anchor_context: dict[str, Any],
        direction: str,
        confidence: float,
    ) -> float:
        anchor_value = float(anchor_context.get("anchor_value") or 0.0)
        value = LinearRAGMemoryService._profile_time_value(profile)
        if not anchor_value or not value:
            return 1.0
        if value == anchor_value:
            return 0.86

        side_matches = value < anchor_value if direction == "before" else value > anchor_value
        low = float(temporal_context.get("min") or 0.0)
        high = float(temporal_context.get("max") or 0.0)
        span = max(1.0, high - low)
        distance = min(1.0, abs(value - anchor_value) / span)
        strength = 0.12 + 0.16 * confidence
        if side_matches:
            return 1.0 + strength * (0.65 + 0.35 * distance)
        return 1.0 - min(0.24, strength * (0.75 + 0.25 * distance))

    @staticmethod
    def _profile_time_value(profile: dict[str, Any]) -> float:
        value = MemoryLayer._time_value(profile.get("message_timestamp"))
        if not value:
            value = MemoryLayer._time_value(profile.get("order_index"))
        return value

    @staticmethod
    def _candidate_recency(profile: dict[str, Any], temporal_context: dict[str, float]) -> float:
        value = LinearRAGMemoryService._profile_time_value(profile)
        low = float(temporal_context.get("min") or 0.0)
        high = float(temporal_context.get("max") or 0.0)
        if not value or high <= low:
            return 0.5
        return max(0.0, min(1.0, (value - low) / (high - low)))

    def _write_search_debug(
        self,
        user_key: str,
        query: str,
        options: list[str],
        temporal_intent: dict[str, Any],
        ranked: list[tuple[str, float]],
        adjusted: list[tuple[str, float]],
        passage_profiles: dict[str, dict[str, Any]],
        temporal_context: dict[str, float],
        anchor_context: dict[str, Any],
    ) -> None:
        if not self.enable_search_debug:
            return
        try:
            before_rank = {passage: index for index, (passage, _) in enumerate(ranked, start=1)}
            score_by_passage = {passage: score for passage, score in ranked}
            rows = []
            for index, (passage, adjusted_score) in enumerate(adjusted[:30], start=1):
                raw_id = MemoryLayer.parse_raw_id(passage) or ""
                profile = passage_profiles.get(passage) or {}
                rows.append(
                    {
                        "rank_after": index,
                        "rank_before": before_rank.get(passage),
                        "raw_id": raw_id,
                        "base_score": score_by_passage.get(passage),
                        "adjusted_score": adjusted_score,
                        "temporal_profile": profile,
                        "content_preview": self._strip_internal_prefix(passage)[:240],
                    }
                )
            self.search_debug_path.parent.mkdir(parents=True, exist_ok=True)
            with self.search_debug_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "user_key": user_key,
                            "query": query,
                            "options": options,
                            "temporal_intent": temporal_intent,
                            "temporal_context": temporal_context,
                            "anchor_context": anchor_context,
                            "candidates": rows,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            return

    @classmethod
    def _rerank_tokens(cls, text: str) -> set[str]:
        stopwords = {
            "the", "and", "for", "with", "which", "what", "when", "where", "who", "why", "how",
            "did", "does", "do", "was", "were", "is", "are", "am", "be", "been", "being",
            "to", "of", "in", "on", "at", "by", "from", "as", "a", "an", "or", "if", "then",
            "i", "me", "my", "mine", "we", "our", "ours", "you", "your", "yours", "user",
            "answer", "answers", "best", "match", "matches", "memory", "option", "first", "second",
            "third", "fourth", "following", "about", "tell", "ask", "asked", "said", "say",
        }
        tokens = set()
        for token in re.findall(r"[a-z0-9][a-z0-9'_-]*", text.lower()):
            normalized = cls._normalize_rerank_token(token.strip("'_-"))
            if len(normalized) > 2 and normalized not in stopwords:
                tokens.add(normalized)
        return tokens

    @staticmethod
    def _normalize_rerank_token(token: str) -> str:
        if len(token) > 5 and token.endswith("ies"):
            return token[:-3] + "y"
        preference_tokens = {
            "preference": "prefer",
            "preferences": "prefer",
            "preferred": "prefer",
            "prefers": "prefer",
            "preferring": "prefer",
        }
        if token in preference_tokens:
            return preference_tokens[token]
        if len(token) > 4 and token.endswith("ing"):
            base = token[:-3]
            if len(base) > 2 and base[-1] == base[-2]:
                base = base[:-1]
            return base
        if len(token) > 4 and token.endswith("ed"):
            base = token[:-2]
            if base.endswith("v"):
                base += "e"
            if len(base) > 2 and base[-1] == base[-2]:
                base = base[:-1]
            return base
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    @classmethod
    def _important_phrases(cls, text: str) -> list[str]:
        phrases = []
        for phrase in re.findall(r"\b[A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*){0,3}\b", text):
            normalized = phrase.lower().strip()
            if len(normalized) > 2 and normalized not in {"which", "what", "when", "where", "who", "options"}:
                phrases.append(normalized)
        return phrases

    @classmethod
    def _best_option_match(cls, options: list[str], passage_tokens: set[str], content_lower: str) -> float:
        best = 0.0
        for option in options or []:
            option_text = cls._strip_option_label(str(option))
            option_tokens = cls._rerank_tokens(option_text)
            if not option_tokens:
                continue
            overlap = len(option_tokens & passage_tokens)
            coverage = overlap / max(len(option_tokens), 1)
            score = min(0.16, 0.04 * overlap + 0.08 * coverage)
            option_lower = option_text.lower().strip()
            if len(option_lower) > 3 and option_lower in content_lower:
                score += 0.06
            best = max(best, score)
        return min(0.22, best)

    @staticmethod
    def _strip_option_label(option: str) -> str:
        return re.sub(r"^\s*(?:[A-Z]|\d+)[\).:\-]\s*", "", option.strip())

    @classmethod
    def _time_values(cls, text: str) -> list[str]:
        patterns = [
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            r"\b(?:19|20)\d{2}\b",
            r"\b\d{1,2}:\d{2}(?:\s?[ap]\.?m\.?)?\b",
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?\b",
            r"\b(?:today|yesterday|tomorrow|tonight|last week|next week|last month|next month|last year|next year|recently|currently|now|before|previously|earlier|later)\b",
        ]
        values = []
        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                value = str(match).strip().lower()
                if value and value not in values:
                    values.append(value)
        return values

    @classmethod
    def _has_time_expression(cls, text: str) -> bool:
        return bool(cls._time_values(text))

    @staticmethod
    def _is_temporal_query(query: str) -> bool:
        lowered = query.lower()
        return any(token in lowered for token in ("when", "date", "day", "month", "year", "time", "before", "after", "earlier", "later"))

    @staticmethod
    def _strip_internal_prefix(passage: str) -> str:
        text = re.sub(r"^\d+:", "", passage, count=1).lstrip()
        lines = text.splitlines()
        if not lines:
            return text.strip()

        first_line = lines[0].strip()
        if re.fullmatch(r"(?:\[[^\]]+\]\s*)+", first_line):
            return "\n".join(lines[1:]).strip()

        text = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", text).lstrip()
        return text.strip()


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
