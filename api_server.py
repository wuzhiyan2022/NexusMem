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
        self.evidence_signal_weight = float(os.getenv("LINEARRAG_EVIDENCE_SIGNAL_WEIGHT", "2.25"))
        self.enable_evidence_completion = self._env_bool("LINEARRAG_ENABLE_EVIDENCE_COMPLETION", True)
        self.completion_anchor_top_n = int(os.getenv("LINEARRAG_COMPLETION_ANCHOR_TOP_N", "12"))
        self.completion_window = int(os.getenv("LINEARRAG_COMPLETION_WINDOW", "2"))
        self.completion_max_neighbors = int(os.getenv("LINEARRAG_COMPLETION_MAX_NEIGHBORS", "2"))
        self.completion_related_top_n = int(os.getenv("LINEARRAG_COMPLETION_RELATED_TOP_N", "12"))
        self.completion_max_related = int(os.getenv("LINEARRAG_COMPLETION_MAX_RELATED", "3"))
        self.completion_min_related_fit = float(os.getenv("LINEARRAG_COMPLETION_MIN_RELATED_FIT", "0.24"))
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

            raw_query = request.query.strip()
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
            ranked_results = self._rank_search_results(user_key, raw_query, request.options or [], passages, scores, limit)
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
        limit: int,
    ) -> list[tuple[str, float]]:
        ranked = [(passage, float(score)) for passage, score in zip(passages, scores)]
        if not ranked:
            return ranked

        query_profile = self._query_evidence_profile(query, options)
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

        max_base_score = max([abs(float(score)) for _, score in ranked] or [1.0]) or 1.0
        adjusted = []
        for rank_index, (passage, score) in enumerate(ranked, start=1):
            adjusted_score = score
            raw_id = MemoryLayer.parse_raw_id(passage)
            if raw_status:
                status_info = raw_status.get(raw_id) if raw_id else None
                adjusted_score = MemoryLayer.adjust_score(adjusted_score, status_info, intent)
            if self.enable_evidence_rerank:
                temporal_profile = passage_profiles.get(passage, {})
                evidence_signal = self._direct_evidence_score(
                    query,
                    options,
                    passage,
                    query_profile,
                    temporal_intent,
                    temporal_profile,
                )
                adjusted_score *= self._evidence_rerank_factor(
                    query,
                    options,
                    passage,
                    query_profile,
                    temporal_intent,
                    temporal_profile,
                    temporal_context,
                    anchor_context,
                )
                rank_prior = 1.0 / max(rank_index, 1) ** 0.35
                adjusted_score += max_base_score * self.evidence_signal_weight * evidence_signal
                adjusted_score += max_base_score * 0.08 * rank_prior
            adjusted.append((passage, adjusted_score))
        adjusted.sort(key=lambda item: item[1], reverse=True)
        adjusted = self._complete_neighbor_evidence(query_profile, adjusted, limit)
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
        query_profile: dict[str, Any],
        temporal_intent: dict[str, Any] | None = None,
        temporal_profile: dict[str, Any] | None = None,
        temporal_context: dict[str, float] | None = None,
        anchor_context: dict[str, Any] | None = None,
    ) -> float:
        metadata = self._passage_metadata(passage)
        content = metadata.get("evidence_text") or self._strip_internal_prefix(passage)
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
        factor *= self._speaker_rerank_factor(query_profile, metadata)
        factor *= self._semantic_evidence_factor(query_profile, passage_tokens, content_lower)
        factor *= self._answer_type_rerank_factor(query_profile, content, metadata)
        factor *= self._target_rerank_factor(temporal_intent or {}, passage_tokens, content_lower)
        factor *= self._temporal_rerank_factor(
            temporal_intent or {},
            temporal_profile or {},
            temporal_context or {},
            anchor_context or {},
        )
        return max(0.65, min(2.05, factor))

    def _direct_evidence_score(
        self,
        query: str,
        options: list[str],
        passage: str,
        query_profile: dict[str, Any],
        temporal_intent: dict[str, Any] | None,
        temporal_profile: dict[str, Any] | None,
    ) -> float:
        metadata = self._passage_metadata(passage)
        content = str(metadata.get("evidence_text") or self._strip_internal_prefix(passage))
        content_lower = content.lower()
        passage_tokens = self._rerank_tokens(content)
        topic_tokens = set(query_profile.get("topic_tokens") or [])

        score = 0.0
        if topic_tokens and passage_tokens:
            overlap = topic_tokens & passage_tokens
            coverage = len(overlap) / max(len(topic_tokens), 1)
            score += min(0.42, 0.08 * len(overlap) + 0.28 * coverage)
        elif not topic_tokens:
            score += 0.06

        target_speakers = [
            self._normalize_person_name(value)
            for value in query_profile.get("target_speakers") or []
            if self._normalize_person_name(value)
        ]
        if target_speakers:
            speaker = self._normalize_person_name(metadata.get("speaker", ""))
            if speaker in target_speakers:
                score += 0.24
            elif speaker:
                score -= 0.10
            elif any(target in content_lower for target in target_speakers):
                score += 0.08

        score += 0.24 * self._answer_type_signal(query_profile, content, metadata)

        for phrase in self._important_phrases(query):
            if phrase in content_lower:
                score += 0.04

        intent = temporal_intent or {}
        for field, weight in (("target", 0.12), ("anchor_event", 0.08)):
            value = str(intent.get(field) or "").strip()
            if not value:
                continue
            target_tokens = self._rerank_tokens(value)
            if target_tokens and passage_tokens:
                coverage = len(target_tokens & passage_tokens) / max(len(target_tokens), 1)
                score += weight * coverage
            if len(value) > 4 and value.lower() in content_lower:
                score += min(weight, 0.08)

        if options:
            score += min(0.10, self._best_option_match(options, passage_tokens, content_lower) * 0.6)

        if temporal_profile:
            if query_profile.get("answer_type") == "time" and temporal_profile.get("has_explicit_time"):
                score += 0.06
            if query_profile.get("answer_type") in {"time", "duration"} and temporal_profile.get("has_relative_time"):
                score += 0.04

        if query_profile.get("answer_type") == "time":
            has_text_time = self._has_time_expression(content)
            if has_text_time:
                score += 0.12
            elif metadata.get("memory_timestamp") or metadata.get("memory_time_iso") or metadata.get("session_time"):
                score -= 0.04
            if not has_text_time:
                score *= 0.72

        return max(0.0, min(1.0, score))

    @classmethod
    def _answer_type_signal(cls, query_profile: dict[str, Any], content: str, metadata: dict[str, Any]) -> float:
        answer_type = str(query_profile.get("answer_type") or "generic")
        lowered = str(content or "").lower()
        if answer_type == "time":
            signal = 0.0
            has_text_time = cls._has_time_expression(content)
            has_relative = cls._has_relative_time_expression(content)
            has_structured_time = bool(
                metadata.get("session_time")
                or metadata.get("memory_timestamp")
                or metadata.get("memory_time_iso")
            )
            if has_text_time:
                signal += 0.80
            if has_relative:
                signal += 0.20
            if has_structured_time:
                signal += 0.10 if (has_text_time or has_relative) else 0.05
            return min(1.0, signal)
        if answer_type == "duration":
            return 1.0 if re.search(r"\b(?:for|since)\b|\b\d+\s+(?:years?|months?|weeks?|days?|hours?)\b", lowered) else 0.0
        if answer_type == "location":
            return 1.0 if re.search(
                r"\b(?:from|to|in|at|near|moved|move|home country|city|town|country|state|beach|mountains?|forest|sweden)\b",
                lowered,
            ) else 0.0
        if answer_type == "reason":
            return 1.0 if re.search(
                r"\b(?:because|since|so that|due to|thanks to|as a result|made me|helped me|realized|learnt|learned)\b",
                lowered,
            ) else 0.0
        if answer_type == "relation":
            return 1.0 if re.search(
                r"\b(?:single|married|wife|husband|partner|friend|family|colleague|parent|breakup|relationship|support)\b",
                lowered,
            ) else 0.0
        if answer_type == "identity":
            return 1.0 if re.search(
                r"\b(?:identity|transgender|gay|lesbian|bisexual|queer|nonbinary|woman|man|parent|student|artist|writer)\b",
                lowered,
            ) else 0.0
        if answer_type == "person":
            return 1.0 if re.search(r"\b[A-Z][a-z]{2,}\b", str(content or "")) else 0.0
        return 0.0

    @classmethod
    def _query_evidence_profile(cls, query: str, options: list[str]) -> dict[str, Any]:
        query = str(query or "").strip()
        option_text = " ".join(str(option) for option in options or [] if str(option).strip())
        answer_type = "generic"
        lowered = query.lower()
        if re.search(r"\bhow long|since when|for how many\b", lowered):
            answer_type = "duration"
        elif re.search(r"\bwhen|what date|which date|what day|which day|what month|what year\b", lowered):
            answer_type = "time"
        elif re.search(r"\bwho\b", lowered):
            answer_type = "person"
        elif re.search(r"\bwhere\b", lowered):
            answer_type = "location"
        elif re.search(r"\bwhy\b", lowered):
            answer_type = "reason"
        elif re.search(r"\bidentity\b", lowered):
            answer_type = "identity"
        elif re.search(r"\brelationship|relation status|married|single\b", lowered):
            answer_type = "relation"
        elif re.search(r"^\s*(?:would|could|should|is|are|was|were|do|does|did|can)\b", lowered):
            answer_type = "boolean"

        target_speakers: list[str] = []
        excluded = {
            "what", "when", "where", "who", "why", "how", "which", "would", "could", "should",
            "the", "a", "an", "options", "answer", "first", "second", "third", "fourth",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december",
        }
        for match in re.finditer(r"\b([A-Z][a-z][A-Za-z'’-]{1,})(?:'s)?\b", query):
            value = match.group(1).strip()
            if value.lower() not in excluded and value not in target_speakers:
                target_speakers.append(value)

        speaker_tokens = set()
        for speaker in target_speakers:
            speaker_tokens |= cls._rerank_tokens(speaker)
        topic_tokens = cls._rerank_tokens(f"{query} {option_text}") - speaker_tokens
        return {
            "answer_type": answer_type,
            "target_speakers": target_speakers,
            "target_speaker_tokens": speaker_tokens,
            "topic_tokens": topic_tokens,
        }

    @classmethod
    def _speaker_rerank_factor(cls, query_profile: dict[str, Any], metadata: dict[str, Any]) -> float:
        target_speakers = [
            cls._normalize_person_name(value)
            for value in query_profile.get("target_speakers") or []
            if cls._normalize_person_name(value)
        ]
        if not target_speakers:
            return 1.0
        speaker = cls._normalize_person_name(metadata.get("speaker", ""))
        if not speaker:
            return 0.96
        if speaker in target_speakers:
            return 1.24
        return 0.90

    @classmethod
    def _semantic_evidence_factor(
        cls,
        query_profile: dict[str, Any],
        passage_tokens: set[str],
        content_lower: str,
    ) -> float:
        topic_tokens = set(query_profile.get("topic_tokens") or [])
        if not topic_tokens:
            return 1.0
        overlap = topic_tokens & passage_tokens
        coverage = len(overlap) / max(len(topic_tokens), 1)
        factor = 1.0
        if coverage >= 0.65:
            factor += 0.34
        elif coverage >= 0.40:
            factor += 0.24
        elif overlap:
            factor += min(0.18, 0.05 * len(overlap) + 0.10 * coverage)
        else:
            factor -= 0.08

        answer_type = str(query_profile.get("answer_type") or "")
        if answer_type == "identity" and re.search(
            r"\b(identity|transgender|gay|lesbian|bisexual|queer|nonbinary|single parent|mother|father|student|teacher|artist|writer|engineer|doctor|nurse)\b",
            content_lower,
        ):
            factor += 0.10
        return max(0.86, min(1.38, factor))

    @classmethod
    def _answer_type_rerank_factor(
        cls,
        query_profile: dict[str, Any],
        content: str,
        metadata: dict[str, Any],
    ) -> float:
        answer_type = str(query_profile.get("answer_type") or "generic")
        lowered = content.lower()
        factor = 1.0
        if answer_type == "time":
            has_text_time = cls._has_time_expression(content)
            has_relative = cls._has_relative_time_expression(content)
            has_structured_time = bool(
                metadata.get("session_time")
                or metadata.get("memory_timestamp")
                or metadata.get("memory_time_iso")
            )
            if has_text_time:
                factor += 0.18
            if has_relative and has_structured_time:
                factor += 0.08
            if not has_text_time:
                factor -= 0.10
        elif answer_type == "duration":
            if re.search(r"\b(?:for|since)\b|\b\d+\s+(?:years?|months?|weeks?|days?|hours?)\b", lowered):
                factor += 0.14
            else:
                factor -= 0.04
        elif answer_type == "location":
            if re.search(r"\b(?:from|to|in|at|near|moved|move|home country|city|town|country|beach|mountains?|forest)\b", lowered):
                factor += 0.08
        elif answer_type == "reason":
            if re.search(r"\b(?:because|since|so that|due to|thanks to|as a result|made me|helped me)\b", lowered):
                factor += 0.08
        elif answer_type == "relation":
            if re.search(r"\b(?:single|married|wife|husband|partner|friend|family|colleague|parent|breakup|relationship)\b", lowered):
                factor += 0.10
        elif answer_type == "identity":
            if re.search(r"\b(?:i am|i'm|as a|identity|transgender|gay|lesbian|bisexual|queer|woman|man|parent)\b", lowered):
                factor += 0.10
        return max(0.90, min(1.22, factor))

    def _complete_neighbor_evidence(
        self,
        query_profile: dict[str, Any],
        ranked: list[tuple[str, float]],
        limit: int,
    ) -> list[tuple[str, float]]:
        if not self.enable_evidence_completion or not ranked:
            return ranked
        window = max(0, int(self.completion_window))
        anchor_top_n = max(0, int(self.completion_anchor_top_n))
        max_neighbors = max(0, int(self.completion_max_neighbors))
        related_top_n = max(0, int(self.completion_related_top_n))
        max_related = max(0, int(self.completion_max_related))
        min_related_fit = max(0.0, float(self.completion_min_related_fit))
        if anchor_top_n == 0 and related_top_n == 0:
            return ranked

        metadata_by_passage = {passage: self._passage_metadata(passage) for passage, _ in ranked}
        original_order = {passage: index for index, (passage, _) in enumerate(ranked)}
        boosted_scores = {passage: float(score) for passage, score in ranked}
        anchors = ranked[: min(len(ranked), max(anchor_top_n, min(limit, anchor_top_n)))] if anchor_top_n else []

        if window > 0 and max_neighbors > 0:
            for anchor_passage, anchor_score in anchors:
                anchor_meta = metadata_by_passage.get(anchor_passage) or {}
                anchor_session = str(anchor_meta.get("session_index") or "")
                anchor_message = self._metadata_int(anchor_meta.get("message_index"))
                if not anchor_session or anchor_message is None:
                    continue

                neighbors: list[tuple[int, float, str, float]] = []
                for passage, score in ranked:
                    if passage == anchor_passage:
                        continue
                    meta = metadata_by_passage.get(passage) or {}
                    if str(meta.get("session_index") or "") != anchor_session:
                        continue
                    message_index = self._metadata_int(meta.get("message_index"))
                    if message_index is None:
                        continue
                    distance = abs(message_index - anchor_message)
                    if distance < 1 or distance > window:
                        continue
                    if not self._neighbor_speaker_allowed(query_profile, anchor_meta, meta):
                        continue
                    fit = self._neighbor_fit_score(query_profile, meta)
                    neighbors.append((distance, -fit, passage, score))

                neighbors.sort(key=lambda item: (item[0], item[1], original_order.get(item[2], 0)))
                for distance, negative_fit, passage, _score in neighbors[:max_neighbors]:
                    fit = -negative_fit
                    candidate_factor = 0.96 - 0.06 * distance + 0.04 * fit
                    candidate_factor = max(0.72, min(0.96, candidate_factor))
                    boosted_scores[passage] = max(boosted_scores[passage], float(anchor_score) * candidate_factor)

        if related_top_n and max_related:
            related_anchors = sorted(
                ranked,
                key=lambda item: (boosted_scores.get(item[0], item[1]), -original_order.get(item[0], 0)),
                reverse=True,
            )[: min(len(ranked), max(related_top_n, min(limit, related_top_n)))]
            for anchor_passage, anchor_score in related_anchors:
                anchor_meta = metadata_by_passage.get(anchor_passage) or {}
                related_candidates: list[tuple[float, str, float]] = []
                for passage, score in ranked:
                    if passage == anchor_passage:
                        continue
                    meta = metadata_by_passage.get(passage) or {}
                    if not self._neighbor_speaker_allowed(query_profile, anchor_meta, meta):
                        continue
                    fit = self._related_evidence_fit_score(query_profile, anchor_meta, meta)
                    if fit < min_related_fit:
                        continue
                    same_session = bool(
                        anchor_meta.get("session_index")
                        and anchor_meta.get("session_index") == meta.get("session_index")
                    )
                    candidate_factor = 0.56 + 0.24 * fit + (0.06 if same_session else 0.0)
                    candidate_factor = max(0.50, min(0.86, candidate_factor))
                    candidate_score = max(float(score), float(anchor_score) * candidate_factor)
                    related_candidates.append((-fit, passage, candidate_score))

                related_candidates.sort(key=lambda item: (item[0], original_order.get(item[1], 0)))
                for _negative_fit, passage, candidate_score in related_candidates[:max_related]:
                    boosted_scores[passage] = max(boosted_scores[passage], candidate_score)

        ordered = sorted(
            ranked,
            key=lambda item: (boosted_scores.get(item[0], item[1]), -original_order.get(item[0], 0)),
            reverse=True,
        )
        return [(passage, boosted_scores.get(passage, score)) for passage, score in ordered]

    @classmethod
    def _neighbor_speaker_allowed(
        cls,
        query_profile: dict[str, Any],
        anchor_meta: dict[str, Any],
        candidate_meta: dict[str, Any],
    ) -> bool:
        target_speakers = [
            cls._normalize_person_name(value)
            for value in query_profile.get("target_speakers") or []
            if cls._normalize_person_name(value)
        ]
        candidate_speaker = cls._normalize_person_name(candidate_meta.get("speaker", ""))
        if target_speakers:
            return not candidate_speaker or candidate_speaker in target_speakers
        anchor_speaker = cls._normalize_person_name(anchor_meta.get("speaker", ""))
        return not anchor_speaker or not candidate_speaker or anchor_speaker == candidate_speaker

    @classmethod
    def _neighbor_fit_score(cls, query_profile: dict[str, Any], metadata: dict[str, Any]) -> float:
        topic_tokens = set(query_profile.get("topic_tokens") or [])
        passage_tokens = cls._rerank_tokens(str(metadata.get("evidence_text") or ""))
        if not topic_tokens:
            return 0.0
        return len(topic_tokens & passage_tokens) / max(len(topic_tokens), 1)

    @classmethod
    def _related_evidence_fit_score(
        cls,
        query_profile: dict[str, Any],
        anchor_meta: dict[str, Any],
        candidate_meta: dict[str, Any],
    ) -> float:
        candidate_text = str(candidate_meta.get("evidence_text") or "")
        anchor_text = str(anchor_meta.get("evidence_text") or "")
        topic_tokens = set(query_profile.get("topic_tokens") or [])
        candidate_tokens = cls._rerank_tokens(candidate_text)
        anchor_tokens = cls._rerank_tokens(anchor_text)
        topic_fit = 0.0
        if topic_tokens:
            topic_fit = len(topic_tokens & candidate_tokens) / max(len(topic_tokens), 1)
        type_fit = cls._answer_type_signal(query_profile, candidate_text, candidate_meta)
        shared_fit = 0.0
        if anchor_tokens and candidate_tokens:
            shared_fit = len(anchor_tokens & candidate_tokens) / max(min(len(anchor_tokens), len(candidate_tokens), 8), 1)
        return max(0.0, min(1.0, 0.52 * topic_fit + 0.35 * type_fit + 0.13 * min(shared_fit, 1.0)))

    @staticmethod
    def _metadata_int(value: Any) -> int | None:
        try:
            return int(str(value).strip())
        except Exception:
            return None

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
            "he", "she", "him", "her", "his", "hers", "they", "them", "their", "theirs", "it", "its",
            "answer", "answers", "best", "match", "matches", "memory", "option", "first", "second",
            "third", "fourth", "following", "about", "tell", "ask", "asked", "said", "say",
            "would", "could", "should", "likely", "maybe", "probably", "possibly", "please",
            "best", "matches", "match", "memory",
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
        aliases = {
            "edu": "education",
            "counseling": "counsel",
            "counselor": "counsel",
            "counsellor": "counsel",
            "counselling": "counsel",
            "researched": "research",
            "researching": "research",
            "transitioning": "transition",
            "transitioned": "transition",
        }
        if token in aliases:
            return aliases[token]
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

    @staticmethod
    def _has_relative_time_expression(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:today|yesterday|tomorrow|tonight|last week|next week|last month|next month|last year|next year|recently|currently|now|before|previously|earlier|later|this month|this week)\b",
                str(text or ""),
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _has_time_expression(cls, text: str) -> bool:
        return bool(cls._time_values(text))

    @staticmethod
    def _is_temporal_query(query: str) -> bool:
        lowered = query.lower()
        return any(token in lowered for token in ("when", "date", "day", "month", "year", "time", "before", "after", "earlier", "later"))

    @classmethod
    def _passage_metadata(cls, passage: str) -> dict[str, Any]:
        visible = cls._strip_internal_prefix(passage)
        metadata = {
            "raw_id": MemoryLayer.parse_raw_id(passage) or "",
            "dia_id": cls._bracket_value(visible, "dia_id"),
            "session_index": cls._bracket_value(visible, "session_index")
            or cls._bracket_value(visible, "source_session_index"),
            "message_index": cls._bracket_value(visible, "message_index")
            or cls._bracket_value(visible, "source_message_index"),
            "session_time": cls._bracket_value(visible, "session_time"),
            "memory_timestamp": cls._bracket_value(visible, "memory_timestamp"),
            "memory_time_iso": cls._bracket_value(visible, "memory_time_iso"),
            "order_index": cls._bracket_value(visible, "order_index"),
            "session_id": cls._bracket_value(visible, "session_id"),
            "request_id": cls._bracket_value(visible, "request_id"),
            "source_session_index": cls._bracket_value(visible, "source_session_index"),
            "source_message_index": cls._bracket_value(visible, "source_message_index"),
            "speaker": "",
            "evidence_text": "",
        }
        speaker = ""
        evidence_lines: list[str] = []
        for line in visible.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"^([A-Z][A-Za-z0-9 ._'-]{0,60}):\s*(.*)$", stripped)
            if match:
                if not speaker:
                    speaker = match.group(1).strip()
                stripped = match.group(2).strip()
            stripped = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", stripped).strip()
            if not stripped or re.fullmatch(r"(?:\[[^\]]+\]\s*)+", stripped):
                continue
            evidence_lines.append(stripped)
        metadata["speaker"] = speaker
        metadata["evidence_text"] = "\n".join(evidence_lines).strip() or visible
        return metadata

    @staticmethod
    def _bracket_value(text: str, name: str) -> str:
        match = re.search(rf"\[{re.escape(name)}=([^\]]+)\]", str(text or ""))
        return match.group(1).strip() if match else ""

    @staticmethod
    def _normalize_person_name(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"'s$", "", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

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
