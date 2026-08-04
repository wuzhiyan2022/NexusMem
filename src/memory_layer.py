from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


class MemoryLayer:
    """A lightweight, generic memory consolidation layer for the add/search API.

    The layer keeps raw evidence as the main index text and stores long-term
    memory state in sidecar files. This lets LinearRAG retrieve faithful source
    passages while the API can still reason about stale, updated, or conflicting
    memories during reranking.
    """

    CURRENT_HINTS = {
        "now",
        "currently",
        "current",
        "latest",
        "recent",
        "recently",
        "today",
        "these days",
        "at the moment",
        "现在",
        "目前",
        "当前",
        "最近",
        "最新",
        "如今",
    }
    HISTORY_HINTS = {
        "before",
        "previously",
        "earlier",
        "used to",
        "old",
        "history",
        "historical",
        "past",
        "once",
        "formerly",
        "以前",
        "之前",
        "曾经",
        "原来",
        "过去",
        "历史",
    }
    UPDATE_HINTS = {
        "now",
        "currently",
        "recently",
        "no longer",
        "not anymore",
        "anymore",
        "used to",
        "instead",
        "changed",
        "moved",
        "switched",
        "updated",
        "became",
        "turns out",
        "again",
        "现在",
        "目前",
        "最近",
        "不再",
        "再也不",
        "曾经",
        "以前",
        "改为",
        "换成",
        "搬到",
        "变成",
        "重新",
        "又",
    }
    UNCERTAINTY_HINTS = {
        "maybe",
        "might",
        "possibly",
        "probably",
        "not sure",
        "i think",
        "perhaps",
        "可能",
        "也许",
        "大概",
        "不确定",
    }
    SINGLE_VALUE_PREDICATES = {
        "current_location",
        "workplace",
        "education",
        "favorite",
    }
    EVENT_VERB_DEPS = {
        "ROOT",
        "conj",
        "advcl",
        "xcomp",
        "ccomp",
        "relcl",
    }
    EVENT_STOP_LEMMAS = {
        "be",
        "do",
        "have",
        "like",
        "love",
        "prefer",
        "hate",
        "enjoy",
        "know",
        "think",
        "believe",
        "want",
        "need",
        "mean",
        "seem",
        "sound",
        "look",
        "feel",
        "understand",
        "remember",
        "forget",
    }

    def __init__(self, user_dir: Path, *, max_chunk_chars: int) -> None:
        self.user_dir = user_dir
        self.max_chunk_chars = max_chunk_chars
        self.raw_path = self.user_dir / "raw_memories.jsonl"
        self.consolidated_path = self.user_dir / "consolidated_memories.json"
        self.relations_path = self.user_dir / "memory_relations.jsonl"
        self.event_nodes_path = self.user_dir / "event_nodes.json"
        self.time_nodes_path = self.user_dir / "time_nodes.json"
        self.graph_edges_path = self.user_dir / "memory_graph_edges.jsonl"

    def build_passages(
        self,
        request: Any,
        split_content: Callable[[str], list[str]],
        start_seq: int,
        split_sentences: Callable[[str], list[str]] | None = None,
        extract_events: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> list[str]:
        self.user_dir.mkdir(parents=True, exist_ok=True)
        consolidated = self._load_consolidated()
        raw_records: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []

        for message_index, message in enumerate(request.messages):
            role = (message.role or "").strip() or "unknown"
            content = (message.content or "").strip()
            if not content:
                continue
            timestamp = "" if message.timestamp is None else str(message.timestamp)
            for part_index, part in enumerate(split_content(content)):
                raw_id = self._raw_id(
                    request.user_id,
                    request.session_id,
                    request.request_id,
                    message_index,
                    part_index,
                    part,
                )
                raw_candidates = self._extract_candidates(
                    part,
                    role=role,
                    timestamp=timestamp,
                    raw_id=raw_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_id=request.request_id,
                    message_index=message_index,
                    part_index=part_index,
                    split_sentences=split_sentences,
                    extract_events=extract_events,
                )
                raw_records.append(
                    {
                        "raw_id": raw_id,
                        "user_id": request.user_id,
                        "session_id": request.session_id,
                        "request_id": request.request_id,
                        "message_index": message_index,
                        "part_index": part_index,
                        "role": role,
                        "timestamp": timestamp,
                        "content": part,
                        "candidate_ids": [candidate["candidate_id"] for candidate in raw_candidates],
                        "candidate_count": len(raw_candidates),
                        "created_at": int(time.time()),
                    }
                )
                candidates.extend(raw_candidates)

        event_nodes, time_nodes, graph_edges = self._build_event_time_graph(candidates)
        consolidated, relation_records = self._merge_candidates(consolidated, candidates)
        self._append_jsonl(self.raw_path, raw_records)
        self._save_consolidated(consolidated)
        self._append_jsonl(self.relations_path, relation_records)
        self._merge_json_list(self.event_nodes_path, event_nodes, "event_id")
        self._merge_json_list(self.time_nodes_path, time_nodes, "time_id")
        self._append_jsonl(self.graph_edges_path, graph_edges)

        raw_to_memories = self._raw_to_memory_summary(consolidated)
        passages: list[str] = []
        seq = start_seq
        for raw in raw_records:
            summary = raw_to_memories.get(raw["raw_id"], {})
            header_parts = [
                "[view=raw_evidence]",
                f"[raw_id={raw['raw_id']}]",
                f"[session_id={raw['session_id']}]",
                f"[request_id={raw['request_id']}]",
                f"[message_index={raw['message_index']}]",
                f"[part_index={raw['part_index']}]",
                f"[role={raw['role']}]",
            ]
            if raw["timestamp"]:
                header_parts.append(f"[timestamp={raw['timestamp']}]")
            if summary.get("memory_ids"):
                header_parts.append(f"[memory_ids={','.join(summary['memory_ids'])}]")
            if summary.get("memory_types"):
                header_parts.append(f"[memory_types={','.join(summary['memory_types'])}]")
            if summary.get("event_times"):
                header_parts.append(f"[event_times={','.join(summary['event_times'])}]")
            if summary.get("event_ids"):
                header_parts.append(f"[event_ids={','.join(summary['event_ids'])}]")
            if summary.get("time_ids"):
                header_parts.append(f"[time_ids={','.join(summary['time_ids'])}]")

            header = " ".join(header_parts)
            passages.append(f"{seq}:{header}\n{raw['role']}: {raw['content']}")
            seq += 1
        return passages

    def load_raw_memory_status(self) -> dict[str, dict[str, Any]]:
        consolidated = self._load_consolidated()
        return self._raw_to_memory_summary(consolidated)

    @staticmethod
    def parse_raw_id(passage: str) -> str | None:
        match = re.search(r"\[raw_id=([^\]]+)\]", passage)
        return match.group(1) if match else None

    @classmethod
    def query_intent(cls, query: str) -> str:
        normalized = query.lower()
        has_current = any(hint in normalized for hint in cls.CURRENT_HINTS)
        has_history = any(hint in normalized for hint in cls.HISTORY_HINTS)
        if has_current and not has_history:
            return "current"
        if has_history and not has_current:
            return "history"
        if has_current and has_history:
            return "mixed"
        return "neutral"

    @staticmethod
    def adjust_score(score: float, status_info: dict[str, Any] | None, intent: str) -> float:
        if not status_info:
            return score

        statuses = set(status_info.get("statuses") or [])
        if not statuses:
            return score

        factor = 1.0
        if "active" in statuses:
            factor *= 1.08
        if "conflicting" in statuses:
            factor *= 0.95
        if "expired" in statuses:
            factor *= 0.9

        if intent == "current":
            if statuses == {"expired"}:
                factor *= 0.72
            elif "active" in statuses:
                factor *= 1.08
        elif intent == "history":
            if "expired" in statuses:
                factor *= 1.12
            if statuses == {"active"}:
                factor *= 0.98

        return score * factor

    def _extract_candidates(
        self,
        text: str,
        *,
        role: str,
        timestamp: str,
        raw_id: str,
        user_id: str,
        session_id: str,
        request_id: str,
        message_index: int,
        part_index: int,
        split_sentences: Callable[[str], list[str]] | None = None,
        extract_events: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        sentence_splitter = split_sentences or self._split_sentences
        for sentence_index, sentence in enumerate(sentence_splitter(text)):
            sentence_candidates: list[dict[str, Any]] = []
            sentence_candidates.extend(self._preference_candidates(sentence))
            sentence_candidates.extend(self._fact_candidates(sentence))
            sentence_candidates.extend(self._relation_candidates(sentence))
            sentence_candidates.extend(self._rule_candidates(sentence))
            if extract_events is not None:
                sentence_candidates.extend(extract_events(sentence))
            else:
                sentence_candidates.extend(self._event_candidates(sentence))

            for candidate in sentence_candidates:
                time_expressions = self._extract_time_expressions(sentence, timestamp)
                for time_expression in candidate.get("time_expressions") or []:
                    self._append_unique(time_expressions, time_expression)
                candidate.update(
                    {
                        "raw_id": raw_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "request_id": request_id,
                        "message_index": message_index,
                        "part_index": part_index,
                        "sentence_index": sentence_index,
                        "source_text": sentence,
                        "role": role,
                        "timestamp": timestamp,
                        "time_expressions": time_expressions,
                        "update_signal": self._has_update_signal(sentence),
                    }
                )
                candidate["object_norm"] = self._normalize_value(candidate.get("object", ""))
                candidate["subject_norm"] = self._normalize_value(candidate.get("subject", ""))
                candidate["predicate_norm"] = self._normalize_value(candidate.get("predicate", ""))
                candidate["time_node_ids"] = [
                    self._time_node_id(value, timestamp) for value in candidate.get("time_expressions", [])
                ]
                if candidate.get("memory_type") == "event":
                    candidate["event_id"] = self._event_id(candidate)
                candidate["confidence"] = self._confidence(candidate)
                candidate["candidate_id"] = self._candidate_id(candidate)
                candidates.append(candidate)
        return candidates

    def _preference_candidates(self, sentence: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        patterns = [
            (r"\b(?:i|we)\s+(?:do not|don't|dont|no longer|never)\s+(?:like|love|prefer|enjoy)\s+(.+)$", "negative"),
            (r"\b(?:i|we)\s+(?:dislike|hate|cannot stand|can't stand)\s+(.+)$", "negative"),
            (r"\b(?:i|we)\s+(?:really\s+)?(?:like|love|prefer|enjoy)\s+(.+)$", "positive"),
            (r"\bmy favorite\s+[\w\s-]{0,40}?\s+is\s+(.+)$", "positive"),
        ]
        for pattern, polarity in patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                candidates.append(
                    self._candidate(
                        memory_type="preference",
                        subject="user",
                        predicate="preference",
                        object_text=self._clean_object(match.group(1)),
                        polarity=polarity,
                        pattern="preference",
                    )
                )

        zh_patterns = [
            (r"我(?:现在|目前|最近)?(?:不再|再也不|不)(?:喜欢|爱|偏好)(.+)$", "negative"),
            (r"我(?:现在|目前|最近)?(?:喜欢|爱|偏好)(.+)$", "positive"),
            (r"我最喜欢(?:的)?[\w\u4e00-\u9fff\s]{0,20}是(.+)$", "positive"),
        ]
        for pattern, polarity in zh_patterns:
            match = re.search(pattern, sentence)
            if match:
                candidates.append(
                    self._candidate(
                        memory_type="preference",
                        subject="user",
                        predicate="preference",
                        object_text=self._clean_object(match.group(1)),
                        polarity=polarity,
                        pattern="preference_zh",
                    )
                )
        return [candidate for candidate in candidates if candidate["object"]]

    def _fact_candidates(self, sentence: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        fact_patterns = [
            (r"\b(?:i|we)\s+(?:currently\s+)?(?:live|reside)\s+in\s+(.+)$", "current_location"),
            (r"\b(?:i|we)\s+moved\s+to\s+(.+)$", "current_location"),
            (r"\b(?:i|we)\s+work\s+(?:at|for|in)\s+(.+)$", "workplace"),
            (r"\bmy\s+(?:company|employer|workplace)\s+is\s+(.+)$", "workplace"),
            (r"\b(?:i|we)\s+(?:study|studied)\s+at\s+(.+)$", "education"),
        ]
        for pattern, predicate in fact_patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                candidates.append(
                    self._candidate(
                        memory_type="fact",
                        subject="user",
                        predicate=predicate,
                        object_text=self._clean_object(match.group(1)),
                        polarity="neutral",
                        pattern=predicate,
                    )
                )

        zh_patterns = [
            (r"我(?:现在|目前)?(?:住在|居住在|搬到)(.+)$", "current_location"),
            (r"我(?:现在|目前)?在(.+?)(?:工作|上班)$", "workplace"),
            (r"我(?:现在|目前)?在(.+?)(?:学习|读书|上学)$", "education"),
        ]
        for pattern, predicate in zh_patterns:
            match = re.search(pattern, sentence)
            if match:
                candidates.append(
                    self._candidate(
                        memory_type="fact",
                        subject="user",
                        predicate=predicate,
                        object_text=self._clean_object(match.group(1)),
                        polarity="neutral",
                        pattern=f"{predicate}_zh",
                    )
                )
        return [candidate for candidate in candidates if candidate["object"]]

    def _relation_candidates(self, sentence: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        relation_words = (
            "friend|colleague|coworker|wife|husband|partner|sister|brother|mother|father|"
            "manager|boss|teammate|classmate|roommate|assistant|doctor|teacher"
        )
        patterns = [
            (rf"\b([A-Z][A-Za-z .'-]{{1,80}})\s+is\s+my\s+({relation_words})\b", 1, 2),
            (rf"\bmy\s+({relation_words})\s+is\s+([A-Z][A-Za-z .'-]{{1,80}})\b", 2, 1),
        ]
        for pattern, object_group, relation_group in patterns:
            match = re.search(pattern, sentence)
            if match:
                relation = self._clean_object(match.group(relation_group)).lower()
                candidates.append(
                    self._candidate(
                        memory_type="relation",
                        subject="user",
                        predicate=f"relation:{relation}",
                        object_text=self._clean_object(match.group(object_group)),
                        polarity="neutral",
                        pattern="relation",
                    )
                )

        zh_match = re.search(r"([\w\u4e00-\u9fff .'-]{1,40})是我的(朋友|同事|妻子|丈夫|伴侣|姐妹|兄弟|妈妈|爸爸|经理|老板|队友|同学|室友|老师)", sentence)
        if zh_match:
            candidates.append(
                self._candidate(
                    memory_type="relation",
                    subject="user",
                    predicate=f"relation:{zh_match.group(2)}",
                    object_text=self._clean_object(zh_match.group(1)),
                    polarity="neutral",
                    pattern="relation_zh",
                )
            )
        return [candidate for candidate in candidates if candidate["object"]]

    def _rule_candidates(self, sentence: str) -> list[dict[str, Any]]:
        lowered = sentence.lower()
        if not any(token in lowered for token in ("always", "never", "must", "should", "need to", "if ", "when ")):
            if not any(token in sentence for token in ("总是", "从不", "必须", "应该", "如果", "当")):
                return []
        return [
            self._candidate(
                memory_type="rule",
                subject="user",
                predicate="rule",
                object_text=self._clean_object(sentence),
                polarity="neutral",
                pattern="rule",
            )
        ]

    def _event_candidates(self, sentence: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        event_verbs = (
            "moved|went|visited|traveled|travelled|met|joined|left|started|finished|completed|"
            "attended|booked|bought|called|emailed|planned|scheduled|decided|changed|switched|"
            "graduated|married|divorced|adopted|lost|found|watched|read|saw|ate|had|made|created|launched"
        )
        patterns = [
            rf"\b(?P<subject>i|we)\s+(?P<trigger>{event_verbs})\b(?P<object>[^.!?。！？]*)",
            rf"\b(?P<subject>[A-Z][A-Za-z .'-]{{1,80}})\s+(?P<trigger>{event_verbs})\b(?P<object>[^.!?。！？]*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if not match:
                continue
            trigger = match.group("trigger")
            object_text = self._clean_event_object(match.group("object"))
            if not object_text and trigger.lower() not in {"graduated", "married", "divorced"}:
                continue
            subject = self._event_subject(match.group("subject"))
            candidates.append(
                self._candidate(
                    memory_type="event",
                    subject=subject,
                    predicate=self._canonical_event_trigger(trigger),
                    object_text=object_text or trigger,
                    polarity="neutral",
                    pattern="event",
                )
                | {
                    "event_trigger": trigger.lower(),
                    "event_participants": [value for value in [subject, object_text] if value],
                }
            )

        return [candidate for candidate in candidates if candidate["object"]]

    @classmethod
    def event_candidates_from_spacy(cls, doc: Any) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        text = getattr(doc, "text", "")
        spacy_times = [
            ent.text
            for ent in getattr(doc, "ents", [])
            if getattr(ent, "label_", "") in {"DATE", "TIME"}
        ]
        named_entities = [
            ent.text
            for ent in getattr(doc, "ents", [])
            if getattr(ent, "label_", "") not in {"DATE", "TIME", "ORDINAL", "CARDINAL"}
        ]

        for token in doc:
            if not cls._is_event_verb_token(token):
                continue
            subject = cls._find_event_subject(token)
            object_text = cls._find_event_object(token)
            if not object_text:
                object_text = cls._nearest_entity_after(token, named_entities)
            object_text = cls._clean_event_object(object_text)
            if not subject or not object_text:
                continue

            participants = [value for value in [subject, object_text, *named_entities] if value]
            unique_participants: list[str] = []
            for participant in participants:
                if participant not in unique_participants:
                    unique_participants.append(participant)

            candidates.append(
                cls._candidate(
                    memory_type="event",
                    subject=subject,
                    predicate=cls._canonical_event_trigger(getattr(token, "lemma_", "") or getattr(token, "text", "")),
                    object_text=object_text,
                    polarity="neutral",
                    pattern="event_spacy_verb",
                )
                | {
                    "event_trigger": getattr(token, "text", "").lower(),
                    "event_participants": unique_participants,
                    "time_expressions": spacy_times,
                    "source_text": text,
                }
            )
        return candidates

    @classmethod
    def _is_event_verb_token(cls, token: Any) -> bool:
        lemma = str(getattr(token, "lemma_", "") or getattr(token, "text", "")).lower()
        dep = str(getattr(token, "dep_", ""))
        pos = str(getattr(token, "pos_", ""))
        if pos != "VERB":
            return False
        if dep not in cls.EVENT_VERB_DEPS:
            return False
        if lemma in cls.EVENT_STOP_LEMMAS:
            return False
        if getattr(token, "is_stop", False) and lemma not in {"move", "visit", "meet", "join", "leave", "start", "finish", "buy", "book", "change"}:
            return False
        return True

    @classmethod
    def _find_event_subject(cls, token: Any) -> str:
        for node in [token, *list(getattr(token, "ancestors", []))]:
            for child in getattr(node, "children", []):
                if getattr(child, "dep_", "") in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
                    return cls._event_subject(cls._subtree_text(child))
        return ""

    @classmethod
    def _find_event_object(cls, token: Any) -> str:
        direct_deps = {"dobj", "obj", "attr", "dative", "oprd"}
        prep_deps = {"prep", "agent"}
        for child in getattr(token, "children", []):
            if getattr(child, "dep_", "") in direct_deps:
                return cls._subtree_text(child)
        for child in getattr(token, "children", []):
            if getattr(child, "dep_", "") in prep_deps:
                for grandchild in getattr(child, "children", []):
                    if getattr(grandchild, "dep_", "") in {"pobj", "obj"}:
                        return cls._subtree_text(grandchild)
        for child in getattr(token, "children", []):
            if getattr(child, "dep_", "") in {"xcomp", "ccomp"}:
                nested = cls._find_event_object(child)
                if nested:
                    return nested
        return ""

    @staticmethod
    def _subtree_text(token: Any) -> str:
        subtree = sorted(list(getattr(token, "subtree", [token])), key=lambda item: getattr(item, "i", 0))
        return " ".join(getattr(item, "text", "") for item in subtree).strip()

    @staticmethod
    def _nearest_entity_after(token: Any, entities: list[str]) -> str:
        token_index = getattr(token, "i", 0)
        doc_text = getattr(getattr(token, "doc", None), "text", "")
        best_entity = ""
        best_pos = len(doc_text) + 1
        for entity in entities:
            pos = doc_text.find(entity)
            if pos < 0:
                continue
            entity_token_count = doc_text[:pos].count(" ")
            if entity_token_count >= token_index and pos < best_pos:
                best_entity = entity
                best_pos = pos
        return best_entity

    @staticmethod
    def _candidate(
        *,
        memory_type: str,
        subject: str,
        predicate: str,
        object_text: str,
        polarity: str,
        pattern: str,
    ) -> dict[str, Any]:
        return {
            "memory_type": memory_type,
            "subject": subject,
            "predicate": predicate,
            "object": object_text,
            "polarity": polarity,
            "pattern": pattern,
        }

    def _merge_candidates(
        self,
        consolidated: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        relations: list[dict[str, Any]] = []
        memories = [self._normalize_memory(memory) for memory in consolidated]

        for candidate in candidates:
            if candidate.get("confidence", 0.0) < 0.45:
                continue

            exact = [
                memory
                for memory in memories
                if self._same_slot(memory, candidate) and memory.get("object_norm") == candidate.get("object_norm")
            ]
            related = [memory for memory in memories if self._same_slot(memory, candidate)]

            if exact:
                newest = self._newest_memory(exact)
                if newest.get("polarity") == candidate.get("polarity"):
                    if newest.get("status") == "expired" and candidate.get("update_signal"):
                        new_memory = self._memory_from_candidate(candidate)
                        new_memory["status"] = "active"
                        new_memory["supersedes"] = [newest["memory_id"]]
                        memories.append(new_memory)
                        relations.append(self._relation("reactivate", new_memory["memory_id"], newest["memory_id"], candidate))
                        continue
                    self._append_evidence(newest, candidate)
                    relations.append(self._relation("duplicate", newest["memory_id"], newest["memory_id"], candidate))
                    continue

                relation = "update" if self._should_update(candidate, newest) else "conflict"
                new_memory = self._memory_from_candidate(candidate)
                if relation == "update":
                    for memory in exact:
                        self._expire(memory, new_memory["memory_id"], candidate)
                    new_memory["supersedes"] = [memory["memory_id"] for memory in exact]
                    new_memory["status"] = "active"
                else:
                    for memory in exact:
                        memory["status"] = "conflicting"
                        self._link(memory, "conflicts_with", new_memory["memory_id"])
                    new_memory["status"] = "conflicting"
                    new_memory["conflicts_with"] = [memory["memory_id"] for memory in exact]
                memories.append(new_memory)
                relations.append(self._relation(relation, new_memory["memory_id"], newest["memory_id"], candidate))
                continue

            if related and self._is_single_value(candidate):
                newest = self._newest_memory(related)
                new_memory = self._memory_from_candidate(candidate)
                if self._should_update(candidate, newest):
                    for memory in related:
                        self._expire(memory, new_memory["memory_id"], candidate)
                    new_memory["status"] = "active"
                    new_memory["supersedes"] = [memory["memory_id"] for memory in related]
                    relations.append(self._relation("update", new_memory["memory_id"], newest["memory_id"], candidate))
                else:
                    for memory in related:
                        memory["status"] = "conflicting"
                        self._link(memory, "conflicts_with", new_memory["memory_id"])
                    new_memory["status"] = "conflicting"
                    new_memory["conflicts_with"] = [memory["memory_id"] for memory in related]
                    relations.append(self._relation("conflict", new_memory["memory_id"], newest["memory_id"], candidate))
                memories.append(new_memory)
                continue

            new_memory = self._memory_from_candidate(candidate)
            memories.append(new_memory)
            relations.append(self._relation("new", new_memory["memory_id"], "", candidate))

        memories.sort(key=lambda item: (self._time_value(item.get("updated_at")), item.get("memory_id", "")))
        return memories, relations

    def _memory_from_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        now = int(time.time())
        timestamp = candidate.get("timestamp") or ""
        memory_id = "mem-" + hashlib.sha256(candidate["candidate_id"].encode("utf-8")).hexdigest()[:16]
        memory = {
            "memory_id": memory_id,
            "memory_type": candidate.get("memory_type"),
            "subject": candidate.get("subject"),
            "predicate": candidate.get("predicate"),
            "object": candidate.get("object"),
            "subject_norm": candidate.get("subject_norm"),
            "predicate_norm": candidate.get("predicate_norm"),
            "object_norm": candidate.get("object_norm"),
            "polarity": candidate.get("polarity"),
            "status": "active",
            "confidence": candidate.get("confidence"),
            "source_text": candidate.get("source_text"),
            "evidence_raw_ids": [candidate.get("raw_id")],
            "candidate_ids": [candidate.get("candidate_id")],
            "time_expressions": candidate.get("time_expressions") or [],
            "time_node_ids": candidate.get("time_node_ids") or [],
            "valid_from": timestamp,
            "valid_to": "",
            "created_at": now,
            "updated_at": now,
            "supersedes": [],
            "superseded_by": "",
            "conflicts_with": [],
        }
        if candidate.get("memory_type") == "event":
            memory["event_id"] = candidate.get("event_id")
            memory["event_trigger"] = candidate.get("event_trigger")
            memory["event_participants"] = candidate.get("event_participants") or []
        return memory

    @staticmethod
    def _normalize_memory(memory: dict[str, Any]) -> dict[str, Any]:
        memory = dict(memory)
        memory.setdefault("status", "active")
        memory.setdefault("evidence_raw_ids", [])
        memory.setdefault("candidate_ids", [])
        memory.setdefault("time_expressions", [])
        memory.setdefault("time_node_ids", [])
        memory.setdefault("supersedes", [])
        memory.setdefault("superseded_by", "")
        memory.setdefault("conflicts_with", [])
        memory.setdefault("valid_to", "")
        for field in ("subject", "predicate", "object"):
            norm_field = f"{field}_norm"
            if not memory.get(norm_field):
                memory[norm_field] = MemoryLayer._normalize_value(memory.get(field, ""))
        return memory

    def _raw_to_memory_summary(self, memories: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        raw_to_memories: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "memory_ids": [],
                "memory_types": [],
                "statuses": [],
                "event_times": [],
                "event_ids": [],
                "time_ids": [],
            }
        )
        for memory in memories:
            for raw_id in memory.get("evidence_raw_ids", []):
                summary = raw_to_memories[raw_id]
                self._append_unique(summary["memory_ids"], memory.get("memory_id"))
                self._append_unique(summary["memory_types"], memory.get("memory_type"))
                self._append_unique(summary["statuses"], memory.get("status"))
                self._append_unique(summary["event_ids"], memory.get("event_id"))
                for time_id in memory.get("time_node_ids") or []:
                    self._append_unique(summary["time_ids"], str(time_id))
                for event_time in memory.get("time_expressions") or []:
                    self._append_unique(summary["event_times"], str(event_time))
        return dict(raw_to_memories)

    @staticmethod
    def _same_slot(memory: dict[str, Any], candidate: dict[str, Any]) -> bool:
        return (
            memory.get("memory_type") == candidate.get("memory_type")
            and memory.get("subject_norm") == candidate.get("subject_norm")
            and memory.get("predicate_norm") == candidate.get("predicate_norm")
        )

    @classmethod
    def _is_single_value(cls, candidate: dict[str, Any]) -> bool:
        return candidate.get("predicate") in cls.SINGLE_VALUE_PREDICATES

    def _should_update(self, candidate: dict[str, Any], old_memory: dict[str, Any]) -> bool:
        if candidate.get("update_signal"):
            return True
        if self._is_single_value(candidate):
            return self._time_value(candidate.get("timestamp")) >= self._time_value(old_memory.get("valid_from"))
        return False

    def _expire(self, memory: dict[str, Any], superseded_by: str, candidate: dict[str, Any]) -> None:
        memory["status"] = "expired"
        memory["superseded_by"] = superseded_by
        memory["valid_to"] = candidate.get("timestamp") or ""
        memory["updated_at"] = int(time.time())

    @staticmethod
    def _link(memory: dict[str, Any], field: str, value: str) -> None:
        values = memory.setdefault(field, [])
        if value and value not in values:
            values.append(value)

    def _append_evidence(self, memory: dict[str, Any], candidate: dict[str, Any]) -> None:
        self._append_unique(memory.setdefault("evidence_raw_ids", []), candidate.get("raw_id"))
        self._append_unique(memory.setdefault("candidate_ids", []), candidate.get("candidate_id"))
        for event_time in candidate.get("time_expressions") or []:
            self._append_unique(memory.setdefault("time_expressions", []), event_time)
        for time_id in candidate.get("time_node_ids") or []:
            self._append_unique(memory.setdefault("time_node_ids", []), time_id)
        memory["confidence"] = max(float(memory.get("confidence") or 0.0), float(candidate.get("confidence") or 0.0))
        memory["updated_at"] = int(time.time())

    @staticmethod
    def _relation(relation_type: str, source_id: str, target_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "relation_type": relation_type,
            "source_memory_id": source_id,
            "target_memory_id": target_id,
            "raw_id": candidate.get("raw_id"),
            "candidate_id": candidate.get("candidate_id"),
            "timestamp": candidate.get("timestamp") or "",
            "created_at": int(time.time()),
        }

    @staticmethod
    def _newest_memory(memories: list[dict[str, Any]]) -> dict[str, Any]:
        return max(memories, key=lambda item: MemoryLayer._time_value(item.get("valid_from")) or MemoryLayer._time_value(item.get("updated_at")))

    @staticmethod
    def _time_value(value: Any) -> float:
        if value is None or value == "":
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value)
        match = re.search(r"\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else 0.0

    def _confidence(self, candidate: dict[str, Any]) -> float:
        score = 0.45
        if candidate.get("pattern"):
            score += 0.25
        if candidate.get("object"):
            score += 0.1
        if candidate.get("time_expressions") or candidate.get("timestamp"):
            score += 0.05
        if candidate.get("update_signal"):
            score += 0.05
        source = str(candidate.get("source_text", "")).lower()
        if any(hint in source for hint in self.UNCERTAINTY_HINTS):
            score -= 0.18
        return max(0.0, min(0.95, score))

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        return [piece.strip() for piece in pieces if piece.strip()]

    @staticmethod
    def _event_subject(value: str) -> str:
        normalized = str(value or "").strip()
        return "user" if normalized.lower() in {"i", "we"} or normalized in {"我", "我们"} else normalized

    @staticmethod
    def _canonical_event_trigger(trigger: str) -> str:
        value = str(trigger or "").strip().lower()
        groups = {
            "move": {"moved"},
            "go": {"went"},
            "visit": {"visited"},
            "travel": {"traveled", "travelled"},
            "meet": {"met"},
            "join": {"joined"},
            "leave": {"left"},
            "start": {"started"},
            "finish": {"finished", "completed"},
            "book": {"booked"},
            "buy": {"bought"},
            "call": {"called"},
            "email": {"emailed"},
            "plan": {"planned", "scheduled"},
            "decide": {"decided"},
            "change": {"changed", "switched"},
            "graduate": {"graduated"},
            "marry": {"married"},
            "divorce": {"divorced"},
            "create": {"made", "created"},
            "launch": {"launched"},
        }
        for canonical, aliases in groups.items():
            if value in aliases:
                return canonical
        return value or "event"

    @classmethod
    def _has_update_signal(cls, text: str) -> bool:
        lowered = text.lower()
        return any(hint in lowered for hint in cls.UPDATE_HINTS)

    @staticmethod
    def _extract_time_expressions(text: str, timestamp: str) -> list[str]:
        values: list[str] = []
        patterns = [
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            r"\b(?:19|20)\d{2}\b",
            r"\b\d{1,2}:\d{2}(?:\s?[ap]\.?m\.?)?\b",
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(?:spring|summer|fall|autumn|winter)\s+(?:19|20)\d{2}\b",
            r"\bq[1-4]\s+(?:19|20)\d{2}\b",
            r"\b(?:today|yesterday|tomorrow|tonight|last week|next week|last month|next month|last year|next year|recently|currently|now|before|previously|earlier this year|later this year)\b",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?\b",
            r"\d{4}年\d{1,2}月\d{1,2}日",
            r"\d{4}年\d{1,2}月",
            r"\d{1,2}月\d{1,2}日",
            r"\d{1,2}点\d{0,2}分?",
            r"(?:今天|昨天|明天|今晚|上周|下周|上个月|下个月|去年|明年|最近|目前|现在|以前|曾经|今年早些时候|今年晚些时候)",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                value = str(match).strip()
                if value and value not in values:
                    values.append(value)
        return values

    def _build_event_time_graph(
        self,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        event_nodes: list[dict[str, Any]] = []
        time_nodes_by_id: dict[str, dict[str, Any]] = {}
        graph_edges: list[dict[str, Any]] = []
        now = int(time.time())

        for candidate in candidates:
            if candidate.get("memory_type") != "event":
                continue
            event_id = candidate.get("event_id") or self._event_id(candidate)
            candidate["event_id"] = event_id
            event_node = {
                "event_id": event_id,
                "event_trigger": candidate.get("event_trigger") or candidate.get("predicate"),
                "canonical_trigger": candidate.get("predicate"),
                "subject": candidate.get("subject"),
                "object": candidate.get("object"),
                "participants": candidate.get("event_participants") or [],
                "source_text": candidate.get("source_text"),
                "raw_id": candidate.get("raw_id"),
                "request_id": candidate.get("request_id"),
                "session_id": candidate.get("session_id"),
                "timestamp": candidate.get("timestamp") or "",
                "created_at": now,
            }
            event_nodes.append(event_node)
            graph_edges.append(self._graph_edge(event_id, "evidence", str(candidate.get("raw_id") or ""), candidate))

            for time_expression in candidate.get("time_expressions") or []:
                time_node = self._time_node(time_expression, candidate.get("timestamp") or "")
                time_nodes_by_id[time_node["time_id"]] = time_node
                graph_edges.append(self._graph_edge(event_id, "occurs_at", time_node["time_id"], candidate))

            for participant in candidate.get("event_participants") or []:
                participant_id = "participant-" + hashlib.sha256(
                    str(participant).strip().lower().encode("utf-8")
                ).hexdigest()[:16]
                graph_edges.append(self._graph_edge(event_id, "has_participant", participant_id, candidate))

        return event_nodes, list(time_nodes_by_id.values()), graph_edges

    @staticmethod
    def _graph_edge(source_id: str, edge_type: str, target_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_id": source_id,
            "edge_type": edge_type,
            "target_id": target_id,
            "raw_id": candidate.get("raw_id"),
            "candidate_id": candidate.get("candidate_id"),
            "created_at": int(time.time()),
        }

    def _time_node(self, expression: str, message_timestamp: str) -> dict[str, Any]:
        normalized = self._normalize_time_expression(expression, message_timestamp)
        return {
            "time_id": self._time_node_id(expression, message_timestamp),
            "expression": expression,
            "normalized": normalized,
            "kind": self._time_kind(expression),
            "source": "text",
            "created_at": int(time.time()),
        }

    @staticmethod
    def _time_kind(expression: str) -> str:
        lowered = expression.lower()
        if re.fullmatch(r"(?:19|20)\d{2}", lowered):
            return "year"
        if re.search(r"\d{1,2}:\d{2}|\d{1,2}点", lowered):
            return "clock_time"
        if any(token in lowered for token in ("today", "yesterday", "tomorrow", "last", "next", "recently", "currently", "now")):
            return "relative"
        if any(token in expression for token in ("今天", "昨天", "明天", "去年", "明年", "最近", "目前", "现在", "以前", "曾经")):
            return "relative"
        return "date"

    @staticmethod
    def _normalize_time_expression(expression: str, message_timestamp: str) -> str:
        value = str(expression or "").strip()
        return re.sub(r"\s+", " ", value.lower())

    @staticmethod
    def _time_node_id(expression: str, message_timestamp: str) -> str:
        payload = f"{expression}|{message_timestamp}"
        return "time-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _event_id(candidate: dict[str, Any]) -> str:
        payload = "|".join(
            [
                str(candidate.get("raw_id", "")),
                str(candidate.get("sentence_index", "")),
                str(candidate.get("subject_norm", "")),
                str(candidate.get("predicate_norm", "")),
                str(candidate.get("object_norm", "")),
                str(candidate.get("timestamp", "")),
            ]
        )
        return "event-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _clean_object(value: str) -> str:
        text = value.strip(" \t\r\n\"'`.,!?。！？")
        text = re.split(r"\b(?:but|because|although|while|and then)\b|[,;。；，]", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = text.strip(" \t\r\n\"'`.,!?。！？")
        text = re.sub(r"\b(?:now|currently|today|these days|anymore|instead|again)$", "", text, flags=re.IGNORECASE).strip()
        text = text.strip(" \t\r\n\"'`.,!?。！？")
        text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.IGNORECASE)
        return text.strip(" \t\r\n\"'`.,!?。！？")

    @staticmethod
    def _clean_event_object(value: str) -> str:
        text = MemoryLayer._clean_object(value)
        for expression in MemoryLayer._extract_time_expressions(text, ""):
            if expression.startswith("message_timestamp:"):
                continue
            text = re.sub(re.escape(expression), "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^(?:to|in|at|for|from|with|on|into|onto|about)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+(?:to|in|at|for|from|with|on|into|onto|about)$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:到|去|在|从|和|与|把)", "", text)
        text = re.sub(r"(?:今天|昨天|明天|今晚|上周|下周|上个月|下个月|去年|明年|最近|目前|现在|以前|曾经)$", "", text)
        return text.strip(" \t\r\n\"'`.,!?。！？")

    @staticmethod
    def _normalize_value(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"^(?:a|an|the)\s+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" \t\r\n\"'`.,!?。！？")

    @staticmethod
    def _candidate_id(candidate: dict[str, Any]) -> str:
        payload = "|".join(
            [
                str(candidate.get("raw_id", "")),
                str(candidate.get("sentence_index", "")),
                str(candidate.get("memory_type", "")),
                str(candidate.get("subject_norm", "")),
                str(candidate.get("predicate_norm", "")),
                str(candidate.get("object_norm", "")),
                str(candidate.get("polarity", "")),
            ]
        )
        return "cand-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _raw_id(
        user_id: str,
        session_id: str,
        request_id: str,
        message_index: int,
        part_index: int,
        content: str,
    ) -> str:
        payload = "|".join([user_id, session_id, request_id, str(message_index), str(part_index), content])
        return "raw-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _load_consolidated(self) -> list[dict[str, Any]]:
        if not self.consolidated_path.exists():
            return []
        with self.consolidated_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []

    def _save_consolidated(self, memories: list[dict[str, Any]]) -> None:
        tmp_path = self.consolidated_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(memories, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(self.consolidated_path)

    @staticmethod
    def _merge_json_list(path: Path, records: list[dict[str, Any]], id_field: str) -> None:
        if not records:
            return
        existing: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get(id_field):
                        existing[str(item[id_field])] = item
        for record in records:
            record_id = record.get(id_field)
            if not record_id:
                continue
            existing[str(record_id)] = {**existing.get(str(record_id), {}), **record}

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(list(existing.values()), handle, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    @staticmethod
    def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _append_unique(values: list[Any], value: Any) -> None:
        if value and value not in values:
            values.append(value)
