#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_server import AddRequest, LinearRAGMemoryService, SearchRequest


EVIDENCE_RE = re.compile(r"\[dia_id=([^\]\s]+)\]")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


def resolve_public_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    locomo_dir = Path(args.locomo_dir).expanduser().resolve() if args.locomo_dir else None
    if locomo_dir:
        public_dir = locomo_dir / "data" / "public"
        conversations_path = Path(args.conversations_path or public_dir / "conversations.jsonl")
        questions_path = Path(args.questions_path or public_dir / "questions.jsonl")
    else:
        if not args.conversations_path or not args.questions_path:
            raise ValueError("Provide --locomo-dir or both --conversations-path and --questions-path.")
        conversations_path = Path(args.conversations_path)
        questions_path = Path(args.questions_path)
    if not conversations_path.exists():
        raise FileNotFoundError(f"Missing conversations file: {conversations_path}")
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing questions file: {questions_path}")
    return conversations_path.resolve(), questions_path.resolve()


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def message_content(message: dict[str, Any], *, include_multimodal: bool) -> str:
    dia_id = str(message.get("dia_id") or "").strip()
    speaker = str(message.get("speaker") or message.get("role") or "").strip()
    text = str(message.get("text") or "").strip()
    session_index = message.get("session_index", "")
    message_index = message.get("message_index", "")
    session_date_time = str(message.get("session_date_time") or "").strip()

    header = (
        f"[dia_id={dia_id}] [session_index={session_index}] "
        f"[message_index={message_index}]"
    )
    if session_date_time:
        header += f" [session_time={session_date_time}]"

    lines = [header]
    lines.append(f"{speaker}: {text}" if speaker else text)

    if include_multimodal:
        caption = str(message.get("blip_caption") or "").strip()
        query = str(message.get("query") or "").strip()
        images = [str(item).strip() for item in message.get("images") or [] if str(item).strip()]
        if caption:
            lines.append(f"[caption] {caption}")
        if query:
            lines.append(f"[query] {query}")
        if images and not args_global.omit_image_urls:
            lines.append("[images] " + ", ".join(images))
    return "\n".join(line for line in lines if line)


def iter_messages(conversation: dict[str, Any], *, include_multimodal: bool) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for session in conversation.get("sessions") or []:
        session_id = str(session.get("session_index") or "")
        date_time = str(session.get("date_time") or "")
        for message in session.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message_content(message, include_multimodal=include_multimodal)
            if not content.strip():
                continue
            messages.append(
                {
                    "role": str(message.get("role") or "user"),
                    "speaker": str(message.get("speaker") or "").strip() or None,
                    "timestamp": str(message.get("session_date_time") or date_time or ""),
                    "content": content,
                    "_session_id": session_id,
                }
            )
    return messages


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def extract_retrieved_evidence(search_data: list[dict[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for item in search_data:
        content = str(item.get("content") or "")
        found = EVIDENCE_RE.findall(content)
        if found:
            evidence.extend(found)
    return evidence


def score_question(gold_evidence: list[str], retrieved_evidence: list[str], ks: list[int]) -> dict[str, Any]:
    gold = [item for item in gold_evidence if item]
    gold_set = set(gold)
    row: dict[str, Any] = {
        "gold_count": len(gold_set),
        "retrieved_evidence": retrieved_evidence,
    }
    if not gold_set:
        for k in ks:
            row[f"hit@{k}"] = None
            row[f"recall@{k}"] = None
            row[f"precision@{k}"] = None
            row[f"full_recall@{k}"] = None
        row["mrr"] = None
        return row

    first_rank: int | None = None
    for rank, evidence_id in enumerate(retrieved_evidence, start=1):
        if evidence_id in gold_set:
            first_rank = rank
            break
    row["mrr"] = 0.0 if first_rank is None else 1.0 / first_rank

    for k in ks:
        top = retrieved_evidence[:k]
        top_set = set(top)
        matched = gold_set & top_set
        row[f"hit@{k}"] = 1.0 if matched else 0.0
        row[f"recall@{k}"] = len(matched) / len(gold_set)
        row[f"precision@{k}"] = len(matched) / k if k else 0.0
        row[f"full_recall@{k}"] = 1.0 if gold_set.issubset(top_set) else 0.0
    return row


def summarize(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    answerable = [row for row in rows if row.get("gold_count", 0) > 0]
    summary: dict[str, Any] = {
        "question_count": len(rows),
        "questions_with_gold_evidence": len(answerable),
        "questions_without_gold_evidence": len(rows) - len(answerable),
    }
    for k in ks:
        for metric in ["hit", "recall", "precision", "full_recall"]:
            key = f"{metric}@{k}"
            values = [float(row[key]) for row in answerable if row.get(key) is not None]
            summary[key] = mean(values) if values else None
    mrr_values = [float(row["mrr"]) for row in answerable if row.get("mrr") is not None]
    summary["mrr"] = mean(mrr_values) if mrr_values else None
    return summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "retrieval_details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "retrieval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_remove_storage_dir(storage_dir: Path) -> None:
    resolved = storage_dir.resolve()
    repo_root = REPO_ROOT.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), repo_root}
    if resolved in forbidden:
        raise ValueError(f"Refusing to remove unsafe storage directory: {resolved}")
    if repo_root not in resolved.parents:
        raise ValueError(f"Refusing to remove storage outside repository: {resolved}")
    shutil.rmtree(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LinearRAG retrieval against LoCoMo gold evidence.")
    parser.add_argument("--locomo-dir", default="", help="Path to LoCoMo_refined repository root.")
    parser.add_argument("--conversations-path", default="", help="Override path to conversations.jsonl.")
    parser.add_argument("--questions-path", default="", help="Override path to questions.jsonl.")
    parser.add_argument("--storage-dir", default="import_api_locomo_eval", help="Temporary LinearRAG API storage dir.")
    parser.add_argument("--output-dir", default="outputs/locomo_retrieval_eval", help="Where to write metrics.")
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--add-batch-size", type=int, default=20, help="Messages per simulated Add request.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ks", default="1,5,10,20,50,100", help="Comma-separated k values for metrics.")
    parser.add_argument("--text-only", action="store_true", help="Ignore caption/query/image fields.")
    parser.add_argument("--omit-image-urls", action="store_true", help="Do not include image URLs in chunks.")
    parser.add_argument("--reuse-storage", action="store_true", help="Do not delete storage-dir before indexing.")
    return parser


def main() -> None:
    global args_global
    parser = build_parser()
    args = parser.parse_args()
    args_global = args

    conversations_path, questions_path = resolve_public_paths(args)
    output_dir = resolve_repo_path(args.output_dir)
    storage_dir = resolve_repo_path(args.storage_dir)
    ks = sorted({int(item.strip()) for item in args.ks.split(",") if item.strip()})
    if args.top_k not in ks:
        ks.append(args.top_k)
        ks = sorted(set(ks))

    if storage_dir.exists() and not args.reuse_storage:
        safe_remove_storage_dir(storage_dir)
    os.environ["LINEARRAG_STORAGE_DIR"] = str(storage_dir)

    conversations = load_jsonl(conversations_path)
    questions = load_jsonl(questions_path)
    if args.max_conversations > 0:
        allowed = {str(item.get("sample_id") or "") for item in conversations[: args.max_conversations]}
        conversations = [item for item in conversations if str(item.get("sample_id") or "") in allowed]
        questions = [item for item in questions if str(item.get("sample_id") or "") in allowed]
    if args.max_questions > 0:
        questions = questions[: args.max_questions]
        allowed = {str(item.get("sample_id") or "") for item in questions}
        conversations = [item for item in conversations if str(item.get("sample_id") or "") in allowed]

    questions_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        questions_by_sample[str(question.get("sample_id") or "")].append(question)

    service = LinearRAGMemoryService()
    rows: list[dict[str, Any]] = []
    include_multimodal = not args.text_only

    for conversation_index, conversation in enumerate(conversations, start=1):
        sample_id = str(conversation.get("sample_id") or f"conversation-{conversation_index}")
        user_id = f"locomo-refined:{sample_id}"
        messages = iter_messages(conversation, include_multimodal=include_multimodal)
        for batch_index, batch in enumerate(batched(messages, max(1, args.add_batch_size))):
            add_request = AddRequest(
                request_id=f"locomo-eval:{sample_id}:chunk-{batch_index:04d}",
                user_id=user_id,
                session_id=f"locomo-eval:{sample_id}",
                messages=[{k: v for k, v in item.items() if not k.startswith("_")} for item in batch],
            )
            service.add(add_request)

        for question in questions_by_sample.get(sample_id, []):
            qa_id = str(question.get("qa_id") or "")
            search_request = SearchRequest(
                user_id=user_id,
                query=str(question.get("question") or ""),
                top_k=args.top_k,
                options=None,
            )
            response = service.search(search_request)
            data = response.get("data", [])
            retrieved_evidence = extract_retrieved_evidence(data)
            row = {
                "qa_id": qa_id,
                "sample_id": sample_id,
                "question": str(question.get("question") or ""),
                "gold_evidence": [str(item) for item in question.get("evidence") or []],
                "top_k": args.top_k,
                "retrieved_count": len(data),
            }
            row.update(score_question(row["gold_evidence"], retrieved_evidence, ks))
            rows.append(row)

    summary = summarize(rows, ks)
    summary.update(
        {
            "conversations_path": str(conversations_path),
            "questions_path": str(questions_path),
            "storage_dir": str(storage_dir),
            "output_dir": str(output_dir),
            "top_k": args.top_k,
            "ks": ks,
            "include_multimodal_text": include_multimodal,
        }
    )
    write_outputs(rows, summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


args_global: argparse.Namespace


if __name__ == "__main__":
    main()
