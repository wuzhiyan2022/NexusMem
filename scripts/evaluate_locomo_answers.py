#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def select_dataset(
    conversations: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    max_conversations: int,
    max_questions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_conversations > 0:
        allowed = {str(item.get("sample_id") or "") for item in conversations[:max_conversations]}
        conversations = [item for item in conversations if str(item.get("sample_id") or "") in allowed]
        questions = [item for item in questions if str(item.get("sample_id") or "") in allowed]
    if max_questions > 0:
        questions = questions[:max_questions]
        allowed = {str(item.get("sample_id") or "") for item in questions}
        conversations = [item for item in conversations if str(item.get("sample_id") or "") in allowed]
    return conversations, questions


def message_content(message: dict[str, Any], *, include_multimodal: bool, omit_image_urls: bool) -> str:
    dia_id = str(message.get("dia_id") or "").strip()
    speaker = str(message.get("speaker") or message.get("role") or "").strip()
    text = str(message.get("text") or "").strip()
    session_index = message.get("session_index", "")
    message_index = message.get("message_index", "")
    session_date_time = str(message.get("session_date_time") or "").strip()

    header = f"[dia_id={dia_id}] [session_index={session_index}] [message_index={message_index}]"
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
        if images and not omit_image_urls:
            lines.append("[images] " + ", ".join(images))
    return "\n".join(line for line in lines if line)


def iter_messages(
    conversation: dict[str, Any],
    *,
    include_multimodal: bool,
    omit_image_urls: bool,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for session in conversation.get("sessions") or []:
        date_time = str(session.get("date_time") or "")
        for message in session.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message_content(
                message,
                include_multimodal=include_multimodal,
                omit_image_urls=omit_image_urls,
            )
            if not content.strip():
                continue
            messages.append(
                {
                    "role": str(message.get("role") or "user"),
                    "speaker": str(message.get("speaker") or "").strip() or None,
                    "timestamp": str(message.get("session_date_time") or date_time or ""),
                    "content": content,
                }
            )
    return messages


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def safe_remove_storage_dir(storage_dir: Path) -> None:
    resolved = storage_dir.resolve()
    repo_root = REPO_ROOT.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), repo_root}
    if resolved in forbidden:
        raise ValueError(f"Refusing to remove unsafe storage directory: {resolved}")
    if repo_root not in resolved.parents:
        raise ValueError(f"Refusing to remove storage outside repository: {resolved}")
    shutil.rmtree(resolved)


class AnswerGenerator:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float,
        max_tokens: int,
    ) -> None:
        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.max_tokens = max_tokens
        if not self.model:
            raise ValueError("Answer model is empty. Set --answer-model or LINEARRAG_ANSWER_LLM_MODEL.")
        if not self.base_url:
            raise ValueError("Answer base URL is empty. Set --answer-base-url or OPENAI_BASE_URL.")

    def answer(self, *, question: str, contexts: list[dict[str, Any]]) -> str:
        prompt = self._build_prompt(question=question, contexts=contexts)
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer long-term memory questions using only the retrieved evidence. "
                        "Return a concise answer, not an explanation. If the answer is not supported, "
                        "return 'I don't know'. Preserve the time wording from the evidence when the "
                        "question asks about time."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._chat_completions_url(),
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"] or "").strip()

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    @staticmethod
    def _build_prompt(*, question: str, contexts: list[dict[str, Any]]) -> str:
        evidence_lines = []
        for index, item in enumerate(contexts, start=1):
            score = item.get("score")
            score_text = f" score={score:.4f}" if isinstance(score, (int, float)) else ""
            content = str(item.get("content") or "").strip()
            evidence_lines.append(f"[{index}{score_text}]\n{content}")
        evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "(no retrieved evidence)"
        return (
            f"Question:\n{question.strip()}\n\n"
            f"Retrieved evidence:\n{evidence_block}\n\n"
            "Return only the final answer."
        )


def build_contexts(data: list[dict[str, Any]], *, context_k: int, max_chars: int) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    used_chars = 0
    for item in data[: max(0, context_k)]:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining].rstrip()
        contexts.append(
            {
                "id": item.get("id"),
                "score": item.get("score"),
                "content": content,
            }
        )
        used_chars += len(content)
    return contexts


def generate_predictions(
    *,
    service: Any,
    conversations: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    args: argparse.Namespace,
    generator: AnswerGenerator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    AddRequest, SearchRequest = load_memory_api_models()
    questions_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        questions_by_sample[str(question.get("sample_id") or "")].append(question)

    predictions: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    include_multimodal = not args.text_only
    started_at = time.monotonic()

    for conversation_index, conversation in enumerate(conversations, start=1):
        sample_id = str(conversation.get("sample_id") or f"conversation-{conversation_index}")
        user_id = f"locomo-refined:{sample_id}"
        messages = iter_messages(
            conversation,
            include_multimodal=include_multimodal,
            omit_image_urls=args.omit_image_urls,
        )
        for batch_index, batch in enumerate(batched(messages, max(1, args.add_batch_size))):
            add_request = AddRequest(
                request_id=f"locomo-answer-eval:{sample_id}:chunk-{batch_index:04d}",
                user_id=user_id,
                session_id=f"locomo-answer-eval:{sample_id}",
                messages=batch,
            )
            service.add(add_request)

        sample_questions = questions_by_sample.get(sample_id, [])
        for question_index, question in enumerate(sample_questions, start=1):
            qa_id = str(question.get("qa_id") or "")
            query = str(question.get("question") or "")
            search_request = SearchRequest(
                user_id=user_id,
                query=query,
                top_k=args.top_k,
                options=None,
            )
            response = service.search(search_request)
            data = response.get("data", []) if isinstance(response, dict) else []
            contexts = build_contexts(
                data,
                context_k=args.answer_context_k,
                max_chars=args.max_context_chars,
            )
            try:
                predicted_answer = generator.answer(question=query, contexts=contexts)
                error = None
            except Exception as exc:
                predicted_answer = ""
                error = str(exc)

            predictions.append({"qa_id": qa_id, "predicted_answer": predicted_answer})
            details.append(
                {
                    "qa_id": qa_id,
                    "sample_id": sample_id,
                    "question": query,
                    "predicted_answer": predicted_answer,
                    "generation_error": error,
                    "retrieved_count": len(data),
                    "contexts": contexts,
                }
            )
            if args.progress:
                elapsed = time.monotonic() - started_at
                print(
                    f"[{len(predictions)}/{len(questions)}] {qa_id} "
                    f"conversation={conversation_index}/{len(conversations)} "
                    f"question={question_index}/{len(sample_questions)} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
    return predictions, details


def load_memory_api_models() -> tuple[Any, Any]:
    from api_server import AddRequest, SearchRequest

    return AddRequest, SearchRequest


def create_memory_service() -> Any:
    from api_server import LinearRAGMemoryService

    return LinearRAGMemoryService()


def score_with_locomo(
    *,
    locomo_dir: Path,
    questions_path: Path,
    predictions_path: Path,
    output_path: Path,
    summary_path: Path,
    markdown_summary_path: Path,
    metrics: Sequence[str],
    llm_judge: str,
    concurrency: int,
    evaluator_model: str | None,
    evaluator_base_url: str | None,
    evaluator_api_key: str | None,
) -> dict[str, Any]:
    locomo_src = locomo_dir / "src"
    if not locomo_src.exists():
        raise FileNotFoundError(f"Missing LoCoMo_refined src directory: {locomo_src}")
    sys.path.insert(0, str(locomo_src))
    from evaluate import evaluate_public_predictions
    from summarize import render_public_score_markdown, summarize_public_scores

    evaluated = asyncio.run(
        evaluate_public_predictions(
            questions_path=questions_path,
            predictions_path=predictions_path,
            output_path=output_path,
            metrics=metrics,
            llm_judge=llm_judge,
            concurrency=concurrency,
            evaluator_model=evaluator_model,
            evaluator_base_url=evaluator_base_url,
            evaluator_api_key=evaluator_api_key,
            strict=True,
            progress=True,
        )
    )
    summary = summarize_public_scores(evaluated)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_summary_path.write_text(render_public_score_markdown(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate answers with the current LinearRAG memory service and score them with LoCoMo-Refined metrics."
    )
    parser.add_argument("--locomo-dir", required=True, help="Path to LoCoMo_refined repository root.")
    parser.add_argument("--conversations-path", default="", help="Override conversations.jsonl path.")
    parser.add_argument("--questions-path", default="", help="Override questions.jsonl path.")
    parser.add_argument("--storage-dir", default="import_api_locomo_answer_eval")
    parser.add_argument("--output-dir", default="outputs/locomo_answer_eval")
    parser.add_argument("--predictions-path", default="")
    parser.add_argument("--scored-path", default="")
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--markdown-summary-path", default="")
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--add-batch-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--answer-context-k", type=int, default=12)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--omit-image-urls", action="store_true")
    parser.add_argument("--reuse-storage", action="store_true")
    parser.add_argument("--disable-llm-memory-extraction", action="store_true")
    parser.add_argument("--disable-llm-query-intent", action="store_true")
    parser.add_argument("--score-only", action="store_true", help="Score an existing predictions file.")
    parser.add_argument("--skip-score", action="store_true", help="Only write predictions/details.")
    parser.add_argument("--metrics", nargs="+", default=["llm", "f1", "bleu"])
    parser.add_argument("--llm-judge", default="refined")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--answer-model", default=os.getenv("LINEARRAG_ANSWER_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--answer-base-url",
        default=(
            os.getenv("LINEARRAG_ANSWER_LLM_BASE_URL")
            or os.getenv("LINEARRAG_QUERY_LLM_BASE_URL")
            or os.getenv("LINEARRAG_EXTRACT_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or ""
        ),
    )
    parser.add_argument(
        "--answer-api-key",
        default=(
            os.getenv("LINEARRAG_ANSWER_LLM_API_KEY")
            or os.getenv("LINEARRAG_QUERY_LLM_API_KEY")
            or os.getenv("LINEARRAG_EXTRACT_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ),
    )
    parser.add_argument("--answer-timeout", type=float, default=60.0)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--evaluator-model", default=None)
    parser.add_argument("--evaluator-base-url", default=None)
    parser.add_argument("--evaluator-api-key", default=None)
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    locomo_dir = Path(args.locomo_dir).expanduser().resolve()
    conversations_path, questions_path = resolve_public_paths(args)
    output_dir = resolve_repo_path(args.output_dir)
    storage_dir = resolve_repo_path(args.storage_dir)
    predictions_path = resolve_repo_path(args.predictions_path or output_dir / "predictions.jsonl")
    scored_path = resolve_repo_path(args.scored_path or output_dir / "predictions_scored.jsonl")
    summary_path = resolve_repo_path(args.summary_path or output_dir / "predictions_scored_summary.json")
    markdown_summary_path = resolve_repo_path(
        args.markdown_summary_path or output_dir / "predictions_scored_summary.md"
    )
    eval_questions_path = output_dir / "questions_eval.jsonl"
    details_path = output_dir / "prediction_details.jsonl"

    conversations = load_jsonl(conversations_path)
    questions = load_jsonl(questions_path)
    conversations, questions = select_dataset(
        conversations,
        questions,
        max_conversations=args.max_conversations,
        max_questions=args.max_questions,
    )
    write_jsonl(eval_questions_path, questions)

    if not args.score_only:
        if storage_dir.exists() and not args.reuse_storage:
            safe_remove_storage_dir(storage_dir)
        os.environ["LINEARRAG_STORAGE_DIR"] = str(storage_dir)
        if args.disable_llm_memory_extraction:
            os.environ["LINEARRAG_ENABLE_LLM_MEMORY_EXTRACTION"] = "0"
        if args.disable_llm_query_intent:
            os.environ["LINEARRAG_ENABLE_LLM_QUERY_INTENT"] = "0"

        service = create_memory_service()
        generator = AnswerGenerator(
            model=args.answer_model,
            base_url=args.answer_base_url,
            api_key=args.answer_api_key,
            timeout=args.answer_timeout,
            max_tokens=args.answer_max_tokens,
        )
        predictions, details = generate_predictions(
            service=service,
            conversations=conversations,
            questions=questions,
            args=args,
            generator=generator,
        )
        write_jsonl(predictions_path, predictions)
        write_jsonl(details_path, details)

    result: dict[str, Any] = {
        "questions_path": str(eval_questions_path),
        "predictions_path": str(predictions_path),
        "details_path": str(details_path),
        "scored_path": str(scored_path),
        "summary_path": str(summary_path),
        "markdown_summary_path": str(markdown_summary_path),
        "record_count": len(questions),
    }

    if not args.skip_score:
        summary = score_with_locomo(
            locomo_dir=locomo_dir,
            questions_path=eval_questions_path,
            predictions_path=predictions_path,
            output_path=scored_path,
            summary_path=summary_path,
            markdown_summary_path=markdown_summary_path,
            metrics=args.metrics,
            llm_judge=args.llm_judge,
            concurrency=args.concurrency,
            evaluator_model=args.evaluator_model,
            evaluator_base_url=args.evaluator_base_url,
            evaluator_api_key=args.evaluator_api_key,
        )
        result["primary_metric"] = summary.get("primary_metric")
        result["overall"] = summary.get("overall")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
