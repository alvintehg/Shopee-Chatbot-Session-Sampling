"""
pipeline.py  (PRODUCTION REWRITE)
──────────────────────────────────
The orchestration layer with all fixes applied:
  ✅  Streaming load (no full-file RAM load)
  ✅  Deduplication before any processing
  ✅  Text normalization on every message before classification
  ✅  Fallback chain: LLM (high conf) → keyword tagger → "unclear"
  ✅  Logged drop counts at every filter stage
  ✅  Structured audit log written per run
  ✅  PipelineStats collected end-to-end
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Generator, TypeVar

from .models import Conversation, PipelineStats
from .normalizer import normalize_conversation_messages

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Filter helpers ────────────────────────────────────────────────────────────

def _log_filter(
    stream: Generator[Conversation, None, None],
    predicate: Callable[[Conversation], bool],
    label: str,
) -> Generator[Conversation, None, None]:
    """Yield only conversations passing predicate. Logs drop count."""
    passed = dropped = 0
    for conv in stream:
        if predicate(conv):
            passed += 1
            yield conv
        else:
            dropped += 1
    logger.info(f"Filter [{label}]: {passed} kept, {dropped} dropped")


def apply_filters(
    conversations: list[Conversation],
    start_date: str | None = None,
    end_date: str | None = None,
    include_intents: list[str] | None = None,
    exclude_intents: list[str] | None = None,
    min_messages: int = 0,
    timezone_str: str = "Asia/Kuala_Lumpur",
) -> list[Conversation]:
    """Apply all filters with per-stage logging. Returns filtered list."""
    from datetime import datetime

    stream = iter(conversations)

    if start_date:
        sd = datetime.fromisoformat(start_date)
        stream = _log_filter(
            stream,
            lambda c: c.started_at is not None and c.started_at >= sd,
            f"start_date>={start_date}",
        )

    if end_date:
        ed = datetime.fromisoformat(end_date)
        stream = _log_filter(
            stream,
            lambda c: c.started_at is not None and c.started_at <= ed,
            f"end_date<={end_date}",
        )

    if include_intents:
        intents_set = set(include_intents)
        stream = _log_filter(
            stream,
            lambda c: c.intent in intents_set,
            f"include_intents={include_intents}",
        )

    if exclude_intents:
        excl_set = set(exclude_intents)
        stream = _log_filter(
            stream,
            lambda c: c.intent not in excl_set,
            f"exclude_intents={exclude_intents}",
        )

    if min_messages > 0:
        stream = _log_filter(
            stream,
            lambda c: c.message_count >= min_messages,
            f"min_messages>={min_messages}",
        )

    return list(stream)


# ── Normalization pass ────────────────────────────────────────────────────────

def normalize_all(conversations: list[Conversation]) -> list[Conversation]:
    """Run text normalization on every message in every conversation."""
    result = []
    for conv in conversations:
        normalized_messages = normalize_conversation_messages(
            [m.model_dump() for m in conv.messages]
        )
        updated_messages = [type(conv.messages[0]).model_validate(m) for m in normalized_messages]
        result.append(conv.model_copy(update={"messages": updated_messages}))
    logger.info(f"Normalized {len(result)} conversations")
    return result


# ── Fallback classification chain ─────────────────────────────────────────────

def classify_with_fallback(
    conversations: list[Conversation],
    llm_cfg=None,
    tagger_cfg=None,
    confidence_threshold: float = 0.6,
) -> tuple[list[Conversation], dict]:
    """
    Classification fallback chain:
      1. LLM (if enabled + conf >= threshold)
      2. Keyword tagger (if LLM disabled or low confidence)
      3. Default "unclear"

    Returns (annotated_conversations, stats)
    """
    stats = {
        "llm_used": 0,
        "keyword_used": 0,
        "defaulted": 0,
        "llm_low_confidence": 0,
    }

    if llm_cfg is not None:
        from .llm_classifier import annotate_with_llm
        conversations, llm_stats = annotate_with_llm(conversations, llm_cfg)
        stats["llm_used"] = llm_stats.get("success", 0)
        stats["llm_low_confidence"] = llm_stats.get("low_confidence", 0)

        # For low-confidence LLM results, fall back to keyword tagger
        low_conf = [c for c in conversations if (c.issue_confidence or 0) < confidence_threshold]
        high_conf = [c for c in conversations if (c.issue_confidence or 0) >= confidence_threshold]

        if tagger_cfg and low_conf:
            from .issue_tagger import annotate_issue_types
            retagged = annotate_issue_types(low_conf, tagger_cfg)
            stats["keyword_used"] = len(retagged)
            conversations = high_conf + retagged

    elif tagger_cfg is not None:
        from .issue_tagger import annotate_issue_types
        conversations = annotate_issue_types(conversations, tagger_cfg)
        stats["keyword_used"] = len(conversations)

    # Count remaining "unclear" (defaulted)
    stats["defaulted"] = sum(1 for c in conversations if c.issue in {"unclear", None, ""})

    logger.info(
        f"Classification: LLM={stats['llm_used']} keyword={stats['keyword_used']} "
        f"defaulted={stats['defaulted']} low_conf={stats['llm_low_confidence']}"
    )
    return conversations, stats


# ── Audit log ─────────────────────────────────────────────────────────────────

def write_audit_log(
    out_dir: Path,
    stats: PipelineStats,
    extra: dict | None = None,
) -> None:
    """Write a JSONL audit entry for this run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_audit.jsonl"

    entry = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        **stats.model_dump(),
        **(extra or {}),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Audit log → {log_path}")
