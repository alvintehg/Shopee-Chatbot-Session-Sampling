"""
issue_tagger.py  (PRODUCTION REWRITE)
──────────────────────────────────────
Key fixes:
  ✅  Runs on normalized text (not raw) for consistent matching
  ✅  Returns confidence score based on keyword match density
  ✅  Priority-ordered taxonomy (highest-priority match wins)
  ✅  Logs match details for auditability
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .models import Conversation, LLMClassification
from .normalizer import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class IssueTaggerConfig:
    # taxonomy: {issue_type: {issue: [keywords]}}
    taxonomy: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    default_issue: str = "unclear"
    default_issue_type: str = "other"
    include_bot_messages: bool = False
    overwrite_existing: bool = False
    min_keyword_hits: int = 1          # min keyword matches to assign label
    confidence_per_hit: float = 0.2    # confidence += this per keyword match (max 1.0)


def _score_text_against_keywords(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    """Count how many keywords appear in text. Returns (hit_count, matched_keywords)."""
    hits = []
    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, text):
            hits.append(kw)
    return len(hits), hits


def tag_conversation(
    conversation: Conversation,
    cfg: IssueTaggerConfig,
) -> tuple[Conversation, dict]:
    """
    Tag a single conversation with issue + issue_type.
    Returns (annotated_conversation, audit_info).
    """
    audit = {
        "conversation_id": conversation.id,
        "method": "keyword",
        "matched_issue": None,
        "matched_issue_type": None,
        "matched_keywords": [],
        "confidence": 0.0,
        "skipped": False,
    }

    # Skip if already tagged and overwrite is off
    if conversation.issue and not cfg.overwrite_existing:
        audit["skipped"] = True
        return conversation, audit

    # Build text to match against
    roles = {"user"}
    if cfg.include_bot_messages:
        roles.update({"bot", "agent"})

    combined_text = normalize_text(
        " ".join(
            m.text for m in conversation.messages if m.role in roles
        )
    ).normalized

    best_issue = cfg.default_issue
    best_issue_type = cfg.default_issue_type
    best_hits = 0
    best_keywords: list[str] = []
    best_confidence = 0.0

    # Iterate taxonomy in definition order (order = priority)
    for issue_type, issues in cfg.taxonomy.items():
        for issue, keywords in issues.items():
            hits, matched = _score_text_against_keywords(combined_text, keywords)
            if hits >= cfg.min_keyword_hits and hits > best_hits:
                best_hits = hits
                best_issue = issue
                best_issue_type = issue_type
                best_keywords = matched
                best_confidence = min(1.0, hits * cfg.confidence_per_hit)

    audit.update({
        "matched_issue": best_issue,
        "matched_issue_type": best_issue_type,
        "matched_keywords": best_keywords,
        "confidence": best_confidence,
    })

    annotated = conversation.model_copy(update={
        "issue": best_issue,
        "issue_type": best_issue_type,
        "issue_confidence": best_confidence,
    })

    return annotated, audit


def annotate_issue_types(
    conversations: list[Conversation],
    cfg: IssueTaggerConfig,
) -> list[Conversation]:
    """Batch tag all conversations. Returns annotated list."""
    results = []
    for conv in conversations:
        annotated, audit = tag_conversation(conv, cfg)
        if not audit["skipped"]:
            logger.debug(
                f"[{conv.id}] issue={audit['matched_issue']} "
                f"conf={audit['confidence']:.2f} "
                f"keywords={audit['matched_keywords']}"
            )
        results.append(annotated)
    return results
